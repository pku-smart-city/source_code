import copy
import itertools
from abc import ABC, abstractmethod

import numpy as np
import torch
from gym import spaces
from rl_agents.agents.common.abstract import AbstractStochasticAgent
from rl_agents.agents.common.differential_evolution import differential_evolution
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.memory import ReplayMemory, Transition


class AbstractDQNAgent(AbstractStochasticAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractDQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(
            env.action_space, spaces.Tuple
        ), "Only compatible with Discrete action spaces."
        self.memory = ReplayMemory(self.config)
        self.memory_total = ReplayMemory(self.config)
        self.memory_joint = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(
            self.config["exploration"], self.env.action_space
        )
        self.episode_count = 0
        self.training = True
        self.previous_state = None
        self.count = 0
        self.istrain = None

    @classmethod
    def default_config(cls):
        return dict(
            model=dict(type="DuelingNetwork"),
            optimizer=dict(type="ADAM", lr=5e-4, weight_decay=0, k=5),
            optimizer_total=dict(type="ADAM", lr=5e-3, weight_decay=0, k=5),
            loss_function="l2",
            memory_capacity=50000,
            batch_size=100,
            gamma=0.99,
            device="cuda:0",
            exploration=dict(method="EpsilonGreedy"),
            target_update=1,
            double=True,
        )

    def record_joint(self, state, action, rewards, next_state, done, info):
        state_joint = np.concatenate(state, axis=0)
        rewards_joint = np.sum(rewards)
        next_state_joint = np.concatenate(next_state, axis=0)
        self.memory_joint.push(
            state_joint, action, rewards_joint, next_state_joint, done, info
        )

        batch_joint = self.sample_jointbatch()
        if batch_joint:
            loss = self.compute_bellman_residual_joint(batch_joint)
            self.step_optimizer_joint(loss)
            self.update_target_joint_network()

    def record(self, state, action, rewards, next_state, done, info):

        if not self.training:
            return
        if isinstance(state, tuple) and isinstance(
            action, tuple
        ):  # Multi-agent setting
            [
                self.memory.push(
                    agent_state, agent_action, reward, agent_next_state, done, info
                )
                for agent_state, agent_action, reward, agent_next_state in zip(
                    state, action, rewards, next_state
                )
            ]

            # 将total更新所用元祖传入memory
            state_total = np.concatenate(state, axis=0)
            rewards_total = np.sum(rewards)
            next_state_total = np.concatenate(next_state, axis=0)
            self.memory_total.push(
                state_total, action, rewards_total, next_state_total, done, info
            )

        else:  # Single-agent setting
            self.memory.push(state, action, rewards, next_state, done, info)
        batch = self.sample_minibatch()
        batch_total = self.sample_minibatch_total()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            self.step_optimizer(loss)
            self.update_target_network()

        if batch_total:
            loss, _, _ = self.compute_bellman_residual_total(batch_total)
            self.step_optimizer_total(loss)
            lameda = self.adjust_lameda()
            loss_reg, _, _ = self.compute_bellman_residual_reg(batch_total)
            self.step_optimizer(lameda * loss_reg)

    def perturb_state(self, state, eps, worst_action_index):
        adv_state = self.get_adv_state([state], eps, worst_action_index)
        return adv_state

    def act(self, state, eps):
        step_exploration_time = True
        self.previous_state = state[-1]
        if step_exploration_time:
            self.exploration_policy.step_time()

        attack_budget = eps[-1]
        other_attacked_action = []
        best_act = self.get_agent_action(state)

        if attack_budget >= 0:
            state_ = torch.from_numpy(np.array(state)).reshape(-1)
            action_ = torch.from_numpy(np.array(best_act)).reshape(-1)
            state_joint_action = torch.cat((state_, action_), dim=-1)
            state_joint_value = self.get_state_joint_value(state_joint_action)

            _, agent_action_id = self.attack_q_de(
                img=action_,
                label=state_joint_value,
                actions=action_,
                n_agent=len(state),
                n_action=5,
                state_batch=state_,
                pixels=eps[1],
                maxiter=75,
                popsize=1000,
                verbose=False,
            )

            best_act = self.get_final_action(
                state, eps, copy.deepcopy(action_), agent_action_id
            )

        return tuple(best_act), other_attacked_action

    def get_agent_action(self, state):
        act_temp = []
        if isinstance(state, tuple):  # 返回一个智能体的动作turple（4,2,0）
            _ = 0
            for idx, agent_state in enumerate(state):
                urgency = self._urgency_judge(agent_state)
                values_origin = self.get_state_action_values(agent_state)
                _ += 1
                if urgency > 1:
                    self.exploration_policy.update(values_origin)
                    act_temp.append(self.exploration_policy.sample())
                else:
                    # 非紧急情况，将最优及次优动作列入选择
                    self.exploration_policy.update(values_origin)
                    act_temp.append(self.exploration_policy.sample_total())
            best_act = act_temp
            # 写如何处理有三个ndarray的list
            values = [list(arr) for arr in act_temp]
            combinations = list(itertools.product(*values))
            max_value = 0
            con_state = (
                np.concatenate(
                    (state[0], state[1], state[2], state[3], state[4], state[5]), axis=0
                ),
            )
            _ = 0
            value_total = self.get_state_action_values_total(con_state)[0]
            for combination in combinations:
                index = (
                    combination[0]
                    + combination[1] * 5
                    + combination[2] * 5**2
                    + combination[3] * 5**3
                    + combination[4] * 5**4
                    + combination[5] * 5**5
                )
                if _ == 0:
                    max_value = value_total[index]
                    best_act = combination
                if value_total[index] > max_value:
                    max_value = value_total[index]
                    best_act = combination
                    _ = 1
        return best_act

    def get_final_action(self, state, eps, raw_actions, agent_and_aciton):
        perturbed_state = ()
        for i in range(0, len(agent_and_aciton), 2):
            agent_id, action_id = agent_and_aciton[i], agent_and_aciton[i + 1]
            agent_state = state[agent_id]
            adv_agent_state = self.perturb_state(agent_state, eps[0], action_id)
            perturbed_state += (adv_agent_state,)

        return perturbed_state

    def _urgency_judge(self, single_state):
        'MQLC urgency judge'
        min = 0
        max_mean_v = 1
        max_traffic_d = 100
        max_speed_v = 0.05
        mean_v = (np.mean(np.sqrt(np.sum(single_state[:, 3:5] ** 2, axis=1))) - min) / (
            max_mean_v - min
        )
        traffic_d = (
            np.max(np.abs(single_state[:, 1][:, np.newaxis] - single_state[:, 1])) - min
        ) / (max_traffic_d - min)
        speed_v = (
            np.var(np.sqrt(single_state[:, 3] ** 2 + single_state[:, 4] ** 2)) - min
        ) / (max_speed_v - min)

        return mean_v + traffic_d + 2 * speed_v

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def sample_jointbatch(self):
        if len(self.memory_joint) < self.config["batch_size"]:
            return None
        transitions = self.memory_joint.sample_joint(self.config["batch_size"])
        return Transition(*zip(*transitions)) if transitions is not None else None

    def sample_minibatch_total(self):
        if len(self.memory_total) < self.config["batch_size"]:
            return None
        transitions = self.memory_total.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
            self.target_net_total.load_state_dict(self.value_net_total.state_dict())

    def update_target_joint_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_joint_action_value_net.load_state_dict(
                self.joint_action_value_net.state_dict()
            )

    def adjust_lameda(self):
        explameda = 0.1
        adjust_episodes = 1000
        self.episode_count += 1
        return (
            self.episode_count
            // adjust_episodes
            * (explameda / (10000 / adjust_episodes))
        )

    @abstractmethod
    def compute_bellman_residual(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    def compute_bellman_residual_other(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    def compute_bellman_residual_global(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    def compute_bellman_residual_total(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    def compute_bellman_residual_reg(self, batch, target_state_action_value=None):
        """
            Compute the Bellman Residual Loss over a batch
        :param batch: batch of transitions
        :param target_state_action_value: if provided, acts as a target (s,a)-value
                                          if not, it will be computed from batch and model (Double DQN target)
        :return: the loss over the batch, and the computed target
        """
        raise NotImplementedError

    def compute_bellman_residual_joint(self, batch_joint):
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_values(self, states):
        """
        Get the state values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values, actions:
                 - [V1; ...; VN] the array of the state values for each state
                 - [a1*; ...; aN*] the array of corresponding optimal action indexes for each state
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_state_action_values(self, states):
        """
        Get the state-action values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values:[[Q11, ..., Q1n]; ...] the array of all action values for each state
        """
        raise NotImplementedError

    @abstractmethod
    def get_adv_state(self, state, eps, worst_action_index):

        raise NotImplementedError

    def get_batch_total_effects_values(self, states):
        """ """
        raise NotImplementedError

    # Qorigin计算影响
    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def get_state_joint_value(self, state_joint_action):
        raise NotImplementedError

    def get_state_action_values_total(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions 这里是125维的联合动作估计值
        """
        return self.get_batch_total_effects_values([state])[0]

    def step_optimizer(self, loss):
        raise NotImplementedError

    def step_optimizer_total(self, loss):
        raise NotImplementedError

    def step_optimizer_joint(self, loss):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def set_writer(self, writer):
        super().set_writer(writer)
        try:
            self.exploration_policy.set_writer(writer)
        except AttributeError:
            pass

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "Greedy"
        self.exploration_policy = exploration_factory(
            self.config["exploration"], self.env.action_space
        )

    def perturb_actions(
        self, xs, actions
    ):  # xs:[x, y] x为智能体id y为目标动作  img: 当前的动作
        # If this function is passed just one perturbation vector,
        # pack it in a list to keep the computation the same
        if xs.ndim < 2:
            xs = np.array([xs])
        batch = len(xs)
        actions = actions.repeat(batch, 1, 1)
        xs = xs.astype(int)

        count = 0
        for x in xs:
            pixels = np.split(x, len(x) / 2)
            # print(pixels)
            for pixel in pixels:
                agent_id, action = pixel.astype(int)
                actions[count, 0, agent_id] = action
            count += 1
        # actions[agents_available_actions == 0] = 10.0
        return actions

    def predict_classes(self, xs, img, state_batch):
        imgs_perturbed = self.perturb_actions(xs, img.clone())
        batch_size = imgs_perturbed.shape[0]
        # 将状态和扰动后的动作拼接为 state_joint_action
        state_batch_tensor = torch.FloatTensor(state_batch).repeat(batch_size, 1)
        state_joint_action = torch.cat(
            [state_batch_tensor, imgs_perturbed.squeeze(1)], dim=1
        )
        # 计算联合状态价值
        predictions = self.get_state_joint_value(state_joint_action)

        return predictions

    def attack_success(self, x, img, target_class, state_batch, verbose=False):
        attack_image = self.perturb_actions(x, img.clone())
        # 拼接状态和扰动后的动作
        state_joint_action = torch.cat(
            [state_batch, attack_image.squeeze(0).squeeze(0)], dim=-1
        )
        # 计算扰动后的Q值
        q_tot = self.get_state_joint_value(state_joint_action).item()

        if verbose:
            print(f"Original Q: {target_class.item()}, Perturbed Q: {q_tot}")
        return q_tot < target_class.item()  # 成功条件：扰动后Q值降低

    # img: 动作
    # label： 真实的qmix_value
    # learner: qmix 网络
    # pixels：被攻击智能体的数量
    def attack_q_de(
        self,
        img,
        label,
        actions,
        n_agent,
        n_action,
        state_batch,
        pixels=1,
        maxiter=75,
        popsize=400,
        verbose=False,
    ):
        target_calss = label
        # print(target_calss)
        # print(agents_available_actions)
        bounds = [(0, n_agent - 1), (0, n_action - 1)] * pixels  # len(bounds) = 5
        popmul = max(1, popsize // len(bounds))

        def predict_fn(xs):
            return self.predict_classes(xs, img, state_batch)  # 要最小化的目标函数

        def callback_fn(x, convergence):
            return self.attack_success(x, img, target_calss, state_batch, verbose)

        # callback_fn = None
        inits = np.zeros([popmul * len(bounds), len(bounds)])
        for init in inits:  # 随机初始化
            for i in range(pixels):
                init[i * 2 + 0] = np.random.randint(0, n_agent - 1)

                init[i * 2 + 1] = np.random.randint(0, n_action - 1)

        attack_result = differential_evolution(
            predict_fn,
            bounds,
            maxiter=maxiter,
            popsize=popmul,
            recombination=1,
            atol=-1,
            callback=callback_fn,
            polish=False,
            init=inits,
        )

        attack_image = self.perturb_actions(attack_result.x.astype(int), img.clone())

        return attack_image[0, 0].data.cpu().numpy().tolist(), attack_result.x.astype(
            int
        )

# 返修 补充

# import copy
# import time
# from abc import ABC, abstractmethod
# from collections import defaultdict
# import numpy as np
# from gym import spaces
# from torch.autograd import Variable
# from rl_agents.agents.common.abstract import AbstractStochasticAgent
# from rl_agents.agents.common.exploration.abstract import exploration_factory
# from rl_agents.agents.common.memory import ReplayMemory, Transition
# from rl_agents.agents.common.differential_evolution import differential_evolution
# import torch
# import random
# import itertools
# import math
# import contextlib
# import hashlib
# import json


# def decode_candidate(z, combos, n_action, k):
#     z = np.asarray(z, dtype=float).reshape(-1)
#     if z.size != k + 1 or not np.isfinite(z).all():
#         raise ValueError("Malformed DE candidate")
#     subset = min(max(int(np.floor(z[0])), 0), len(combos) - 1)
#     result = []
#     for agent, value in zip(combos[subset], z[1:]):
#         result.extend((int(agent), min(max(int(np.floor(value)), 0), n_action - 1)))
#     return result

# def enumerate_candidates(n_agent, n_action, k):
#     return np.asarray([
#         [value for pair in zip(agents, actions) for value in pair]
#         for agents in itertools.combinations(range(n_agent), k)
#         for actions in itertools.product(range(n_action), repeat=k)
#     ], dtype=np.int64)

# METHODS = ("exh", "de")


# def _integer(value, name, minimum=0):
#     if isinstance(value, (bool, np.bool_)):
#         raise ValueError(name + " must be an integer")
#     try:
#         result = int(value)
#     except (TypeError, ValueError, OverflowError):
#         raise ValueError(name + " must be an integer") from None
#     if result != value or result < minimum:
#         raise ValueError(name + " must be an integer >= " + str(minimum))
#     return result


# def _mutation_metadata(mutation):
#     values = np.asarray(mutation, dtype=float)
#     if (values.ndim > 1 or values.size not in (1, 2) or
#             not np.isfinite(values).all() or (values < 0).any() or
#             (values >= 2).any()):
#         raise ValueError("Mutation must be finite in [0, 2), or a pair in that range")
#     if values.ndim == 0:
#         return float(values)
#     if values.size != 2 or values[0] >= values[1]:
#         raise ValueError("Mutation dither needs an increasing pair")
#     return [float(x) for x in values]


# def _require_original_de_settings(population, maxiter, strategy="best1bin",
#                                   mutation=(.5, 1), recombination=1):
#     """Reject alternate budgets/settings instead of silently relabelling them."""
#     population = _integer(population, "population", 5)
#     maxiter = _integer(maxiter, "maxiter", 1)
#     if population != 1000 or maxiter != 75:
#         raise ValueError("This public DE entry requires actual P=1000 and maxiter=75")
#     if (strategy != "best1bin" or _mutation_metadata(mutation) != [.5, 1.0] or
#             isinstance(recombination, (bool, np.bool_)) or float(recombination) != 1):
#         raise ValueError("DE requires best1bin, mutation=(0.5,1.0), recombination=1")
#     return population, maxiter


# class _PipelineScorer:
#     """Group only when caching is on; count every requested candidate row."""

#     def __init__(self, score_one, n_agent, n_action, k, cache_enabled, synchronize):
#         self.score_one = score_one
#         self.n_agent, self.n_action, self.k = n_agent, n_action, k
#         self.cache_enabled = cache_enabled
#         self.synchronize = synchronize
#         self.cache = {}
#         self.seen = set()
#         self.optimizer_rows = self.callback_rows = self.requests = 0
#         self.pipeline_evaluations = self.cache_hits = self.batch_duplicate_rows = 0
#         self.fitness_batches = 0
#         self.pipeline_time_s = 0.0

#     def _key(self, row):
#         if len(row) != self.k * 2:
#             raise ValueError("Candidate must contain exactly k agent/action pairs")
#         values = [_integer(x, "candidate coordinate") for x in row]
#         pairs = list(zip(values[::2], values[1::2]))
#         if (len({a for a, _ in pairs}) != self.k or
#                 any(a >= self.n_agent or c >= self.n_action for a, c in pairs)):
#             raise ValueError("Candidate must select k distinct legal agents/actions")
#         return tuple(x for pair in sorted(pairs) for x in pair)

#     def _physical(self, key):
#         self.synchronize()
#         started = time.perf_counter()
#         value = np.asarray(self.score_one(list(key)))
#         self.synchronize()
#         self.pipeline_time_s += time.perf_counter() - started
#         if value.ndim != 0:
#             raise ValueError("score_one must return one finite scalar Q")
#         try:
#             q = float(value)
#         except (TypeError, ValueError, OverflowError):
#             raise ValueError("score_one must return one finite scalar Q") from None
#         if not np.isfinite(q):
#             raise ValueError("score_one must return one finite scalar Q")
#         self.pipeline_evaluations += 1
#         return q

#     def evaluate(self, candidates, callback=False):
#         xs = np.asarray(candidates)
#         if xs.ndim != 2 or len(xs) == 0:
#             raise ValueError("Expected a nonempty 2D interleaved candidate array")
#         keys = [self._key(row) for row in xs]
#         self.fitness_batches += 1
#         self.requests += len(keys)
#         if callback:
#             self.callback_rows += len(keys)
#         else:
#             self.optimizer_rows += len(keys)
#         self.seen.update(keys)
#         if not self.cache_enabled:
#             return np.asarray([self._physical(key) for key in keys], dtype=np.float64)

#         groups = {}
#         for index, key in enumerate(keys):
#             groups.setdefault(key, []).append(index)
#         self.batch_duplicate_rows += len(keys) - len(groups)
#         values = np.empty(len(keys), dtype=np.float64)
#         for key, indices in groups.items():
#             if key in self.cache:
#                 self.cache_hits += 1
#                 q = self.cache[key]
#             else:
#                 q = self._physical(key)
#                 self.cache[key] = q
#             values[indices] = q
#         return values


# def run_full_search(method, n_agent, n_action, k, score_one, baseline_q,
#                     algorithm_seed, population=None, maxiter=None,
#                     strategy="best1bin", mutation=(.5, 1), recombination=1,
#                     solver=None, synchronize=None, early_stop=True,
#                     input_hash="", cache_enabled=True):
#     """Search one frozen input; score_one executes the complete perturbation/Q.

#     Each invocation owns an empty cache.  NFEV counts optimizer rows, while
#     fitness_evaluations also counts callback requests.  With caching enabled,
#     requests = pipeline_evaluations + cache_hits + batch_duplicate_rows.
#     With caching disabled every row executes score_one, including duplicates.
#     Baseline evaluation, warmup, loading and final environment execution belong
#     to the caller and are outside this search timer.
#     """
#     if method not in METHODS:
#         raise ValueError("Unknown full-pipeline method: " + str(method))
#     if method == "de":
#         _require_original_de_settings(population, maxiter, strategy, mutation, recombination)
#     n_agent = _integer(n_agent, "n_agent", 1)
#     n_action = _integer(n_action, "n_action", 1)
#     k = _integer(k, "k", 1)
#     if k > n_agent or not callable(score_one):
#         raise ValueError("Invalid candidate dimensions or scorer")
#     algorithm_seed = _integer(algorithm_seed, "algorithm_seed")
#     if algorithm_seed >= 2 ** 32:
#         raise ValueError("algorithm_seed must be uint32")
#     baseline_q = float(baseline_q)
#     if not np.isfinite(baseline_q):
#         raise ValueError("Nonfinite baseline Q")
#     if not isinstance(cache_enabled, (bool, np.bool_)):
#         raise ValueError("cache_enabled must be boolean")
#     cache_enabled = bool(cache_enabled)
#     mutation_record = None
#     if method != "exh":
#         if not callable(solver):
#             raise ValueError("DE requires the project's batch solver")
#         population = _integer(population, "population", 5)
#         maxiter = _integer(maxiter, "maxiter", 1)
#         if not isinstance(strategy, str) or not strategy:
#             raise ValueError("DE strategy must be a nonempty string")
#         mutation_record = _mutation_metadata(mutation)
#         recombination = float(recombination)
#         if not np.isfinite(recombination) or not 0 <= recombination <= 1:
#             raise ValueError("Recombination must be in [0, 1]")
#     synchronize = synchronize or (lambda: None)
#     synchronize()
#     started = time.perf_counter()
#     rng = np.random.RandomState(algorithm_seed)
#     scorer = _PipelineScorer(score_one, n_agent, n_action, k, cache_enabled, synchronize)
#     combos = list(itertools.combinations(range(n_agent), k))
#     space_size = math.comb(n_agent, k) * n_action ** k
#     stop_requested = False
#     if method == "exh":
#         candidates = enumerate_candidates(n_agent, n_action, k)
#         qs = scorer.evaluate(candidates)
#         optima = np.flatnonzero(qs == float(np.min(qs)))
#         selected = int(rng.choice(optima))
#         best, best_q = candidates[selected].tolist(), float(qs[selected])
#         nfev, nit, population, maxiter = len(candidates), 0, None, None
#         stop_reason, tie_set_size = "all_candidates_enumerated", int(len(optima))
#     else:
#         dim = k + 1
#         bounds = [(0, len(combos))] + [(0, n_action)] * k
#         init = np.empty((population, dim), dtype=np.float64)
#         init[:, 0] = rng.randint(0, len(combos), size=population)
#         init[:, 1:] = rng.randint(0, n_action, size=(population, k))

#         def objective(zs):
#             xs = np.asarray([decode_candidate(z, combos, n_action, k)
#                              for z in np.atleast_2d(zs)], dtype=np.int64)
#             return scorer.evaluate(xs)

#         def callback(z, convergence):
#             nonlocal stop_requested
#             if not early_stop:
#                 return False
#             candidate = decode_candidate(z, combos, n_action, k)
#             q = float(scorer.evaluate([candidate], callback=True)[0])
#             stop_requested = q < baseline_q
#             return stop_requested

#         result = solver(objective, bounds, maxiter=maxiter,
#                         popsize=max(1, population // dim), strategy=strategy,
#                         mutation=mutation, recombination=recombination,
#                         atol=-1, callback=callback, polish=False, init=init, seed=rng)
#         best = decode_candidate(result.x, combos, n_action, k)
#         best_q = float(np.asarray(result.fun).reshape(-1)[0])
#         nfev, nit = int(result.nfev), int(result.nit)
#         if (nfev != scorer.optimizer_rows or nfev != population * (1 + nit) or
#                 not 1 <= nit <= maxiter):
#             raise RuntimeError("Project solver candidate-row budget contract changed")
#         stop_reason = ("baseline_improved" if stop_requested else
#                        "maxiter" if nit >= maxiter else "solver_convergence")
#         tie_set_size = None
#     if not np.isfinite(best_q):
#         raise ValueError("Nonfinite selected fitness")
#     if (scorer.requests != nfev + scorer.callback_rows or
#             scorer.requests != scorer.pipeline_evaluations + scorer.cache_hits +
#             scorer.batch_duplicate_rows):
#         raise RuntimeError("Full-pipeline request/physical-work accounting changed")
#     synchronize()
#     elapsed = time.perf_counter() - started
#     return {
#         "method": method, "input_hash": str(input_hash), "fitness_mode": "full_pipeline",
#         "candidate_space": "distinct_agents_allow_original_action",
#         "k": k, "n_agent": n_agent, "n_action": n_action, "space_size": int(space_size),
#         "best_q": best_q, "agent_action_id": [int(x) for x in best],
#         "nfev": nfev, "nit": nit, "population_size": population, "maxiter": maxiter,
#         "callback_evaluations": int(scorer.callback_rows),
#         "fitness_evaluations": int(scorer.requests),
#         "pipeline_evaluations": int(scorer.pipeline_evaluations),
#         "cache_hits": int(scorer.cache_hits),
#         "batch_duplicate_rows": int(scorer.batch_duplicate_rows),
#         "unique_candidates": len(scorer.seen), "fitness_batches": scorer.fitness_batches,
#         "cache_enabled": cache_enabled, "pipeline_time_s": float(scorer.pipeline_time_s),
#         "search_time_s": float(elapsed), "stop_reason": stop_reason,
#         "algorithm_seed": algorithm_seed, "tie_set_size": tie_set_size,
#         "early_stop_enabled": bool(early_stop) if method != "exh" else False,
#         "strategy": strategy if method != "exh" else None,
#         "mutation": mutation_record, "recombination": recombination if method != "exh" else None,
#     }

# def canonical(value):
#     return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

# def digest(value):
#     return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

# def derived_seed(base, *parts):
#     return int(digest([int(base), *parts])[:8], 16)

# @contextlib.contextmanager
# def isolated_rng(np, torch, seed, agent=None):
#     """Restore globals and the private exploration RNG after each search/attack."""
#     py_state, np_state = random.getstate(), np.random.get_state()
#     torch_state = torch.random.get_rng_state()
#     cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
#     policy_rng = getattr(getattr(agent, "exploration_policy", None), "np_random", None)
#     private = None
#     if policy_rng is not None:
#         if hasattr(policy_rng, "get_state"):
#             private = ("legacy", policy_rng.get_state())
#         elif hasattr(policy_rng, "bit_generator"):
#             private = ("generator", copy.deepcopy(policy_rng.bit_generator.state))
#     try:
#         random.seed(seed)
#         np.random.seed(int(seed) % (2 ** 32))
#         torch.manual_seed(int(seed))
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(int(seed))
#         yield
#     finally:
#         random.setstate(py_state)
#         np.random.set_state(np_state)
#         torch.random.set_rng_state(torch_state)
#         if cuda_states is not None:
#             torch.cuda.set_rng_state_all(cuda_states)
#         if private is not None:
#             if private[0] == "legacy":
#                 policy_rng.set_state(private[1])
#             else:
#                 policy_rng.bit_generator.state = private[1]


# class AbstractDQNAgent(AbstractStochasticAgent, ABC):

#     def __init__(self, env, config=None):
#         super(AbstractDQNAgent, self).__init__(config)
#         self.env = env
#         assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
#             "Only compatible with Discrete action spaces."
#         self.memory = ReplayMemory(self.config)
#         self.memory_total = ReplayMemory(self.config)
#         self.memory_joint = ReplayMemory(self.config)
#         self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
#         self.episode_count = 0
#         self.training = True
#         self.previous_state = None
#         self.attack = None
#         self.count = 0
#         self.stn = 0
#         self.istrain = None
#         # RN 攻击：对 DE 选中的智能体状态加随机噪声，不做梯度扰动
#         self.use_random_noise = config.get('use_random_noise', False)
#         # use Q_est^tot (joint_action_value_net) instead of Q_tot for joint action selection
#         self.use_joint_policy = False
#         self.agent_select_mode = self.config.get("agent_select_mode", "lacat")
#         self.last_attack_search = None
#         self.last_paired_audit = None
#         self.attack_fitness_mode = self.config.get("attack_fitness_mode", "action_only")
#         self.fitness_cache_enabled = self.config.get("fitness_cache_enabled", True)
#         self.attack_base_seed = self.config.get("attack_base_seed", self.config.get("base_seed", 100000))
#         _require_original_de_settings(self.config.get("de_population", 1000),
#                                       self.config.get("de_maxiter", 75))

#     @classmethod
#     def default_config(cls):
#         return dict(model=dict(type="DuelingNetwork"),
#                     optimizer=dict(type="ADAM",
#                                    lr=5e-4,
#                                    weight_decay=0,
#                                    k=5),
#                     optimizer_total=dict(type="ADAM",
#                                    lr=5e-3,
#                                    weight_decay=0,
#                                    k=5),
#                     loss_function="l2",
#                     memory_capacity=50000,
#                     batch_size=100,
#                     gamma=0.99,
#                     device="cuda:0",
#                     exploration=dict(method="EpsilonGreedy"),
#                     target_update=1,
#                     double=True)

#     def record_joint(self, state, action, rewards, next_state, done, info):
#         state_joint = np.concatenate(state, axis=0)
#         rewards_joint = np.sum(rewards)
#         next_state_joint = np.concatenate(next_state, axis=0)
#         self.memory_joint.push(state_joint, action, rewards_joint, next_state_joint, done, info)

#         batch_joint = self.sample_jointbatch()
#         if batch_joint:
#             loss = self.compute_bellman_residual_joint(batch_joint)
#             self.step_optimizer_joint(loss)
#             self.update_target_joint_network()

#     def record(self, state, action, rewards, next_state, done, info):

#         if not self.training:
#             return
#         if isinstance(state, tuple) and isinstance(action, tuple):  # Multi-agent setting
#             [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
#             for agent_state, agent_action, reward, agent_next_state in zip(state, action, rewards, next_state)]

#             # 将total更新所用元祖传入memory
#             state_total = np.concatenate(state, axis=0)
#             rewards_total = np.sum(rewards)
#             next_state_total = np.concatenate(next_state, axis=0)
#             self.memory_total.push(state_total, action, rewards_total, next_state_total, done, info)

#         else:  # Single-agent setting
#             self.memory.push(state, action, rewards, next_state, done, info)
#         batch = self.sample_minibatch()
#         batch_total = self.sample_minibatch_total()
#         if batch:
#             loss, _, _ = self.compute_bellman_residual(batch)
#             self.step_optimizer(loss)
#             self.update_target_network()

#         if batch_total:
#             loss, _, _ = self.compute_bellman_residual_total(batch_total)
#             self.step_optimizer_total(loss)
#             lameda = self.adjust_lameda()
#             loss_reg, _, _ = self.compute_bellman_residual_reg(batch_total)
#             self.step_optimizer(lameda * loss_reg)

#     def get_worst_joint_action_value(self, states):
#         combine_states = (np.concatenate((states[0], states[1], states[2]), axis=0), )
#         value_total = self.get_state_action_values_total(combine_states)[0]
#         worst_joint_action = np.argmin(value_total)
#         value_total_softmax =np.exp(value_total - np.max(value_total)) / np.sum(np.exp(value_total - np.max(value_total)))
#         return torch.tensor(value_total_softmax[worst_joint_action], dtype=torch.float32)

#     def perturb_state(self, state, eps, worst_action_index):
#         adv_state = self.get_adv_state([state], eps, worst_action_index)
#         return adv_state

#     def act(self):
#         pass

#     def act1(self, state, eps):
#         # Legacy main uses >=0; the explicit Table10 mode uses remaining >0.
#         self.previous_state = state[-1]
#         self.exploration_policy.step_time()
#         attack_budget = eps[-1]
#         other_attacked_action = []
#         best_act = self.get_agents_actions(state)
#         self.last_attack_search = None
#         self.last_paired_audit = None
#         fitness_mode = self._resolve_attack_fitness_mode(None)
#         allowed = attack_budget >= 0 if fitness_mode == "action_only" else attack_budget > 0
#         if fitness_mode == "full_pipeline":
#             self._require_full_pipeline_setup()
#         if allowed:
#             state_ = torch.from_numpy(np.array(state)).reshape(-1)
#             action_ = torch.from_numpy(np.array(best_act)).reshape(-1)
#             with torch.no_grad():
#                 clean_q = (self.get_state_joint_value(torch.cat((state_, action_), dim=-1))
#                            if fitness_mode == "action_only" else self._full_clean_baseline(state, best_act))
#             label = torch.tensor(self._search_scalar(clean_q, "clean Q"), dtype=torch.float64)
#             n_action = int(self.env.action_space.spaces[0].n) if isinstance(
#                 self.env.action_space, spaces.Tuple) else int(self.env.action_space.n)
#             common = dict(img=action_, label=label, actions=action_, n_agent=len(state),
#                           n_action=n_action, state_batch=state_, pixels=eps[1], fitness_mode=fitness_mode)
#             if fitness_mode == "full_pipeline":
#                 common.update(state_list=state, eps=eps,
#                               input_hash=self._full_input_hash(state, best_act, eps[1]))
#             mode = self.agent_select_mode
#             if mode in ("lacat", "de"):
#                 _, agent_action_id, stats = self.attack_q_de(**common,
#                     maxiter=self.config.get("de_maxiter", 75),
#                     popsize=self.config.get("de_population", 1000), return_stats=True)
#                 self.last_attack_search = stats
#             elif mode in ("exhaustive", "exh"):
#                 _, agent_action_id, stats = self.attack_q_exhaustive(**common)
#                 self.last_attack_search = stats
#             elif mode == "paired_audit":
#                 agent_action_id, audit = self.paired_search_audit(**common,
#                     maxiter=self.config.get("de_maxiter", 75),
#                     popsize=self.config.get("de_population", 1000),
#                     execute_as=self.config.get("paired_audit_execute", "de"))
#                 self.last_paired_audit = audit
#                 self.last_attack_search = audit["de"] if audit["execute_as"] == "de" else audit["exhaustive"]
#                 self.last_attack_search["paired_audit_performed"] = True
#                 self.last_attack_search["paired_total_search_time_s"] = audit["paired_total_search_time_s"]
#             else:
#                 raise ValueError("agent_select_mode must be lacat/de, exhaustive/exh, or paired_audit")
#             if fitness_mode == "action_only":
#                 best_act = self.get_final_action(list(state), eps, agent_action_id)
#             else:
#                 self._synchronize_attack_search()
#                 execution_started = time.perf_counter()
#                 best_act, execution_q = self.execute_full_pipeline_candidate(
#                     state, tuple(action_.cpu().numpy().tolist()), int(eps[1]),
#                     agent_action_id, common["input_hash"], eps)
#                 self._synchronize_attack_search()
#                 stats = self.last_attack_search
#                 stats.update(execution_time_s=time.perf_counter()-execution_started,
#                     execution_pipeline_evaluations=1, execution_q=execution_q,
#                     executed_action=list(best_act))
#                 if abs(execution_q-stats["best_q"]) > 1e-7:
#                     raise RuntimeError("Selected execution Q disagrees with full-pipeline search Q")
#                 stats["full_pipeline_execution_verified"] = True
#         return tuple(best_act), other_attacked_action

#     def get_agents_actions(self, state):
#         act_temp = []
#         if isinstance(state, tuple):  # 返回一个智能体的动作turple（4,2,0）
#             _ = 0
#             for idx, agent_state in enumerate(state):
#                 urgency = self._urgency_judge(agent_state)
#                 values_origin = self.get_state_action_values(agent_state)
#                 _ += 1
#                 if urgency > 1:
#                     self.exploration_policy.update(values_origin)
#                     act_temp.append(self.exploration_policy.sample())
#                 else:
#                     # 非紧急情况，将最优及次优动作列入选择
#                     self.exploration_policy.update(values_origin)
#                     act_temp.append(self.exploration_policy.sample_total())
#             best_act = act_temp
#             # 写如何处理有三个ndarray的list
#             values = [list(arr) for arr in act_temp]
#             combinations = list(itertools.product(*values))
#             max_value = 0
#             con_state = (
#                 np.concatenate(
#                     (state[0], state[1], state[2], state[3], state[4], state[5]), axis=0
#                 ),
#             )
#             _ = 0
#             value_total = self.get_state_action_values_total(con_state)[0]
#             for combination in combinations:
#                 index = (
#                     combination[0]
#                     + combination[1] * 5
#                     + combination[2] * 5**2
#                     + combination[3] * 5**3
#                     + combination[4] * 5**4
#                     + combination[5] * 5**5
#                 )
#                 if _ == 0:
#                     max_value = value_total[index]
#                     best_act = combination
#                 if value_total[index] > max_value:
#                     max_value = value_total[index]
#                     best_act = combination
#                     _ = 1
#         return best_act

#     def get_final_action(self, state, eps, agent_and_aciton, use_random_noise=False):
#         """
#         use_random_noise: 若为 True（RN 攻击），对对应智能体状态加 eps[0] 的随机噪声，不做梯度扰动。
#         """
#         for i in range(0, len(agent_and_aciton), 2):
#             agent_id, action_id = agent_and_aciton[i], agent_and_aciton[i + 1]
#             agent_state = state[agent_id]
#             if use_random_noise:
#                 # RN: 只加 eps[0] 的随机噪声，不做梯度扰动
#                 noise = np.random.uniform(-1.0, 1.0, size=agent_state.shape)
#                 adv_agent_state = np.clip(
#                     agent_state.astype(np.float64) + eps[0] * noise,
#                     0.0, 1.0
#                 ).astype(agent_state.dtype)
#             else:
#                 adv_agent_state = self.perturb_state(agent_state, eps[0], action_id)
#             state[agent_id] = adv_agent_state

#         best_act = self.get_agents_actions(tuple(state))

#         return best_act

#     def _urgency_judge(self, single_state):
#         min = 0
#         max_mean_v = 1
#         max_traffic_d = 100
#         max_speed_v = 0.05
#         mean_v = (np.mean(np.sqrt(np.sum(single_state[:, 3:5] ** 2, axis=1))) - min) / (max_mean_v - min)
#         traffic_d = (np.max(np.abs(single_state[:, 1][:, np.newaxis] - single_state[:, 1])) - min) / (max_traffic_d - min)
#         speed_v = (np.var(np.sqrt(single_state[:, 3] ** 2 + single_state[:, 4] ** 2)) - min) / (max_speed_v - min)

#         return mean_v + traffic_d + 2*speed_v

#     def _get_scenario_risk(self, single_state):
#         x = torch.tensor(single_state)
#         points = x[:, 1:3]
#         distances = torch.norm(points[:, None] - points, dim=2)
#         threshold = 0.3
#         adj_matrix = (distances < threshold).float()

#         rowsum = torch.sum(adj_matrix, dim=1)
#         D = torch.diag(1.0 / torch.sqrt(rowsum))
#         normalized_adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)
#         f = torch.matmul(normalized_adj_matrix, x)

#         frobenius_norm = torch.norm(f, p='fro')
#         # e = torch.sigmoid(frobenius_norm)
#         e = float(frobenius_norm)
#         return e

#     def sample_minibatch(self):
#         if len(self.memory) < self.config["batch_size"]:
#             return None
#         transitions = self.memory.sample(self.config["batch_size"])
#         return Transition(*zip(*transitions))

#     def sample_jointbatch(self):
#         if len(self.memory_joint) < self.config["batch_size"]:
#             return None
#         transitions = self.memory_joint.sample_joint(self.config["batch_size"])
#         return Transition(*zip(*transitions)) if transitions is not None else None

#     def sample_minibatch_total(self):
#         if len(self.memory_total) < self.config["batch_size"]:
#             return None
#         transitions = self.memory_total.sample(self.config["batch_size"])
#         return Transition(*zip(*transitions))

#     def update_target_network(self):
#         self.steps += 1
#         if self.steps % self.config["target_update"] == 0:
#             self.target_net.load_state_dict(self.value_net.state_dict())
#             self.target_net_total.load_state_dict(self.value_net_total.state_dict())

#     def update_target_joint_network(self):
#         self.steps += 1
#         if self.steps % self.config["target_update"] == 0:
#             self.target_joint_action_value_net.load_state_dict(self.joint_action_value_net.state_dict())

#     def adjust_lameda(self):
#         explameda = 0.1
#         adjust_episodes = 1000
#         self.episode_count += 1
#         return self.episode_count//adjust_episodes * (explameda/(10000/adjust_episodes))

#     @abstractmethod
#     def compute_bellman_residual(self, batch, target_state_action_value=None):
#         """
#             Compute the Bellman Residual Loss over a batch
#         :param batch: batch of transitions
#         :param target_state_action_value: if provided, acts as a target (s,a)-value
#                                           if not, it will be computed from batch and model (Double DQN target)
#         :return: the loss over the batch, and the computed target
#         """
#         raise NotImplementedError

#     def compute_bellman_residual_other(self, batch, target_state_action_value=None):
#         """
#             Compute the Bellman Residual Loss over a batch
#         :param batch: batch of transitions
#         :param target_state_action_value: if provided, acts as a target (s,a)-value
#                                           if not, it will be computed from batch and model (Double DQN target)
#         :return: the loss over the batch, and the computed target
#         """
#         raise NotImplementedError

#     def compute_bellman_residual_global(self, batch, target_state_action_value=None):
#         """
#             Compute the Bellman Residual Loss over a batch
#         :param batch: batch of transitions
#         :param target_state_action_value: if provided, acts as a target (s,a)-value
#                                           if not, it will be computed from batch and model (Double DQN target)
#         :return: the loss over the batch, and the computed target
#         """
#         raise NotImplementedError

#     def compute_bellman_residual_total(self, batch, target_state_action_value=None):
#         """
#             Compute the Bellman Residual Loss over a batch
#         :param batch: batch of transitions
#         :param target_state_action_value: if provided, acts as a target (s,a)-value
#                                           if not, it will be computed from batch and model (Double DQN target)
#         :return: the loss over the batch, and the computed target
#         """
#         raise NotImplementedError

#     def compute_bellman_residual_reg(self, batch, target_state_action_value=None):
#         """
#             Compute the Bellman Residual Loss over a batch
#         :param batch: batch of transitions
#         :param target_state_action_value: if provided, acts as a target (s,a)-value
#                                           if not, it will be computed from batch and model (Double DQN target)
#         :return: the loss over the batch, and the computed target
#         """
#         raise NotImplementedError

#     def compute_bellman_residual_joint(self, batch_joint):
#         raise NotImplementedError

#     @abstractmethod
#     def get_batch_state_values(self, states):
#         """
#         Get the state values of several states
#         :param states: [s1; ...; sN] an array of states
#         :return: values, actions:
#                  - [V1; ...; VN] the array of the state values for each state
#                  - [a1*; ...; aN*] the array of corresponding optimal action indexes for each state
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def get_batch_state_action_values(self, states):
#         """
#         Get the state-action values of several states
#         :param states: [s1; ...; sN] an array of states
#         :return: values:[[Q11, ..., Q1n]; ...] the array of all action values for each state
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def get_adv_state(self, state, eps, worst_action_index):

#         raise NotImplementedError

#     def get_batch_total_effects_values(self, states):
#         """
#         """
#         raise NotImplementedError

#     def get_state_value(self, state):
#         """
#         :param state: s, an environment state
#         :return: V, its state-value
#         """
#         values, actions = self.get_batch_state_values([state])
#         return values[0], actions[0]

#     def get_state_action_values(self, state):
#         """
#         :param state: s, an environment state
#         :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
#         """
#         return self.get_batch_state_action_values([state])[0]

#     def get_state_joint_value(self, state_joint_action):
#         raise NotImplementedError

#     def get_state_action_values_total(self, state):
#         """
#         :param state: s, an environment state
#         :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions 这里是125维的联合动作估计值
#         """
#         return self.get_batch_total_effects_values([state])[0]

#     def step_optimizer(self, loss):
#         raise NotImplementedError

#     def step_optimizer_total(self, loss):
#         raise NotImplementedError

#     def step_optimizer_joint(self, loss):
#         raise NotImplementedError

#     def seed(self, seed=None):
#         return self.exploration_policy.seed(seed)

#     def reset(self):
#         pass

#     def set_writer(self, writer):
#         super().set_writer(writer)
#         try:
#             self.exploration_policy.set_writer(writer)
#         except AttributeError:
#             pass

#     def action_distribution(self, state):
#         self.previous_state = state
#         values = self.get_state_action_values(state)
#         self.exploration_policy.update(values)
#         return self.exploration_policy.get_distribution()

#     def set_time(self, time):
#         self.exploration_policy.set_time(time)

#     def eval(self):
#         self.training = False
#         self.config['exploration']['method'] = "Greedy"
#         self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)

#     def get_random(self):
#         # sensitive time 2.95204 -5%  2.86718 -10% (2.81554 -15%) -- 2.77881
#         if self.stn == 2.95204:
#             return 0.05
#         elif self.stn == 2.86718:
#             return 0.1
#         elif self.stn == 2.81554:
#             return 0.15
#         elif self.stn == 2.77881:
#             return 0.2

#     def perturb_actions(self, xs, actions): # xs:[x, y] x为智能体id y为目标动作  img: 当前的动作
#         # If this function is passed just one perturbation vector,
#         # pack it in a list to keep the computation the same
#         if xs.ndim < 2:
#            xs = np.array([xs])
#         batch = len(xs)
#         actions = actions.repeat(batch, 1,1)
#         xs = xs.astype(int)

#         count = 0
#         for x in xs:
#             pixels = np.split(x, len(x)/2)
#             # print(pixels)
#             for pixel in pixels:
#                 agent_id, action = pixel.astype(int)
#                 actions[count, 0, agent_id] = action
#             count += 1
#         # actions[agents_available_actions == 0] = 10.0
#         return actions

#     def predict_classes(self, xs, img, state_batch):
#         imgs_perturbed = self.perturb_actions(xs, img.clone())
#         batch_size = imgs_perturbed.shape[0]
#         # 将状态和扰动后的动作拼接为 state_joint_action
#         state_batch_tensor = torch.FloatTensor(state_batch).repeat(batch_size, 1)
#         state_joint_action = torch.cat([state_batch_tensor, imgs_perturbed.squeeze(1)], dim=1)
#         # 计算联合状态价值
#         predictions = self.get_state_joint_value(state_joint_action)

#         # combinations = imgs_perturbed.squeeze(1).long().cpu().numpy()
#         # weights = np.array([1, 5, 5 ** 2, 5 ** 3, 5 ** 4, 5 ** 5], dtype=np.int64)  # (6,)
#         # indices = (combinations * weights).sum(axis=-1)  # 计算所有索引 (batch_size,)
#         # # # **2. 直接索引 NumPy 数组**
#         # predictions = state_batch[indices]  # (batch_size,)
#         # predictions = predictions.reshape(-1, 1)
#         #
#         # predictions = np.zeros(shape=(batch_size, 1))
#         # idx = 0
#         # for xxx in imgs_perturbed:
#         #     combination = xxx[0]
#         #     index = combination[0] + combination[1] * 5 + combination[2] * 5 ** 2 + combination[3] * 5 ** 3 + combination[4] * 5 ** 4 + combination[5] * 5 ** 5
#         #     # index = combination[0] + combination[1] * 5 + combination[2] * 5 ** 2
#         #     predictions[idx] = state_batch[index]
#         #     idx += 1

#         return predictions

#     def attack_success(self, x, img, target_class, state_batch, verbose=False):
#         attack_image = self.perturb_actions(x, img.clone())
#         # 拼接状态和扰动后的动作
#         state_joint_action = torch.cat([state_batch, attack_image.squeeze(0).squeeze(0)], dim=-1)
#         # 计算扰动后的Q值
#         q_tot = self.get_state_joint_value(state_joint_action).item()

#         if verbose:
#             print(f"Original Q: {target_class.item()}, Perturbed Q: {q_tot}")
#         return q_tot < target_class.item()

#     def attack_q_de(self, img, label, actions, n_agent, n_action, state_batch,
#                     pixels=1, maxiter=75, popsize=1000, verbose=False,
#                     return_stats=False, algorithm_seed=None, state_list=None,
#                     eps=None, input_hash=None, fitness_mode=None,
#                     cache_enabled=None, early_stop=True):
#         """DE on a common strict-k domain, with an explicit scoring mode.

#         popsize is actual P. None seed uses the original global NumPy stream.
#         Default action_only uses the unchanged clean-state main scorer and no
#         cache/observation perturbations during search. Explicit full_pipeline
#         scores seeded observation attacks and owns a new private search cache.
#         """
#         _require_original_de_settings(popsize, maxiter)
#         mode = self._resolve_attack_fitness_mode(fitness_mode)
#         if mode == "full_pipeline":
#             result = self._search_full_pipeline("de", img, label, actions, n_agent, n_action,
#                 state_batch, pixels, state_list, eps, input_hash, algorithm_seed,
#                 popsize, maxiter, cache_enabled, early_stop)
#             if verbose:
#                 print(result[2])
#             return result if return_stats else result[:2]
#         if cache_enabled is True or early_stop is not True:
#             raise ValueError("Legacy action_only uses no cache and its original early-stop callback")
#         if int(np.__version__.split(".")[0]) >= 2:
#             raise RuntimeError("The project's batch DE requires NumPy 1.x")
#         n_agent = self._search_integer(n_agent, "n_agent", 1)
#         n_action = self._search_integer(n_action, "n_action", 1)
#         pixels = self._search_integer(pixels, "pixels", 1)
#         population = self._search_integer(popsize, "popsize", 5)
#         maxiter = self._search_integer(maxiter, "maxiter", 1)
#         baseline = self._search_scalar(label, "label")
#         callback_label = torch.tensor(baseline, dtype=torch.float64)
#         # predict_classes constructs float32 states. Use that SAME dtype for
#         # the unchanged attack_success callback, avoiding a different Q path.
#         callback_state = torch.as_tensor(state_batch, dtype=torch.float32)
#         if algorithm_seed is not None:
#             algorithm_seed = self._search_integer(algorithm_seed, "algorithm_seed")
#             if algorithm_seed >= 2**32:
#                 raise ValueError("algorithm_seed must be uint32")
#         rng = np.random if algorithm_seed is None else np.random.RandomState(algorithm_seed)
#         self._synchronize_attack_search()
#         started = time.perf_counter()
#         combos = self._agent_combos(n_agent, pixels)
#         dimension = pixels+1
#         bounds = [(0, len(combos))] + [(0, n_action)]*pixels
#         init = np.empty((population, dimension), dtype=np.float64)
#         init[:, 0] = rng.randint(0, len(combos), size=population)
#         init[:, 1:] = rng.randint(0, n_action, size=(population, pixels))
#         optimizer_rows, callback_rows, batches = 0, 0, 0
#         seen = set()
#         stopped = False

#         def objective(zs):
#             nonlocal optimizer_rows, batches
#             candidates = np.asarray([self._decode_de_candidate(z, combos, n_action, pixels)
#                                      for z in np.atleast_2d(zs)], dtype=np.int64)
#             optimizer_rows += len(candidates)
#             batches += 1
#             seen.update(tuple(int(v) for v in row) for row in candidates)
#             return self._action_only_values(candidates, img, state_batch)

#         def callback(z, convergence):
#             nonlocal callback_rows, stopped
#             candidate = self._decode_de_candidate(z, combos, n_action, pixels)
#             callback_rows += 1
#             seen.add(tuple(candidate))
#             with torch.no_grad():
#                 stopped = bool(self.attack_success(np.asarray(candidate, dtype=np.int64),
#                     img, callback_label, callback_state, verbose))
#             return stopped

#         result = differential_evolution(objective, bounds, maxiter=maxiter,
#             popsize=max(1, population//dimension), strategy="best1bin",
#             mutation=(.5, 1), recombination=1, atol=-1, callback=callback,
#             polish=False, init=init, seed=None if algorithm_seed is None else rng)
#         best = np.asarray(self._decode_de_candidate(result.x, combos, n_action, pixels), dtype=np.int64)
#         best_q = self._search_scalar(result.fun, "selected Q")
#         nfev, nit = int(result.nfev), int(result.nit)
#         if nfev != optimizer_rows or nfev != population*(1+nit):
#             raise RuntimeError("Custom batch DE evaluation accounting changed")
#         if callback_rows != nit:
#             raise RuntimeError("Expected one attack_success callback per completed generation")
#         attack_image = self.perturb_actions(best, img.clone())
#         target_actions = attack_image[0, 0].data.cpu().numpy().tolist()
#         self._synchronize_attack_search()
#         elapsed = time.perf_counter()-started
#         stats = dict(method="de", fitness_mode="action_only",
#             candidate_space="exactly_k_distinct_agents_allow_original_action",
#             n_agent=n_agent, n_action=n_action, k=pixels,
#             space_size=len(combos)*n_action**pixels,
#             population_size=population, maxiter=maxiter, nit=nit, nfev=nfev,
#             callback_evaluations=callback_rows, total_q_rows=nfev+callback_rows,
#             fitness_evaluations=nfev+callback_rows, optimizer_batches=batches,
#             fitness_batches=batches+callback_rows,
#             unique_candidates=len(seen), best_q=best_q, baseline_q=baseline,
#             agent_action_id=best.tolist(), search_time_s=elapsed,
#             strategy="best1bin", mutation=[.5, 1.0], recombination=1,
#             algorithm_seed=algorithm_seed, optimizer_success=bool(result.success),
#             optimizer_message=str(result.message), attack_success=best_q < baseline,
#             stop_reason="baseline_improved" if stopped else
#                         "maxiter" if nit >= maxiter else "solver_convergence")
#         self.last_attack_search = stats
#         return (target_actions, best, stats) if return_stats else (target_actions, best)

#     @staticmethod
#     def _search_integer(value, name, minimum=0):
#         if isinstance(value, (bool, np.bool_)):
#             raise ValueError(name + " must be an integer")
#         try:
#             result = int(value)
#         except (TypeError, ValueError, OverflowError):
#             raise ValueError(name + " must be an integer") from None
#         if result != value or result < minimum:
#             raise ValueError(name + " must be an integer >= " + str(minimum))
#         return result

#     @staticmethod
#     def _search_scalar(value, name):
#         if torch.is_tensor(value):
#             value = value.detach().cpu().numpy()
#         array = np.asarray(value)
#         if array.size != 1 or not np.isfinite(array).all():
#             raise ValueError(name + " must be exactly one finite scalar")
#         return float(array.reshape(-1)[0])

#     def _agent_combos(self, n_agent, pixels):
#         n_agent = self._search_integer(n_agent, "n_agent", 1)
#         pixels = self._search_integer(pixels, "pixels", 1)
#         if pixels > n_agent:
#             raise ValueError("pixels cannot exceed n_agent")
#         return list(itertools.combinations(range(n_agent), pixels))

#     def _search_space_size(self, n_agent, n_action, pixels):
#         n_action = self._search_integer(n_action, "n_action", 1)
#         return len(self._agent_combos(n_agent, pixels)) * n_action ** int(pixels)

#     def _enumerate_attack_candidates(self, n_agent, n_action, pixels):
#         n_action = self._search_integer(n_action, "n_action", 1)
#         return np.asarray([
#             [v for pair in zip(agents, action_tuple) for v in pair]
#             for agents in self._agent_combos(n_agent, pixels)
#             for action_tuple in itertools.product(range(n_action), repeat=int(pixels))
#         ], dtype=np.int64)

#     def _decode_de_candidate(self, z, combos, n_action, pixels):
#         values = np.asarray(z, dtype=np.float64).reshape(-1)
#         if values.size != int(pixels)+1 or not np.isfinite(values).all():
#             raise ValueError("Malformed DE vector [combination_index, actions...]")
#         subset = min(max(int(np.floor(values[0])), 0), len(combos)-1)
#         return [v for agent, target in zip(combos[subset], values[1:])
#                 for v in (int(agent), min(max(int(np.floor(target)), 0), int(n_action)-1))]

#     def _synchronize_attack_search(self):
#         device = getattr(self, "device", None)
#         if device is not None and torch.device(device).type == "cuda":
#             torch.cuda.synchronize(device)

#     def _action_only_values(self, candidates, img, state_batch):
#         # The unchanged main predict_classes is shared by BOTH search methods.
#         with torch.no_grad():
#             values = self.predict_classes(candidates, img, state_batch)
#         if torch.is_tensor(values):
#             values = values.detach().cpu().numpy()
#         values = np.asarray(values, dtype=np.float64).reshape(-1)
#         if values.size != len(candidates) or not np.isfinite(values).all():
#             raise ValueError("predict_classes must return one finite Q per candidate")
#         return values

#     def attack_q_exhaustive(self, img, label, actions, n_agent, n_action, state_batch,
#                             pixels=1, verbose=False, return_stats=True, state_list=None,
#                             eps=None, input_hash=None, fitness_mode=None,
#                             algorithm_seed=None, cache_enabled=None):
#         """Enumerate the whole strict-k domain under the selected common scorer.

#         action_only uses main predict_classes. Explicit full_pipeline executes
#         the same seeded observation-attack/Q pipeline used by full-mode DE.
#         """
#         mode = self._resolve_attack_fitness_mode(fitness_mode)
#         if mode == "full_pipeline":
#             result = self._search_full_pipeline("exh", img, label, actions, n_agent, n_action,
#                 state_batch, pixels, state_list, eps, input_hash, algorithm_seed,
#                 None, None, cache_enabled, False)
#             if verbose:
#                 print(result[2])
#             return result if return_stats else result[:2]
#         if cache_enabled is True:
#             raise ValueError("Legacy action_only exhaustive search does not use a cache")
#         n_agent = self._search_integer(n_agent, "n_agent", 1)
#         n_action = self._search_integer(n_action, "n_action", 1)
#         pixels = self._search_integer(pixels, "pixels", 1)
#         baseline = self._search_scalar(label, "label")
#         self._synchronize_attack_search()
#         started = time.perf_counter()
#         candidates = self._enumerate_attack_candidates(n_agent, n_action, pixels)
#         values = self._action_only_values(candidates, img, state_batch)
#         index = int(np.argmin(values))
#         best = candidates[index].copy()
#         best_q = float(values[index])
#         attack_image = self.perturb_actions(best, img.clone())
#         target_actions = attack_image[0, 0].data.cpu().numpy().tolist()
#         self._synchronize_attack_search()
#         elapsed = time.perf_counter()-started
#         size = len(candidates)
#         stats = dict(method="exh", fitness_mode="action_only",
#             candidate_space="exactly_k_distinct_agents_allow_original_action",
#             n_agent=n_agent, n_action=n_action, k=pixels, space_size=size,
#             population_size=None, maxiter=None, nit=0, nfev=size,
#             callback_evaluations=0, total_q_rows=size, fitness_evaluations=size,
#             optimizer_batches=1, fitness_batches=1, unique_candidates=size,
#             best_q=best_q, baseline_q=baseline, agent_action_id=best.tolist(),
#             tie_set_size=int(np.count_nonzero(values == best_q)), tie_break="first_exact_minimum",
#             search_time_s=elapsed, optimizer_success=None,
#             attack_success=best_q < baseline, stop_reason="all_candidates_enumerated")
#         self.last_attack_search = stats
#         if verbose:
#             print("EXH", "space_size=", size, "best_q=", best_q, "search_time_s=", elapsed)
#         return (target_actions, best, stats) if return_stats else (target_actions, best)

#     def paired_search_audit(self, img, label, actions, n_agent, n_action, state_batch,
#                             pixels=1, maxiter=75, popsize=1000, verbose=False,
#                             algorithm_seed=None, tolerance=1e-5, execute_as="de",
#                             state_list=None, eps=None, input_hash=None,
#                             fitness_mode=None, cache_enabled=None):
#         """Compare the selected search objective on one frozen input.

#         No environment steps or terminal performance metrics. action_only does
#         not perturb observations; full_pipeline scores seeded observation
#         attacks with separate per-method caches and the common state identity.
#         """
#         _require_original_de_settings(popsize, maxiter)
#         tolerance = float(tolerance)
#         if not np.isfinite(tolerance) or tolerance <= 0:
#             raise ValueError("tolerance must be positive and finite")
#         if execute_as not in ("de", "exh", "exhaustive"):
#             raise ValueError("execute_as must be de or exhaustive")
#         mode = self._resolve_attack_fitness_mode(fitness_mode)
#         frozen_img = img.detach().clone()
#         frozen_actions = actions.detach().clone() if torch.is_tensor(actions) else copy.deepcopy(actions)
#         frozen_state = state_batch.detach().clone() if torch.is_tensor(state_batch) else np.array(state_batch, copy=True)
#         common = dict(img=frozen_img, label=label, actions=frozen_actions, n_agent=n_agent,
#                       n_action=n_action, state_batch=frozen_state, pixels=pixels, return_stats=True,
#                       fitness_mode=mode, state_list=state_list, eps=eps, input_hash=input_hash,
#                       cache_enabled=cache_enabled)
#         _, ex_candidate, ex = self.attack_q_exhaustive(**common, verbose=verbose,
#             algorithm_seed=algorithm_seed)
#         _, de_candidate, de = self.attack_q_de(**common, maxiter=maxiter, popsize=popsize,
#             verbose=verbose, algorithm_seed=algorithm_seed)
#         gap = de["best_q"]-ex["best_q"]
#         if gap < -tolerance:
#             raise RuntimeError("EXH oracle inequality violated on the shared " + mode + " objective")
#         selected = de_candidate if execute_as == "de" else ex_candidate
#         record = dict(fitness_mode=mode, space_size=ex["space_size"],
#             exhaustive=ex, de=de, q_gap_de_minus_exh=gap, tolerance=tolerance,
#             dominance_ok=True, de_reaches_exh_optimum=bool(abs(gap) <= tolerance),
#             execute_as="de" if execute_as == "de" else "exh",
#             agent_action_id=selected.tolist())
#         record["paired_total_search_time_s"] = ex["search_time_s"]+de["search_time_s"]
#         record["timing_note"] = "paired audit runs both solvers; separate from one-solver rollout timing"
#         self.last_paired_audit = record
#         self.last_attack_search = de if execute_as == "de" else ex
#         return selected.copy(), record

#     def _resolve_attack_fitness_mode(self, requested):
#         mode = self.attack_fitness_mode if requested is None else requested
#         if mode not in ("action_only", "full_pipeline"):
#             raise ValueError("attack_fitness_mode must be action_only or full_pipeline")
#         return mode

#     def _require_full_pipeline_setup(self):
#         # Match the actual Table10 six-agent/five-action, fixed Greedy policy.
#         if self.config.get("exploration", {}).get("method") != "Greedy" or type(
#                 self.exploration_policy).__name__ != "Greedy":
#             raise RuntimeError("Table10 full_pipeline requires the fixed Greedy victim policy")
#         for name in ("value_net", "value_net_total", "joint_action_value_net"):
#             net = getattr(self, name, None)
#             if not isinstance(net, torch.nn.Module) or net.training:
#                 raise RuntimeError("Table10 requires an initialized eval-mode " + name)
#         if self.use_random_noise:
#             raise ValueError("Table10 full_pipeline uses gradient perturbations, not random noise")
#         if torch.is_inference_mode_enabled():
#             raise RuntimeError("Full-pipeline gradient attacks cannot run in inference_mode; use no_grad")

#     @staticmethod
#     def _full_observation(observation):
#         values = np.asarray(observation, dtype=np.float32)
#         if values.shape != (6, 5, 5) or not np.isfinite(values).all():
#             raise ValueError("Table10 requires six finite 5x5 observations")
#         return values

#     @staticmethod
#     def _full_nominal(nominal):
#         if torch.is_tensor(nominal):
#             nominal = nominal.detach().cpu().numpy()
#         values = np.asarray(nominal).reshape(-1)
#         if len(values) != 6:
#             raise ValueError("Table10 requires six nominal actions")
#         result = tuple(_integer(value, "nominal action") for value in values)
#         if any(value >= 5 for value in result):
#             raise ValueError("Table10 actions must be in 0,...,4")
#         return result

#     def _full_eps(self, eps, k):
#         values = np.asarray(eps, dtype=float).reshape(-1)
#         if values.size != 3 or not np.isfinite(values).all() or values[0] < 0:
#             raise ValueError("eps must be [nonnegative step_size, k, remaining budget]")
#         if _integer(values[1], "eps k", 1) != k:
#             raise ValueError("eps k differs from search k")
#         budget = _integer(self.config.get("table10_attack_budget", 5), "table10_attack_budget", 1)
#         return [float(values[0]), int(k), budget]

#     def _full_input_hash(self, observation, nominal, k):
#         # EXACT measured full_pipeline_rollout_contract.state_hash definition.
#         return digest({"observation": self._full_observation(observation).tolist(),
#                        "original_action": list(self._full_nominal(nominal)), "k": int(k)})

#     def _full_joint_values(self, observation, actions):
#         joined = torch.cat((torch.from_numpy(np.asarray(observation, dtype=np.float32)).reshape(-1),
#                             torch.from_numpy(np.asarray(actions)).reshape(-1)))
#         reference = next(self.joint_action_value_net.parameters())
#         joined = joined.to(device=reference.device, dtype=reference.dtype)
#         return np.asarray(self.get_state_joint_value(joined)).reshape(-1)

#     def _full_clean_baseline(self, observation, nominal):
#         with torch.no_grad():
#             values = self._full_joint_values(self._full_observation(observation), self._full_nominal(nominal))
#         return self._search_scalar(values, "clean full-pipeline Q")

#     def execute_full_pipeline_candidate(self, observation, nominal, k, candidate, input_hash, eps):
#         """Same seeded physical pipeline as the measured Table10 candidate scorer.

#         Returns (executed_actions, Q). Final selected replay is an extra physical
#         evaluation outside optimizer NFEV and its private search cache.
#         """
#         self._require_full_pipeline_setup()
#         k = _integer(k, "k", 1)
#         if k > 6:
#             raise ValueError("k cannot exceed six")
#         eps = self._full_eps(eps, k)
#         observation_array = self._full_observation(observation)
#         nominal = self._full_nominal(nominal)
#         canonical = _PipelineScorer(None, 6, 5, k, True, lambda: None)._key(candidate)
#         aid = [int(value) for value in canonical]
#         original = digest({"observation": np.asarray(observation).tolist(), "actions": list(nominal)})
#         frozen = [np.array(value, dtype=np.float32, copy=True) for value in observation_array]
#         base_seed = _integer(self.attack_base_seed, "attack_base_seed")
#         seed = derived_seed(base_seed, "full_pipeline_candidate", str(input_hash), aid)
#         with isolated_rng(np, torch, seed, self):
#             private = getattr(self.exploration_policy, "np_random", None)
#             if hasattr(private, "seed"):
#                 private.seed(seed)
#             elif hasattr(private, "bit_generator"):
#                 private.bit_generator.state = type(private.bit_generator)(seed).state
#             for name in ("value_net", "value_net_total", "joint_action_value_net"):
#                 getattr(self, name).zero_grad(set_to_none=True)
#             with torch.enable_grad():
#                 executed = self.get_final_action(frozen, eps, aid, use_random_noise=False)
#             executed = self._full_nominal(executed)
#             self._full_observation(frozen)
#             with torch.no_grad():
#                 q = self._search_scalar(self._full_joint_values(frozen, executed), "executed full-pipeline Q")
#         if digest({"observation": np.asarray(observation).tolist(), "actions": list(nominal)}) != original:
#             raise RuntimeError("Full-pipeline execution mutated frozen input")
#         return tuple(executed), q

#     def score_full_pipeline_candidate(self, observation, nominal, k, candidate, input_hash, eps):
#         return self.execute_full_pipeline_candidate(observation, nominal, k, candidate, input_hash, eps)[1]

#     def _search_full_pipeline(self, method, img, label, actions, n_agent, n_action,
#                               state_batch, pixels, state_list, eps, input_hash,
#                               algorithm_seed, population, maxiter, cache_enabled, early_stop):
#         if method not in METHODS:
#             raise ValueError("Search method must be de or exh")
#         if method == "de":
#             _require_original_de_settings(population, maxiter)
#         self._require_full_pipeline_setup()
#         if int(np.__version__.split(".")[0]) >= 2:
#             raise RuntimeError("The project's archived batch DE requires NumPy 1.x")
#         if _integer(n_agent, "n_agent", 1) != 6 or _integer(n_action, "n_action", 1) != 5:
#             raise ValueError("Table10 full_pipeline requires n_agent=6 and n_action=5")
#         k = _integer(pixels, "pixels", 1)
#         if not 1 <= k <= 6:
#             raise ValueError("k must be in 1,...,6")
#         if state_list is None or eps is None:
#             raise ValueError("full_pipeline requires state_list and eps; no objective fallback")
#         observation = tuple(np.array(x, copy=True) for x in self._full_observation(state_list))
#         nominal = self._full_nominal(actions)
#         if self._full_nominal(img) != nominal:
#             raise ValueError("img and nominal actions disagree")
#         eps = self._full_eps(eps, k)
#         if state_batch is not None:
#             flattened = state_batch.detach().cpu().numpy() if torch.is_tensor(state_batch) else state_batch
#             if not np.array_equal(np.asarray(flattened, dtype=np.float32).reshape(-1),
#                                   np.asarray(observation).reshape(-1)):
#                 raise ValueError("state_batch and frozen state_list disagree")
#         state_hash = self._full_input_hash(observation, nominal, k) if input_hash is None else str(input_hash)
#         baseline = self._full_clean_baseline(observation, nominal)
#         if label is not None and abs(self._search_scalar(label, "label")-baseline) > 1e-7:
#             raise ValueError("Supplied label disagrees with full-pipeline clean baseline")
#         if algorithm_seed is None:
#             algorithm_seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
#         cache_enabled = self.fitness_cache_enabled if cache_enabled is None else cache_enabled
#         if type(cache_enabled) is not bool or type(early_stop) is not bool:
#             raise ValueError("cache_enabled and early_stop must be boolean")
#         params = {}
#         if method != "exh":
#             population = _integer(population, "population", 5)
#             maxiter = _integer(maxiter, "maxiter", 1)
#             params = dict(population=population, maxiter=maxiter,
#                           strategy="best1bin", mutation=(.5, 1), recombination=1)
#         solver_status = {}

#         def measured_solver(*args, **kwargs):
#             solved = differential_evolution(*args, **kwargs)
#             solver_status.update(success=bool(solved.success), message=str(solved.message))
#             return solved

#         result = run_full_search(method, 6, 5, k,
#             lambda candidate: self.score_full_pipeline_candidate(observation, nominal, k, candidate, state_hash, eps),
#             baseline, algorithm_seed, solver=measured_solver,
#             synchronize=self._synchronize_attack_search, early_stop=early_stop,
#             input_hash=state_hash, cache_enabled=cache_enabled, **params)
#         # The public core fixes DE parameters; physical scoring stays unchanged.
#         result.update(candidate_base_seed=int(self.attack_base_seed),
#             attack_step_size=eps[0], attack_budget=eps[2],
#             optimizer_success=solver_status.get("success"),
#             optimizer_message=solver_status.get("message"),
#             attack_success=result["best_q"] < baseline, baseline_q=baseline,
#             full_pipeline_execution_verified=False,
#             timing_note="search only; clean baseline and selected replay are separate")
#         target = list(nominal)
#         candidate = np.asarray(result["agent_action_id"], dtype=np.int64)
#         for agent, action in zip(candidate[::2], candidate[1::2]):
#             target[int(agent)] = int(action)
#         self.last_attack_search = result
#         return target, candidate, result
