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
