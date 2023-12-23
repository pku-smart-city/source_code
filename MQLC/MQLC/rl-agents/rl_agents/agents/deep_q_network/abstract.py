from abc import ABC, abstractmethod
import numpy as np
from gym import spaces

from rl_agents.agents.common.abstract import AbstractStochasticAgent
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.memory import ReplayMemory, Transition
import torch

import itertools

class AbstractDQNAgent(AbstractStochasticAgent, ABC):
    def __init__(self, env, config=None):
        super(AbstractDQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
            "Only compatible with Discrete action spaces."
        self.memory = ReplayMemory(self.config)
        # self.memory_other = ReplayMemory(self.config)
        # self.memory_global = ReplayMemory(self.config)
        self.memory_total = ReplayMemory(self.config)
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
        self.episode_count = 0
        self.training = True
        self.previous_state = None

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="DuelingNetwork"),
                    optimizer=dict(type="ADAM",
                                   lr=5e-4,
                                   weight_decay=0,
                                   k=5),
                    optimizer_total=dict(type="ADAM",
                                   lr=1e-2,
                                   weight_decay=0,
                                   k=5),
                    loss_function="l2",
                    memory_capacity=50000,
                    batch_size=100,
                    gamma=0.99,
                    device="cuda:best",
                    exploration=dict(method="EpsilonGreedy"),
                    target_update=1,
                    double=True)

    def record(self, state, action, rewards, next_state, done, info):
        """
            Record a transition by performing a Deep Q-Network iteration

            - push the transition into memory
            - sample a minibatch
            - compute the bellman residual loss over the minibatch
            - perform one gradient descent step
            - slowly track the policy network with the target network
        :param state: a state
        :param action: an action
        :param reward: a reward
        :param next_state: a next state
        :param done: whether state is terminal
        """
        if not self.training:
            return
        if isinstance(state, tuple) and isinstance(action, tuple):  # Multi-agent setting
            [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
             for agent_state, agent_action, reward, agent_next_state in zip(state, action, rewards, next_state)]

            # # 将Qother所用更新元祖使用如下传入memory other
            # for index in range(len(state)):
            #     for j in range(len(state)):
            #         if j != index:
            #             obs_temp = np.vstack((state[index], state[j]))
            #             obs_next_temp = np.vstack((next_state[index], next_state[j]))
            #             # self.memory.push_other(obs_temp, action[index], reward, obs_next_temp, done, info)
            #             self.memory_other.push(obs_temp, action[index], reward, obs_next_temp, done, info)

            # # 将GCN版Qother所用更新元祖使用如下传入memory other
            # [self.memory_other.push(agent_state, agent_action, reward, agent_next_state, done, info)
            #  for agent_state, agent_action, reward, agent_next_state in zip(state, action, rewards, next_state)]
                # obs_temp = state[index][:, 1:3]
                # obs_next_temp = next_state[index][:, 1:3]
                # obs_temp = np.array(obs_temp)
                # obs_next_temp = np.array(obs_next_temp)
                # distances = np.linalg.norm(obs_temp[:, np.newaxis] - obs_temp, axis=2)  #
                # distances_next = np.linalg.norm(obs_next_temp[:, np.newaxis] - obs_next_temp, axis=2)
                # threshold = 0.5  # 距离阈值
                # adj_matrix = (distances < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
                # adj_next_matrix = (distances_next < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
                #
                # # 归一化邻接矩阵
                # rowsum = np.sum(adj_matrix, axis=1)
                # rowsum_next = np.sum(adj_next_matrix, axis=1)
                # D = np.diag(1.0 / np.sqrt(rowsum))
                # D_next = np.diag(1.0 / np.sqrt(rowsum_next))
                # normalized_adj_matrix = np.matmul(np.matmul(D, adj_matrix), D)
                # normalized_adj_matrix_next = np.matmul(np.matmul(D_next, adj_next_matrix), D_next)
                #
                # # 节点特征
                # node_features = torch.tensor(obs_temp, dtype=torch.float)
                # node_features_next = torch.tensor(obs_next_temp, dtype=torch.float)
                #
                # # 前向传播
                # normalized_adj_matrix = torch.tensor(normalized_adj_matrix)
                # normalized_adj_matrix_next = torch.tensor(normalized_adj_matrix_next)
                # node_features = torch.tensor(node_features)
                # node_features_next = torch.tensor(node_features_next)
                # normalized_adj_matrix = normalized_adj_matrix.float()
                # normalized_adj_matrix_next = normalized_adj_matrix_next.float()
                # # result = self.get_batch_other_effects_values(normalized_adj_matrix, node_features)[0]
                # obs_final = [node_features, normalized_adj_matrix]
                # obs_final_next = [node_features_next, normalized_adj_matrix_next]


            # # 将Qglobal所用更新元祖使用如下传入memory global
            # for index in range(len(state)):
            #     ego_state = state[index]
            #     # 计算速度平方和再开根号
            #     norms = np.sqrt(np.sum(ego_state[:, -2:] ** 2, axis=1))
            #     # 计算平均值
            #     average_norm = np.mean(norms)
            #     # print(average_norm)
            #     L = 0.1
            #     near_vehicle = 0
            #     for agent_state in state:
            #         for row in agent_state:
            #             if abs(row[1] - ego_state[0][1]) <= L:
            #                 near_vehicle += 1
            #     density = near_vehicle / (2 * (L * 1500))
            #     density = density * 10
            #     self.memory_global.push_global(average_norm, density)

            # 将total更新所用元祖传入memory
            state_total = np.concatenate(state, axis=0)
            rewards_total = np.sum(rewards)
            next_state_total = np.concatenate(next_state, axis=0)
            self.memory_total.push(state_total, action, rewards_total, next_state_total, done, info)

        else:  # Single-agent setting
            self.memory.push(state, action, rewards, next_state, done, info)
        batch = self.sample_minibatch()
        # batch_other = self.sample_minibatch_other()
        # batch_global = self.sample_minibatch_global()
        batch_total = self.sample_minibatch_total()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            self.step_optimizer(loss)
            self.update_target_network()
        # if batch_other:
        #     loss, _, _ = self.compute_bellman_residual_other(batch_other)
        #     self.step_optimizer_other(loss)
        #     # self.update_target_network_other()
        # if batch_global:
        #     loss, _, _ = self.compute_bellman_residual_global(batch_global)
        #     self.step_optimizer_global(loss)
        #     # self.update_target_network_other()
        if batch_total:
            loss, _, _ = self.compute_bellman_residual_total(batch_total)
            self.step_optimizer_total(loss)
            lameda = self.adjust_lameda()
            loss_reg, _, _ = self.compute_bellman_residual_reg(batch_total)
            self.step_optimizer(lameda * loss_reg)



    # def act(self, state, step_exploration_time=True):
    #     """
    #         Act according to the state-action value model and an exploration policy
    #     :param state: current state
    #     :param step_exploration_time: step the exploration schedule
    #     :return: an action
    #     """
    #     self.previous_state = state
    #     if step_exploration_time:
    #         self.exploration_policy.step_time()
    #     # Handle multi-agent observations
    #     # TODO: it would be more efficient to forward a batch of states
    #     if isinstance(state, tuple):
    #         return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)
    #
    #     # Single-agent setting
    #     values = self.get_state_action_values(state)
    #     self.exploration_policy.update(values)
    #     return self.exploration_policy.sample()

    # def act(self, state, step_exploration_time=True):
    #     """
    #         Act according to the state-action value model and an exploration policy
    #     :param state: current state
    #     :param step_exploration_time: step the exploration schedule
    #     :return: an action
    #     """
    #     self.previous_state = state[-1]
    #     if step_exploration_time:
    #         self.exploration_policy.step_time()
    #     # Handle multi-agent observations
    #     act_temp = []
    #     if isinstance(state, tuple):
    #         # 返回一个智能体的动作turple（4,2,0）
    #         _ = 0
    #         for agent_state in state:
    #             # 输入是一个ndarray
    #             # 输出是一个价值
    #             values_origin = self.get_state_action_values(agent_state)
    #             _ += 1
    #             self.exploration_policy.update(values_origin)
    #             act_temp.append(self.exploration_policy.sample()[0])
    #
    #     return tuple(act_temp)

    def act(self, state, step_exploration_time=True):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        协同决策，对于车辆的紧急情况做判断
        """
        self.previous_state = state[-1]
        if step_exploration_time:
            self.exploration_policy.step_time()
        # Handle multi-agent observations
        act_temp = []
        if isinstance(state, tuple):
            # 返回一个智能体的动作turple（4,2,0）
            _ = 0
            for agent_state in state:
                # 输入是一个ndarray
                # 输出是一个价值
                values_origin = self.get_state_action_values(agent_state)
                _ += 1

                # 判断紧急与否
                min = 0
                max_mean_v = 1
                max_traffic_d = 100
                max_speed_v = 0.05
                mean_v = (np.mean(np.sqrt(np.sum(agent_state[:, 3:5]**2, axis=1))) - min) / (max_mean_v - min)
                traffic_d = (np.max(np.abs(agent_state[:, 1][:, np.newaxis] - agent_state[:, 1])) - min) / (max_traffic_d - min)
                speed_v = (np.var(np.sqrt(agent_state[:, 3]**2 + agent_state[:, 4]**2)) - min) / (max_speed_v - min)
                # print("mean_v traffic_d speed_v:", mean_v, 1/traffic_d, speed_v)
                if (mean_v + traffic_d + 2*speed_v) > 1:
                    self.exploration_policy.update(values_origin)
                    act_temp.append(self.exploration_policy.sample())
                else:
                    # 非紧急情况，将最优及次优动作列入选择
                    self.exploration_policy.update(values_origin)
                    act_temp.append(self.exploration_policy.sample_total())
                    # act_temp.append(self.exploration_policy.sample())
            # 写如何处理有三个ndarray的list
            values = [list(arr) for arr in act_temp]
            combinations = list(itertools.product(*values))
            max_value = 0
            best_act = act_temp
            con_state = (np.concatenate((state[0], state[1], state[2]), axis=0), )
            _ = 0
            value_total = self.get_state_action_values_total(con_state)[0]
            for combination in combinations:
                index = combination[0] + combination[1]*5 + combination[2]*5**2
                if _ == 0:
                    max_value = value_total[index]
                    best_act = combination
                if value_total[index]>max_value:
                    max_value = value_total[index]
                    best_act = combination
                    _ = 1

        return tuple(best_act)

    def sample_minibatch(self):
        if len(self.memory) < self.config["batch_size"]:
            return None
        transitions = self.memory.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    # def sample_minibatch_other(self):
    #     if len(self.memory_other) < self.config["batch_size"]:
    #         return None
    #     transitions = self.memory_other.sample(self.config["batch_size"])
    #     return Transition(*zip(*transitions))
    #
    # def sample_minibatch_global(self):
    #     if len(self.memory_other) < self.config["batch_size"]:
    #         return None
    #     transitions = self.memory_global.sample(self.config["batch_size"])
    #     return Transition(*zip(*transitions))

    def sample_minibatch_total(self):
        if len(self.memory_total) < self.config["batch_size"]:
            return None
        transitions = self.memory_total.sample(self.config["batch_size"])
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
            # self.target_net_other.load_state_dict(self.value_net_other.state_dict())
            self.target_net_total.load_state_dict(self.value_net_total.state_dict())

    # def update_target_network_other(self):
    #     self.steps += 1
    #     if self.steps % self.config["target_update"] == 0:
    #         self.target_net.load_state_dict(self.value_net.state_dict())

    def adjust_lameda(self):
        explameda = 0.1
        adjust_episodes = 1000
        self.episode_count += 1
        return self.episode_count//adjust_episodes * (explameda/(10000/adjust_episodes))

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

    # def get_batch_other_effects_values(self, states):
    #     """
    #     :param index:智能体编号 agent_state:全局观察
    #     需要根据index将输入处理成10(2*obs)*num(feature)的形式，第一个obs需要是决策车辆的观察
    #     :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions+distance影响后的
    #     """
    #     raise NotImplementedError
    #
    # def get_batch_global_effects_values(self, states):
    #     """
    #     :param index:智能体编号 agent_state:全局观察
    #     需要根据index将输入处理成10(2*obs)*num(feature)的形式，第一个obs需要是决策车辆的观察
    #     :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions+distance影响后的
    #     """
    #     raise NotImplementedError

    def get_batch_total_effects_values(self, states):
        """
        """
        raise NotImplementedError

    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    # Qorigin计算影响
    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    # 其他智能体影响
    # def get_other_effects_values(self, index, agent_state):
    #     """
    #         :param index:智能体编号 agent_state:全局观察
    #         需要根据index将输入处理成10(2*obs)*num(feature)的形式，第一个obs需要是决策车辆的观察
    #         :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions+distance影响后的
    #     """
    #     value_temp  = []
    #     distance = []
    #     for j in range(len(agent_state)):
    #         if j != index:
    #             obs_temp = np.vstack((agent_state[index], agent_state[j]))
    #             # 算距离影响
    #             obs_temp_dis = np.vstack((agent_state[index][0, 1:3], agent_state[j][0, 1:3]))
    #             distance.append(np.linalg.norm(obs_temp_dis[0, :] - obs_temp_dis[1, :]))
    #             value_temp.append(self.get_batch_other_effects_values([obs_temp])[0])
    #     # value_temp中含有num(agent)-1条价值,分别表示其他智能体对决策智能体的影响
    #     distance = [1 / x for x in distance]
    #     total = sum(distance)
    #     normalized_dis = [d / total for d in distance]
    #     for i in range(len(value_temp)):
    #         value_temp[i] = value_temp[i] * normalized_dis[i]
    #     result = [sum(x) for x in zip(*value_temp)]
    #     return result

    # def get_other_effects_values(self, state):
    #     """
    #         调用GCN
    #         :param index:智能体编号 state:全局观察
    #         处理成为邻接矩阵与点信息的形式
    #         :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions+distance影响后的
    #     """
    #     obs_temp = state[:, 1:3]
    #     obs_temp = torch.tensor(obs_temp).clone().detach()
    #     result = self.get_batch_other_effects_values(obs_temp)[0]
    #
    #     return result

    # # 全局影响
    # def get_global_effects_values(self, index, state):
    #     """
    #         :param index:智能体编号 agent_state:全局观察
    #         将输入处理成平均速度+车流密度的1×2的矩阵形式
    #         :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions+distance影响后的
    #     """
    #     ego_state = state[index]
    #     # 计算速度平方和再开根号
    #     norms = np.sqrt(np.sum(ego_state[:, -2:] ** 2, axis=1))
    #     # 计算平均值
    #     average_norm = np.mean(norms)
    #     # print(average_norm)
    #     L = 0.1
    #     near_vehicle = 0
    #     for agent_state in state:
    #         for row in agent_state:
    #             if abs(row[1] - ego_state[0][1]) <= L:
    #                 near_vehicle += 1
    #     density = near_vehicle / (2 * (L * 1500))
    #     density = density * 10
    #     # 将距离平均值与周围车辆归一化到0-1之间
    #     # 归一化到0到1之间
    #     # normalized_value1 = (value1 - min(value1, value2)) / (max(value1, value2) - min(value1, value2))
    #     # normalized_value2 = (value2 - min(value1, value2)) / (max(value1, value2) - min(value1, value2))
    #
    #     # 是一个数量级，好像并不用归一化
    #     normalized_matrix = np.array([[average_norm, density]])
    #
    #     result = self.get_batch_global_effects_values([normalized_matrix])[0]
    #     return result

    def get_state_action_values_total(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions 这里是125维的联合动作估计值
        """
        return self.get_batch_total_effects_values([state])[0]


    def step_optimizer(self, loss):
        raise NotImplementedError

    # def step_optimizer_other(self, loss):
    #     raise NotImplementedError
    #
    # def step_optimizer_global(self, loss):
    #     raise NotImplementedError

    def step_optimizer_total(self, loss):
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

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval(self):
        self.training = False
        self.config['exploration']['method'] = "Greedy"
        self.exploration_policy = exploration_factory(self.config["exploration"], self.env.action_space)
