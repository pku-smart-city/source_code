import logging
import torch
from gym import spaces
import numpy as np
import pickle

from rl_agents.agents.common.memory import Transition
from rl_agents.agents.common.models import model_factory, size_model_config, trainable_parameters
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory
from rl_agents.agents.common.utils import choose_device
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent

logger = logging.getLogger(__name__)


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        size_model_config(self.env, self.config["model"])
        self.value_net = model_factory(self.config["model"])
        # self.value_net_other = model_factory(self.config["model"], 2)
        # self.value_net_other = model_factory(self.config["model1"])
        # self.value_net_global = model_factory(self.config["model"], 3)
        self.value_net_total = model_factory(self.config["model"], 4)
        self.target_net = model_factory(self.config["model"])
        # self.target_net_other = model_factory(self.config["model"], 2)
        # self.target_net_other = model_factory(self.config["model1"])
        # self.target_net_global = model_factory(self.config["model"], 3)
        self.target_net_total = model_factory(self.config["model"], 4)
        self.target_net.load_state_dict(self.value_net.state_dict())
        # self.target_net_other.load_state_dict(self.value_net_other.state_dict())
        # self.target_net_global.load_state_dict(self.value_net_global.state_dict())
        self.target_net_total.load_state_dict(self.value_net_total.state_dict())
        self.target_net.eval()
        # self.target_net_other.eval()
        # self.target_net_global.eval()
        self.target_net_total.eval()
        logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)
        # self.value_net_other.to(self.device)
        # self.value_net_global.to(self.device)
        self.value_net_total.to(self.device)
        self.target_net.to(self.device)
        # self.target_net_other.to(self.device)
        # self.target_net_global.to(self.device)
        self.target_net_total.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.parameters(),
                                           **self.config["optimizer"])
        # self.optimizer_other = optimizer_factory(self.config["optimizer"]["type"],
        #                                    self.value_net_other.parameters(),
        #                                    **self.config["optimizer"])
        # self.optimizer_global = optimizer_factory(self.config["optimizer"]["type"],
        #                                    self.value_net_global.parameters(),
        #                                    **self.config["optimizer"])
        self.optimizer_total = optimizer_factory(self.config["optimizer_total"]["type"],
                                           self.value_net_total.parameters(),
                                           **self.config["optimizer_total"])
        self.steps = 0

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # def step_optimizer_other(self, loss):
    #     # Optimize the model
    #     self.optimizer_other.zero_grad()
    #     loss.backward()
    #     for param in self.value_net_other.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.optimizer_other.step()
    #
    # def step_optimizer_global(self, loss):
    #     # Optimize the model
    #     self.optimizer_global.zero_grad()
    #     loss.backward()
    #     for param in self.value_net_global.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.optimizer_global.step()

    def step_optimizer_total(self, loss):
        # Optimize the model
        self.optimizer_total.zero_grad()
        loss.backward()
        for param in self.value_net_total.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_total.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values, target_state_action_value)
        return loss, target_state_action_value, batch

    # # GCN
    # def compute_bellman_residual_other(self, batch, target_state_action_value=None):
    #     # Compute concatenate the batch elements
    #     if not isinstance(batch.state, torch.Tensor):
    #         # logger.info("Casting the batch to torch.tensor")
    #         state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
    #         action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
    #         reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
    #         next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
    #         terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
    #         batch = Transition(state, action, reward, next_state, terminal, batch.info)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken
    #
    #     obs_temp = batch.state[:, :, 1:3]
    #     obs_next_temp = batch.next_state[:, :, 1:3]
    #     obs_temp = np.array(obs_temp.cpu())
    #     obs_next_temp = np.array(obs_next_temp.cpu())
    #
    #     state_action_values = self.value_net_other(obs_temp)
    #     state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    #
    #     if target_state_action_value is None:
    #         with torch.no_grad():
    #             # Compute V(s_{t+1}) for all next states.
    #             next_state_values = torch.zeros(batch.reward.shape).to(self.device)
    #             if self.config["double"]:
    #                 # Double Q-learning: pick best actions from policy network
    #                 _, best_actions = self.value_net_other(obs_next_temp).max(1)
    #                 # Double Q-learning: estimate action values from target network
    #                 best_values = self.target_net_other(obs_next_temp).gather(1, best_actions.unsqueeze(1)).squeeze(1)
    #             else:
    #                 best_values, _ = self.target_net_other(obs_next_temp).max(1)
    #             next_state_values[~batch.terminal] = best_values[~batch.terminal]
    #             # Compute the expected Q values
    #             target_state_action_value = batch.reward + self.config["gamma"] * next_state_values
    #
    #     # Compute loss
    #     loss = self.loss_function(state_action_values, target_state_action_value)
    #     return loss, target_state_action_value, batch

    # def compute_bellman_residual_other(self, batch, target_state_action_value=None):
    #     # Compute concatenate the batch elements
    #     if not isinstance(batch.state, torch.Tensor):
    #         # logger.info("Casting the batch to torch.tensor")
    #         state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
    #         action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
    #         reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
    #         next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
    #         terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
    #         batch = Transition(state, action, reward, next_state, terminal, batch.info)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken
    #     state_action_values = self.value_net_other(batch.state)
    #     state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    #
    #     if target_state_action_value is None:
    #         with torch.no_grad():
    #             # Compute V(s_{t+1}) for all next states.
    #             next_state_values = torch.zeros(batch.reward.shape).to(self.device)
    #             if self.config["double"]:
    #                 # Double Q-learning: pick best actions from policy network
    #                 _, best_actions = self.value_net_other(batch.next_state).max(1)
    #                 # Double Q-learning: estimate action values from target network
    #                 best_values = self.target_net_other(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
    #             else:
    #                 best_values, _ = self.target_net_other(batch.next_state).max(1)
    #             next_state_values[~batch.terminal] = best_values[~batch.terminal]
    #             # Compute the expected Q values
    #             target_state_action_value = batch.reward + self.config["gamma"] * next_state_values
    #
    #     # Compute loss
    #     loss = self.loss_function(state_action_values, target_state_action_value)
    #     return loss, target_state_action_value, batch

    # def compute_bellman_residual_global(self, batch, target_state_action_value=None):
    #     # Compute concatenate the batch elements
    #     if not isinstance(batch.state, torch.Tensor):
    #         # logger.info("Casting the batch to torch.tensor")
    #         state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
    #         action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
    #         reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
    #         next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
    #         terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
    #         batch = Transition(state, action, reward, next_state, terminal, batch.info)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken
    #     state_action_values = self.value_net_global(batch.state)
    #     state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)
    #
    #     if target_state_action_value is None:
    #         with torch.no_grad():
    #             # Compute V(s_{t+1}) for all next states.
    #             next_state_values = torch.zeros(batch.reward.shape).to(self.device)
    #             if self.config["double"]:
    #                 # Double Q-learning: pick best actions from policy network
    #                 _, best_actions = self.value_net_global(batch.next_state).max(1)
    #                 # Double Q-learning: estimate action values from target network
    #                 best_values = self.target_net_global(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
    #             else:
    #                 best_values, _ = self.target_net_global(batch.next_state).max(1)
    #             next_state_values[~batch.terminal] = best_values[~batch.terminal]
    #             # Compute the expected Q values
    #             target_state_action_value = batch.reward + self.config["gamma"] * next_state_values
    #
    #     # Compute loss
    #     loss = self.loss_function(state_action_values, target_state_action_value)
    #     return loss, target_state_action_value, batch

    def compute_bellman_residual_total(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values_total = self.value_net_total(batch.state) #32个值 125×32
        # with open('mt_tuple.pickle', 'wb') as file:
        #     pickle.dump(state_action_values_total, file)
        # with open('my_action.pickle', 'wb') as file:
        #     pickle.dump(batch.action, file)
        action_index = batch.action[:, 0] * 5 ** 0 + batch.action[:, 1] * 5 ** 1 + batch.action[:, 2] * 5 ** 2
        # state_action_values_total = state_action_values_total.transpose(0,1)
        # 这句话要将动作与奖励对应上，只输出一个值的话先屏蔽，但修改要修
        state_action_values_total = state_action_values_total.gather(1, action_index.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net_total(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net_total(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net_total(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values_total, target_state_action_value)
        return loss, target_state_action_value, batch

    def compute_bellman_residual_reg(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values_total = self.value_net_total(batch.state) #32个值 32×125
        action_index = batch.action[:, 0] * 5 ** 0 + batch.action[:, 1] * 5 ** 1 + batch.action[:, 2] * 5 ** 2
        state_action_values_total = state_action_values_total.gather(1, action_index.unsqueeze(1)).squeeze(1)
        # 这句话要将动作与奖励对应上，只输出一个值的话先屏蔽，但修改要修
        # state_action_values_total = state_action_values_total.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                state_action_values1 = self.value_net(batch.state[:, :5, :]).max(1)
                state_action_values2 = self.value_net(batch.state[:, 5:10, :]).max(1)
                state_action_values3 = self.value_net(batch.state[:, 10:, :]).max(1)
                # Compute the expected Q values
                target_state_action_value = state_action_values1[0] + state_action_values2[0] + state_action_values3[0]

        # Compute loss
        loss = self.loss_function(state_action_values_total, target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    # def get_batch_other_effects_values(self, states):
    #     # value_net_other旨在探究某一个其他智能体对当前智能体决策的影响
    #     return self.value_net_other(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    # def get_batch_global_effects_values(self, states):
    #     # value_net_other旨在探究某一个其他智能体对当前智能体决策的影响
    #     return self.value_net_global(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    def get_state_action_values_total(self, states):
    # 对联合动作做估计
        return self.value_net_total(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()



    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        obs_shape = self.env.observation_space.shape if isinstance(self.env.observation_space, spaces.Box) else \
            self.env.observation_space.spaces[0].shape
        model_input = torch.zeros((1, *obs_shape), dtype=torch.float, device=self.device)
        self.writer.add_graph(self.value_net, input_to_model=(model_input,)),
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)
