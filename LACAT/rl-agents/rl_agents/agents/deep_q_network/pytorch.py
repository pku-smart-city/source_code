import copy
import logging

import numpy
import numpy as np
import torch
from gym import spaces
from rl_agents.agents.common.memory import Transition
from rl_agents.agents.common.models import (
    model_factory,
    size_model_config,
    trainable_parameters,
)
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory
from rl_agents.agents.common.utils import choose_device
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent

logger = logging.getLogger(__name__)


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        size_model_config(self.env, self.config["model"])
        self.value_net = model_factory(self.config["model"])
        self.value_net_total = model_factory(self.config["model"], 4)
        self.target_net = model_factory(self.config["model"])
        self.target_net_total = model_factory(self.config["model"], 4)
        self.joint_action_value_net = model_factory(self.config["model"], joint=True)
        self.target_joint_action_value_net = model_factory(
            self.config["model"], joint=True
        )
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net_total.load_state_dict(self.value_net_total.state_dict())
        self.target_joint_action_value_net.load_state_dict(
            self.joint_action_value_net.state_dict()
        )
        self.target_net.eval()
        self.target_net_total.eval()
        self.target_joint_action_value_net.eval()
        logger.debug(
            "Number of trainable parameters: {}".format(
                trainable_parameters(self.value_net)
            )
        )
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)
        self.value_net_total.to(self.device)
        self.target_net.to(self.device)
        self.target_net_total.to(self.device)
        self.joint_action_value_net.to(self.device)
        self.target_joint_action_value_net.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(
            self.config["optimizer"]["type"],
            self.value_net.parameters(),
            **self.config["optimizer"]
        )
        self.optimizer_total = optimizer_factory(
            self.config["optimizer_total"]["type"],
            self.value_net_total.parameters(),
            **self.config["optimizer_total"]
        )
        self.optimizer_joint = optimizer_factory(
            self.config["optimizer"]["type"],
            self.joint_action_value_net.parameters(),
            **self.config["optimizer"]
        )
        self.steps = 0
        self.gamma = self.config["gamma"]

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def step_optimizer_joint(self, loss):
        self.optimizer_joint.zero_grad()
        loss.backward()
        for param in self.joint_action_value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_joint.step()

    def step_optimizer_total(self, loss):
        # Optimize the model
        self.optimizer_total.zero_grad()
        loss.backward()
        for param in self.value_net_total.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_total.step()

    @staticmethod
    def _softmax(qvalues):
        return torch.exp(qvalues - torch.max(qvalues)) / torch.sum(
            torch.exp(qvalues - torch.max(qvalues))
        )

    @staticmethod
    def _compute_ind_loss(distribution, worst_action_idx, best_action_idx):
        return torch.log(distribution[worst_action_idx] + 1e-8) - torch.log(
            distribution[best_action_idx] + 1e-8
        )

    def get_adv_state(self, state, eps, worst_action_index):
        # MI-FGSM
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        adv_state = copy.deepcopy(state)
        adv_state.requires_grad = True
        momentum = torch.zeros(state.shape).to(self.device)
        for _ in range(10):
            action_values = self.value_net(adv_state)
            prob_distribution = self._softmax(action_values)
            loss = self._compute_ind_loss(
                prob_distribution[-1],
                worst_action_index,
                int(torch.argsort(action_values)[-1][-1]),
            )
            loss.backward(retain_graph=True)
            grad = adv_state.grad.data
            grad = grad / torch.norm(grad, dim=(0, 1, 2), keepdim=True)
            grad = grad + momentum
            momentum = grad
            adv_state.data = adv_state.data + eps * torch.sign(momentum)
            eta = torch.clamp(adv_state.data - state.data, min=-0.1, max=0.1)
            adv_state.data = torch.clamp(state + eta, min=0, max=1)
            adv_state.grad.data.zero_()
        adv_state.requires_grad = False
        return adv_state.cpu().data.numpy()[0]

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.state]), dtype=torch.float))
            ).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.next_state]), dtype=torch.float))
            ).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(
            1, batch.action.unsqueeze(1)
        ).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = (
                        self.target_net(batch.next_state)
                        .gather(1, best_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = (
                    batch.reward + self.gamma * next_state_values
                )

        # Compute loss
        loss = self.loss_function(state_action_values, target_state_action_value)
        return loss, target_state_action_value, batch

    def compute_bellman_residual_total(self, batch, target_state_action_value=None):
        if not isinstance(batch.state, torch.Tensor):
            state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.state]), dtype=torch.float))
            ).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.next_state]), dtype=torch.float))
            ).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        state_action_values_total = self.value_net_total(batch.state)  # 32个值 125×32
        action_index = (
            batch.action[:, 0] * 5**0
            + batch.action[:, 1] * 5**1
            + batch.action[:, 2] * 5**2
        )
        # action_index = batch.action[:, 0] * 5 ** 0 + batch.action[:, 1] * 5 ** 1 + batch.action[:, 2] * 5 ** 2 + batch.action[:, 3] * 5 ** 3 + batch.action[:, 4] * 5 ** 4 + batch.action[:, 5] * 5 ** 5
        # 这句话要将动作与奖励对应上，只输出一个值的话先屏蔽，但修改要修
        state_action_values_total = state_action_values_total.gather(
            1, action_index.unsqueeze(1)
        ).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net_total(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = (
                        self.target_net_total(batch.next_state)
                        .gather(1, best_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                else:
                    best_values, _ = self.target_net_total(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = (
                    batch.reward + self.gamma * next_state_values
                )

        # Compute loss
        loss = self.loss_function(state_action_values_total, target_state_action_value)
        return loss, target_state_action_value, batch

    def compute_bellman_residual_joint(self, joint_batch):
        state_size = len(joint_batch.state)
        if not isinstance(joint_batch.state, torch.Tensor):
            state = torch.cat(
                tuple(torch.tensor(numpy.array([joint_batch.state]), dtype=torch.float))
            ).reshape(state_size, -1)
            action = torch.tensor(joint_batch.action, dtype=torch.long)
            state_action = torch.cat((state, action), dim=1).to(self.device)
            reward = (
                torch.tensor(joint_batch.reward, dtype=torch.float)
                .unsqueeze(1)
                .to(self.device)
            )
            next_state = torch.cat(
                tuple(
                    torch.tensor(
                        numpy.array([joint_batch.next_state]), dtype=torch.float
                    )
                )
            )
            next_state1, next_state2, next_state3 = (
                next_state[:, :5, :],
                next_state[:, 5:10, :],
                next_state[:, 10:15, :],
            )
            _, next_action1 = self.value_net(next_state1.to(self.device)).max(1)
            _, next_action2 = self.value_net(next_state2.to(self.device)).max(1)
            _, next_action3 = self.value_net(next_state3.to(self.device)).max(1)
            next_action = torch.cat(
                (
                    next_action1.unsqueeze(1),
                    next_action2.unsqueeze(1),
                    next_action3.unsqueeze(1),
                ),
                dim=1,
            )
            next_state_action = torch.cat(
                (next_state.reshape(state_size, -1).to(self.device), next_action), dim=1
            )
            terminal = (
                torch.tensor(joint_batch.terminal, dtype=torch.float16)
                .unsqueeze(1)
                .to(self.device)
            )
        joint_action_value = self.joint_action_value_net(state_action)
        with torch.no_grad():
            next_joint_action_value = self.target_joint_action_value_net(
                next_state_action
            )
        targets = reward + self.gamma * (1 - terminal) * next_joint_action_value
        loss = self.loss_function(joint_action_value, targets)
        return loss

    def compute_bellman_residual_reg(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.state]), dtype=torch.float))
            ).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(
                tuple(torch.tensor(numpy.array([batch.next_state]), dtype=torch.float))
            ).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values_total = self.value_net_total(batch.state)  # 32个值 32×125
        action_index = (
            batch.action[:, 0] * 5**0
            + batch.action[:, 1] * 5**1
            + batch.action[:, 2] * 5**2
        )
        # action_index = batch.action[:, 0] * 5 ** 0 + batch.action[:, 1] * 5 ** 1 + batch.action[:, 2] * 5 ** 2 + batch.action[:, 3] * 5 ** 3 + batch.action[:, 4] * 5 ** 4 + batch.action[:, 5] * 5 ** 5
        state_action_values_total = state_action_values_total.gather(
            1, action_index.unsqueeze(1)
        ).squeeze(1)
        # 这句话要将动作与奖励对应上，只输出一个值的话先屏蔽，但修改要修
        # state_action_values_total = state_action_values_total.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                state_action_values1 = self.value_net(batch.state[:, :5, :]).max(1)
                state_action_values2 = self.value_net(batch.state[:, 5:10, :]).max(1)
                state_action_values3 = self.value_net(batch.state[:, 10:15, :]).max(1)
                # state_action_values4 = self.value_net(batch.state[:, 15:20, :]).max(1)
                # state_action_values5 = self.value_net(batch.state[:, 20:25, :]).max(1)
                # state_action_values6 = self.value_net(batch.state[:, 25:, :]).max(1)
                # Compute the expected Q values
                # target_state_action_value = state_action_values1[0] + state_action_values2[0] + state_action_values3[0] + state_action_values4[0] + state_action_values5[0] + state_action_values6[0]
                target_state_action_value = (
                    state_action_values1[0]
                    + state_action_values2[0]
                    + state_action_values3[0]
                )

        # Compute loss
        loss = self.loss_function(state_action_values_total, target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(
            torch.tensor(states, dtype=torch.float).to(self.device)
        ).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        states_tensor = torch.tensor(numpy.array(states), dtype=torch.float).to(
            self.device
        )
        out = self.value_net(states_tensor).cpu().data.numpy()
        return out

    def get_state_action_values_total(self, states):
        # 对联合动作做估计
        return (
            self.value_net_total(
                torch.tensor(numpy.array(states), dtype=torch.float).to(self.device)
            )
            .data.cpu()
            .numpy()
        )

    def get_state_joint_value(self, state_joint_action):
        return (
            self.joint_action_value_net(state_joint_action.to(self.device))
            .data.cpu()
            .numpy()
        )

    def save(self, filename):
        state = {
            'state_dict': self.value_net.state_dict(),
            'state_tot_dict': self.value_net_total.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.value_net_total.load_state_dict(checkpoint['state_tot_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        obs_shape = (
            self.env.observation_space.shape
            if isinstance(self.env.observation_space, spaces.Box)
            else self.env.observation_space.spaces[0].shape
        )
        model_input = torch.zeros(
            (1, *obs_shape), dtype=torch.float, device=self.device
        )
        self.writer.add_graph(self.value_net, input_to_model=(model_input,)),
        self.writer.add_scalar(
            "agent/trainable_parameters", trainable_parameters(self.value_net), 0
        )
