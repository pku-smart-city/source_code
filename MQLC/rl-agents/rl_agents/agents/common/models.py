import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F

from rl_agents.configuration import Configurable


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *input):
        if self.normalize:
            input = (input.float() - self.mean.float()) / self.std.float()
        return NotImplementedError


class MultiLayerPerceptron(BaseModule, Configurable):
    def __init__(self, config, input_num):
        super().__init__()
        Configurable.__init__(self, config)
        if input_num == 1:
            sizes = [self.config["in"]] + self.config["layers"]
        elif input_num == 2:
            sizes = [self.config["in_other"]] + self.config["layers"]
        elif input_num == 3:
            sizes = [self.config["in_global"]] + self.config["layers"]
        elif input_num ==4:
            sizes = [self.config["in_total"]] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            if input_num == 4:
                self.predict = nn.Linear(sizes[-1], self.config["out_total"])
            else:
                self.predict = nn.Linear(sizes[-1], self.config["out"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        return x

class MultiLayerPerceptron_he(BaseModule, Configurable):
    def __init__(self, config, input_num):
        super().__init__()
        Configurable.__init__(self, config)
        if input_num == 1:
            sizes = [self.config["in"]] + self.config["layers"]
        elif input_num == 2:
            sizes = [self.config["in_other"]] + self.config["layers"]
        elif input_num == 3:
            sizes = [self.config["in_global"]] + self.config["layers"]
        elif input_num ==4:
            sizes = [self.config["in_total"]] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        # GCN层
        hidden_dim = 256
        hidden_dim_glo = 8
        self.conv1 = nn.Linear(2, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        # global层
        self.global_layers = nn.Linear(2, hidden_dim_glo)
        if self.config.get("out", None):
            if input_num == 4:
                self.predict = nn.Linear(sizes[-1] + hidden_dim + hidden_dim_glo, self.config["out_total"])
                # self.predict = nn.Linear(sizes[-1] + hidden_dim, self.config["out_total"])
            else:
                self.predict = nn.Linear(sizes[-1] + hidden_dim + hidden_dim_glo, self.config["out"])
                # self.predict = nn.Linear(sizes[-1] + hidden_dim, self.config["out"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        # GCN处理的部分
        points_list = x[:, :, 1:3]
        v_list = x[:, :, 1:5]

        if len(points_list.shape) > 2:
            outputs_gcn = []
            outputs_glo = []
            for points in points_list:
                distances = torch.norm(points[:, None] - points, dim=2)  # 计算点之间的欧氏距离
                threshold = 0.3  # 距离阈值
                adj_matrix = (distances < threshold).float()  # 基于距离阈值构建邻接矩阵
                rowsum = torch.sum(adj_matrix, dim=1)
                D = torch.diag(1.0 / torch.sqrt(rowsum))
                normalized_adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)
                node_features = points.clone().detach()
                normalized_adj_matrix = normalized_adj_matrix.clone().detach()
                normalized_adj_matrix = normalized_adj_matrix.float()
                gcn_en = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
                gcn_en = self.relu(gcn_en)
                gcn_en = self.conv2(torch.matmul(normalized_adj_matrix, gcn_en))
                gcn_en = self.fc(gcn_en.mean(dim=0, keepdim=True))
                outputs_gcn.append(gcn_en)
            for v1 in v_list:
                v2 = v1[:, 2:4]
                speeds = torch.norm(v2, dim=1)
                average_speed = torch.mean(speeds)
                v3 = v1[:, :2]
                distances = torch.norm(v3[:, None, :] - v3[None, :, :], dim=2)
                max_distance = torch.max(distances)
                vehicle_density = 1.0 / max_distance

                a = average_speed.item().__float__()
                b = vehicle_density.item().__float__()
                input_data = torch.tensor([a, b], device='mps', dtype=torch.float32)
                glo_en = self.global_layers(input_data)
                glo_en = self.relu(glo_en).unsqueeze(0)
                outputs_glo.append(glo_en)
            gcn_en = outputs_gcn
            glo_en = outputs_glo
        else:
            velocity_squared_sum = (v_list ** 2).sum(dim=2)
            vehicle_speeds = torch.sqrt(velocity_squared_sum)
            average_speed = vehicle_speeds.mean()
            distances_1 = torch.norm(points_list[:, None, :, :] - points_list[:, :, None, :], dim=-1)
            max_distance = distances_1.max()
            epsilon = 1e-6  # 避免除以零的小正数
            vehicle_density = 1 / (max_distance + epsilon)
            a = average_speed.item().__float__()
            b = vehicle_density.item().__float__()
            input_data = torch.tensor([a, b], device='mps', dtype=torch.float32)
            glo_en = self.global_layers(input_data)
            glo_en = self.relu(glo_en).unsqueeze(0)

            points = np.array(points_list.cpu())
            distances = torch.norm(points[:, None] - points, dim=2)  # 计算点之间的欧氏距离
            threshold = 0.3  # 距离阈值
            adj_matrix = (distances < threshold).float()  # 基于距离阈值构建邻接矩阵
            rowsum = torch.sum(adj_matrix, dim=1)
            D = torch.diag(1.0 / torch.sqrt(rowsum))
            normalized_adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)
            node_features = torch.tensor(points, dtype=torch.float)
            normalized_adj_matrix = normalized_adj_matrix.clone().detach()
            normalized_adj_matrix = normalized_adj_matrix.float()
            gcn_en = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
            gcn_en = self.relu(gcn_en)
            gcn_en = self.conv2(torch.matmul(normalized_adj_matrix, gcn_en))
            gcn_en = self.fc(gcn_en.mean(dim=0, keepdim=True))

        # MLP的处理部分，只要最后预测的时候输出正确的就可以
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors 变成32×25或1×25的
        for layer in self.layers:
            x = self.activation(layer(x))
        gcn_en = torch.cat(gcn_en, dim=0)
        glo_en = torch.cat(glo_en, dim=0)
        x = torch.cat((x, gcn_en, glo_en), dim=1)
        # x = torch.cat((x, gcn_en), dim=1)
        if self.config.get("out", None):
            x = self.predict(x)
        return x

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 5)
        self.relu = nn.ReLU()

    def forward(self, points_list):
        points_list = np.array(points_list)
        if len(points_list.shape) > 2:
            outputs = []
            for points in points_list:
                distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)  # 计算点之间的欧氏距离
                threshold = 0.3  # 距离阈值
                adj_matrix = (distances < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
                rowsum = np.sum(adj_matrix, axis=1)
                D = np.diag(1.0 / np.sqrt(rowsum))
                normalized_adj_matrix = np.matmul(np.matmul(D, adj_matrix), D)
                node_features = torch.tensor(points, dtype=torch.float).to("cuda:0")
                normalized_adj_matrix = torch.tensor(normalized_adj_matrix).to("cuda:0")
                normalized_adj_matrix = normalized_adj_matrix.float()
                x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
                x = self.relu(x)
                x = self.conv2(torch.matmul(normalized_adj_matrix, x))
                x = self.fc(x.mean(dim=0, keepdim=True))
                outputs.append(x)
            return torch.cat(outputs, dim=0)
        else:
            points = np.array(points_list)
            distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)  # 计算点之间的欧氏距离
            threshold = 0.5  # 距离阈值
            adj_matrix = (distances < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
            rowsum = np.sum(adj_matrix, axis=1)
            D = np.diag(1.0 / np.sqrt(rowsum))
            normalized_adj_matrix = np.matmul(np.matmul(D, adj_matrix), D)
            node_features = torch.tensor(points, dtype=torch.float)
            # 前向传播
            normalized_adj_matrix = torch.tensor(normalized_adj_matrix).to("cuda:0")
            node_features = node_features.clone().detach().to("cuda:0")
            normalized_adj_matrix = normalized_adj_matrix.float()
            x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
            x = self.relu(x)
            x = self.conv2(torch.matmul(normalized_adj_matrix, x))
            x = self.fc(x.mean(dim=0, keepdim=True))
            return x

class DuelingNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config["base_module"]["in"] = self.config["in"]
        self.base_module = model_factory(self.config["base_module"])
        self.config["value"]["in"] = self.base_module.config["layers"][-1]
        self.config["value"]["out"] = 1
        self.value = model_factory(self.config["value"])
        self.config["advantage"]["in"] = self.base_module.config["layers"][-1]
        self.config["advantage"]["out"] = self.config["out"]
        self.advantage = model_factory(self.config["advantage"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "base_module": {"type": "MultiLayerPerceptron", "out": None},
                "value": {"type": "MultiLayerPerceptron", "layers": [], "out": None},
                "advantage": {"type": "MultiLayerPerceptron", "layers": [], "out": None},
                "out": None}

    def forward(self, x):
        x = self.base_module(x)
        value = self.value(x).expand(-1,  self.config["out"])
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.config["out"])


class ConvolutionalNetwork(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"])))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 64
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        return self.head(x)


class EgoAttention(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1))/2
        return result, attention_matrix


class SelfAttention(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_all = self.query_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_all, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, n_entities, self.config["feature_size"]))) + input_all)/2
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        if not self.config["others_embedding_layer"]["in"]:
            self.config["others_embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.ego_embedding = model_factory(self.config["embedding_layer"])
        self.others_embedding = model_factory(self.config["others_embedding_layer"])
        self.self_attention_layer = None
        if self.config["self_attention_layer"]:
            self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
        self.attention_layer = EgoAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "others_embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "self_attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        if self.self_attention_layer:
            self_att, _ = self.self_attention_layer(ego, others, mask)
            ego, others, mask = self.split_input(self_att, mask=mask)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


class AttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.embedding = model_factory(self.config["embedding_layer"])
        self.attention_layer = SelfAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego, others, mask = self.split_input(x)
        ego_embedded_att, _ = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def get_attention_matrix(self, x):
        ego, others, mask = self.split_input(x)
        _, attention_matrix = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute a Scaled Dot Product Attention.
    :param query: size: batch, head, 1 (ego-entity), features
    :param key:  size: batch, head, entities, features
    :param value: size: batch, head, entities, features
    :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
    :param dropout:
    :return: the attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    if model_config["type"] == "ConvolutionalNetwork":  # Assume CHW observation space
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_height"] = int(obs_shape[1])
        model_config["in_width"] = int(obs_shape[2])
    else:
        model_config["in"] = int(np.prod(obs_shape))
        model_config["in_other"] = int(2 * np.prod(obs_shape))
        model_config["in_global"] = int(2)
        model_config["in_total"] = int(3 * np.prod(obs_shape))

    if isinstance(env.action_space, spaces.Discrete):
        model_config["out"] = env.action_space.n
    elif isinstance(env.action_space, spaces.Tuple):
        model_config["out"] = env.action_space.spaces[0].n
        model_config["out_total"] = int(125)


def model_factory(config: dict, num_inputs: int = 1) -> nn.Module:
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config, num_inputs)
    if config["type"] == "MultiLayerPerceptron_he":
        return MultiLayerPerceptron_he(config, num_inputs)
    elif config["type"] == "GCN":
        input_dim = 2  # 输入维度为2，即横纵坐标
        hidden_dim = 64  # 隐藏层维度
        output_dim = 64  # 输出维度为5，对应五个离散动作
        return GCN(input_dim, hidden_dim, output_dim)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(config)
    elif config["type"] == "ConvolutionalNetwork":
        return ConvolutionalNetwork(config)
    elif config["type"] == "EgoAttentionNetwork":
        return EgoAttentionNetwork(config)
    else:
        raise ValueError("Unknown model type")

