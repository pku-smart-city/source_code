from lib.metrics import mask_evaluation_np, risk_change_rate_torch,atc_round
import numpy as np
import pandas as pd
import torch
import sys
import os
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.myfunction import kl_div, js_div, log_test_results, logger_info, view_tensor
import time
import random
from torch.utils.data import DataLoader

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H) == 8760*48*20*20
        """
        train_temp = np.transpose(
            train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))  # k*48
        self.max = np.max(train_temp, axis=0)  # 48个最大值
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)

        0:risk(numeric,sum)
        1~24:time_period，(one-hot)
        25~31:day_of_week，(one-hot)
        32:holiday，(one-hot)
        33~39:POI (numeric)
        40:temperature (numeric)
        41:Clear,(one-hot)
        42:Cloudy，(one-hot)
        43:Rain，(one-hot)
        44:Snow，(one-hot)
        45:Mist，(one-hot)
        46:inflow(numeric)
        47:outflow(numeric)
        """

        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 32:39] = (data[:, 32:39] - self.min[32:39]) / \
            (self.max[32:39] - self.min[32:39])
        data[:, 40] = (data[:, 40] - self.min[40]) / \
            (self.max[40] - self.min[40])
        data[:, 46] = (data[:, 46] - self.min[46]) / \
            (self.max[46] - self.min[46])
        data[:, 47] = (data[:, 47] - self.min[47]) / \
            (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(
            train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))  # (T*W*H,D)
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33] = (data[:, 33] - self.min[33]) / \
            (self.max[33] - self.min[33])
        data[:, 39] = (data[:, 39] - self.min[39]) / \
            (self.max[39] - self.min[39])
        data[:, 40] = (data[:, 40] - self.min[40]) / \
            (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def mask_loss(predicts, labels, region_mask, data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size, pre_len, _, _ = predicts.shape
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    loss = ((labels-predicts)*region_mask)**2
    if data_type == 'nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type == 'chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 1/17)
        index_3 = (labels > 1/17) & (labels <= 2/17)
        index_4 = labels > 2/17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return torch.mean(loss)


@torch.no_grad()
def compute_loss(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                 grid_node_map, global_step, device,
                 data_type='nyc'):
    """compute val/test loss

    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        device {Device} -- GPU

    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp = []
    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        l = mask_loss(net(feature, target_time, graph_feature, road_adj, risk_adj,
                      poi_adj, grid_node_map), label, risk_mask, data_type)  # l的shape：(1,)
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                         grid_node_map, global_step, scaler, device):
    """predict val/test, return metrics

    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU

    Returns:
        np.float32 -- RMSE，Recall，MAP
        np.array -- label and pre，shape(num_sample,pre_len,W,H)

    """
    # print("数据类型", type(net), type(dataloader), type(risk_mask), type(road_adj), type(grid_node_map),)
    net.eval()
    prediction_list = []
    label_list = []
    for feature, target_time, graph_feature, label in dataloader:
        # print("特征的数据类型", type(feature))
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        # print("the size of feature is".format())
        prediction_list.append(net(feature, target_time, graph_feature,
                               road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    # 1080*1*20*20=34*32*20*20,把34该array拼成一个
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_, risk_change_rate_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, risk_change_rate_, inverse_trans_pre, inverse_trans_label


@torch.no_grad()
def predict_and_lable(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                      grid_node_map, global_step, scaler, device):
    """
    return prediction,label

    """
    net.eval()
    prediction_list = []
    label_list = []
    futere_all = []  # 装载future
    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        prediction_list.append(net(feature, target_time, graph_feature,
                               road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)
    return prediction, label


def k_map(input_map, k):
    # 输入k值和区域矩阵，返回前K个位置
    '''
    0 1 0 0 1
    1 0 1 1 1
    0 0 0 0 1
    1 0 1 1 0
    '''

    d = input.reshape(1, -1)
    d = np.sort(d, axis=1)
    return


def batch_saliency_map(input_grads, k):
    # 输入大小：32*7*48*20*20,数据类型是tensor

    # input_grads = input_grads.mean(dim=0)  # 先样本平均1*7*48*20*20
    # np.save('input_grads.npy', input_grads.cpu().numpy())
    input_grads = torch.where(
        input_grads > 0., input_grads, torch.zeros_like(input_grads))  # 只保留正值
    input_grads = input_grads.mean(dim=0)
    input_grads = input_grads.mean(dim=0)  # 1*1*1*20*20
    # 243个道路区域有多少被包含在里边
    d1 = input_grads.cpu().numpy()  # 寻找k值
    d = d1
    d = d.reshape(1, -1)
    d = np.sort(d, axis=1)  # 变为1维
    kd = d[0][400-k]
    # print(kd)
    node_saliency_map = np.where(d1 < kd, 0, 1)  # 20*20大小的
    return node_saliency_map

# sum_loss的节点选择方法


def item_saliency_map_zz(input_grads, k, batch_size):
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*48*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []
    view_node = []
    # 243个道路区域有多少被包含在里边
    for i in range(batch_size):
        d1 = input_grads.cpu().numpy()  # 寻找k值
        d1 = d1[i, :, :]
        d = d1
        d = d.reshape(1, -1)
        d = np.sort(d, axis=1)  # 变为1维
        kd = d[0][400-k]
        # 查看数据
        view_node.append(kd)
        # print(kd)
        node_saliency_map = np.where(d1 < kd, 0., 1.)  # 20*20大小的
        node_map.append(node_saliency_map)
    np.save('k_node.npy', np.array(view_node))
    return np.array(node_map)


def item_saliency_map_ff(input_grads, k, batch_size):
    # 反向寻找
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*48*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []

    # 243个道路区域有多少被包含在里边
    for i in range(batch_size):
        d1 = input_grads.cpu().numpy()  # 寻找k值
        d1 = d1[i, :, :]
        d = d1
        d = d.reshape(1, -1)
        d = np.sort(d, axis=1)  # 变为1维
        kd = d[0][50]
        # print(kd)
        node_saliency_map = np.where(d1 < kd, 1., 0.)  # 20*20大小的
        node_map.append(node_saliency_map)
    return np.array(node_map)


def saliency(step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
             grid_node_map, kk, scaler, device, data_type='nyc'):
    # 输入特征32*7*48*20*20
    net.train()
    saliency_store = []
    i = 0
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_saliency = feature
        i = i+1
        X = feature
        saliency_steps = 5
        target_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)

        # print(label.shape)
        # break

        X = X.to(device)
        X_saliency = X_saliency.to(device)
        X_saliency.requires_grad = True  # 先放进去在设置
        X_saliency.retain_grad()
        net.to(device)
        for _ in range(saliency_steps):
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                 grid_node_map), label)

            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(
                X_saliency, 0, 1.0), requires_grad=True)
        saliency_map = batch_saliency_map(inputs_grad, 50)  # 应该修改为kk
        with open('./中间数据.txt', 'a') as f:
            f.write('第{}轮样本数据，大小是：{}\n'.format(i, inputs_grad.shape))
            f.close()
        ppp = np.ones((20, 20),)
        risk_mask = risk_mask.astype(np.int32)
        ppp = np.where((saliency_map == risk_mask) &
                       (risk_mask == np.ones_like(risk_mask)), np.ones_like(risk_mask), np.zeros_like(risk_mask))
        saliency_store.append(ppp.sum())
    # print('修改测试')
    np.save('ppp.npy', ppp)
    np.save('risk_mask.npy', risk_mask)
    return sum(saliency_store)/len(saliency_store)/50


# 改进的方法，对损失函数做改进
def saliency_loss(step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                  grid_node_map, kk, ro, scaler, device, data_type='nyc'):
    # 输入特征32*7*48*20*20
    net.train()
    saliency_store = []
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击32*1*20*20
        X_saliency = feature
        X = feature
        saliency_steps = 5
        target_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)

        X = X.to(device)
        X_saliency = X_saliency.to(device)
        X_saliency.requires_grad = True  # 先放进去在设置
        X_saliency.retain_grad()

        net.to(device)
        for _ in range(saliency_steps):
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = wight_loss(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                               grid_node_map), label, ro)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(
                X_saliency, 0, 1.0), requires_grad=True)
        saliency_map = batch_saliency_map(inputs_grad, 50)  # 应该修改为kk
        ppp = np.ones((20, 20),)
        risk_mask = risk_mask.astype(np.int32)
        ppp = np.where((saliency_map == risk_mask) &
                       (risk_mask == np.ones_like(risk_mask)), np.ones_like(risk_mask), np.zeros_like(risk_mask))
        saliency_store.append(ppp.sum())
    print('修改测试')

    return sum(saliency_store)/len(saliency_store)/50

# 改进的方法，对损失函数做改进


def saliency_sumloss_map(step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                         grid_node_map, kk,  scaler, device, data_type='nyc'):
    # 输入特征32*7*48*20*20
    net.train()
    saliency_store = []
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击32*1*20*20
        X_saliency = feature
        X = feature
        saliency_steps = 5
        target_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)
        risk_mask_use = torch.from_numpy(risk_mask).to(device)
        X = X.to(device)
        X_saliency = X_saliency.to(device)
        X_saliency.requires_grad = True  # 先放进去在设置
        X_saliency.retain_grad()

        net.to(device)
        for _ in range(saliency_steps):
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = sum_loss(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                             grid_node_map), risk_mask_use)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(
                X_saliency, 0, 1.0), requires_grad=True)
        saliency_map = batch_saliency_map(inputs_grad, 50)  # 应该修改为kk
        ppp = np.ones((20, 20),)
        risk_mask = risk_mask.astype(np.int32)
        ppp = np.where((saliency_map == risk_mask) &
                       (risk_mask == np.ones_like(risk_mask)), np.ones_like(risk_mask), np.zeros_like(risk_mask))
        saliency_store.append(ppp.sum())
    print('修改测试')

    return saliency_map

# 输入标签矩阵或预测矩阵，输出loss,与标签值无关


def sum_loss(a, risk_mask):
    return torch.sum(a*risk_mask)


def sum_exp_loss(a, risk_mask):
    return torch.sum(torch.exp(a*risk_mask))


def sum_risk_file(a, pre_num, device):  # a为5、10、20、30、40
    filename = 'data/nyc/risk_mask_{}.npy'.format(pre_num)
    risk_file = np.load(filename)
    risk_file = torch.from_numpy(risk_file)
    risk_file = risk_file.to(device=device)
    return torch.sum(a*risk_file)


def wight_loss(a, b, ro):
    e = torch.where(a > 0, a, ro * torch.ones_like(a))
    d = (a-b)*(a-b)*e
    d = d.sum()
    return d

# 节点选择后进行攻击


def attack_saliency(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                    grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():

    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature

        # 选择敏感节点
        X_saliency = feature
        X = feature
        saliency_steps = 5
        target_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)
        risk_mask_use = torch.from_numpy(risk_mask).to(device)
        X = X.to(device)
        X_saliency = X_saliency.to(device)
        X_saliency.requires_grad = True  # 先放进去在设置
        X_saliency.retain_grad()

        net.to(device)
        for _ in range(saliency_steps):
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = sum_loss(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                             grid_node_map), risk_mask_use)
            loss_saliency.backward()

            eta_s = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta_s, requires_grad=True)
            eta_s = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta_s, requires_grad=True)
            X_saliency = Variable(torch.clamp(
                X_saliency, 0, 1.0), requires_grad=True)
        saliency_map = batch_saliency_map(inputs_grad, 50)  # 应该修改为kk

        # 进行攻击
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            # print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            # print(device)
            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            # print("2:", feature.requires_grad)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature
            saliency_map_t = torch.from_numpy(saliency_map).to(device)
            eta = step_size * X_pgd.grad.data.sign()*saliency_map_t
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)

        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    np.save('saliency_map.npy', saliency_map)

    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def attack(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            # print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            # print(device)
            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            # print("2:", feature.requires_grad)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature

            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)

        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def attack_kl(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
              grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            # print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            # print(device)
            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            # print("2:", feature.requires_grad)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = kl_div(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map), label)
            loss.backward()

            X_pgd = feature

            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)

        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def attack_js(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
              grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            # print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            # print(device)
            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            # print("2:", feature.requires_grad)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = js_div(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map), label)
            loss.backward()

            X_pgd = feature

            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)

        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def random_attack(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                  grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature
        X_pgd001.to(device)
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature

            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            # 修改的地方
            eta = torch.clamp(torch.rand(
                size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def fgsm_attack(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                grid_node_map, scaler, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = feature

        opt = optim.SGD([feature], lr=1e-3)
        opt.zero_grad()
        feature = Variable(feature.data, requires_grad=True)
        arget_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)
        feature = feature.to(device)
        # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
        net.to(device)
        feature.retain_grad()
        with torch.enable_grad():
            loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                    grid_node_map), label)
        loss.backward()

        X_pgd = feature
        # X_fgsm = feature
        X_fgsm = Variable(torch.clamp(feature.data + epsilon *
                          feature.grad.data.sign(), 0.0, 1.0), requires_grad=True)
        feature = X_fgsm

        X_pgd001 = X_pgd001.to(device)
        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        # a记录改变量的大小
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        # acc_prediction_list是预测值，label_list是标签
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def min_attack(num_steps, step_size, epsilon, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
               grid_node_map, scaler, device, decay_factor=1.0, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    net.train()
    acc_prediction_list = []
    label_list = []
    a = []
    # with torch.no_grad():
    for feature, target_time, graph_feature, label in dataloader:
        # 对特征进行攻击
        X_pgd001 = Variable(feature.data, requires_grad=True)
        previous_grad = torch.zeros_like(feature.data)
        previous_grad = previous_grad.to(device)
        X_pgd001 = X_pgd001.to(device)

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()

            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)

            # road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)

            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature
            grad = X_pgd.grad.data / \
                torch.mean(torch.abs(X_pgd.grad.data),
                           [1, 2, 3, 4], keepdim=True)
            # 修改了，第一个参数是batchsize的大小
            previous_grad = decay_factor * previous_grad + grad
            X_pgd = Variable(X_pgd.data + step_size *
                             previous_grad.sign(), requires_grad=True)
            eta = torch.clamp(X_pgd.data - X_pgd001.data, -epsilon, epsilon)
            X_pgd = Variable(X_pgd001.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            feature = X_pgd

        eta = torch.clamp(X_pgd.data - X_pgd001.data, -epsilon, epsilon)
        X_pgd = Variable(X_pgd001.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        loss_fn3 = torch.nn.MSELoss(reduction='mean')
        a.append(loss_fn3(X_pgd001, feature).cpu())
        acc_prediction_list.append(net(feature, arget_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        '''
        acc_prediction_list.append(net(feature, target_time, graph_feature,
                                       road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())

        '''
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label, sum(a)/len(a)


def prenoise_pgd(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
                 grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    严格的pgd攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    Random_noise = cfgs.Random_noise
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器
    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)
        feature = Variable(feature.data, requires_grad=True)
        feature = feature.to(device)
        net.to(device)

        if Random_noise:
            random_noise = torch.FloatTensor(
                *feature.shape).uniform_(-epsilon/10, epsilon/10).cuda()
            feature = Variable(feature.data + map *
                               random_noise, requires_grad=True)
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..一个bug,1670行查看会出错X_pgd.grad.data
            eta = step_size * feature.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        # 输出一下信息
        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def nonenoise_pgd(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
                  grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    严格的pgd攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon

    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器
    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)

        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def min(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
        grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    严格的pgd攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon

    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器
    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        X_pgd001 = X_pgd001.to(device)
        # 添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)
        previous_grad = torch.zeros_like(feature.data)
        previous_grad = previous_grad.to(device)
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..
            grad = X_pgd.grad.data / \
                torch.mean(torch.abs(X_pgd.grad.data),
                           [1, 2, 3, 4], keepdim=True)
            previous_grad = torch.tensor(
                1.0) * previous_grad + grad  # 1.0可以是一个参数，写入cfgs文件配置中
            X_pgd = Variable(X_pgd.data + step_size *
                             previous_grad.sign(), requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            feature = X_pgd
        eta = torch.clamp(X_pgd.data - X_pgd001.data, -epsilon, epsilon) * map
        X_pgd = Variable(X_pgd001.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = X_pgd.to(device)
        # 每个batch的三个量
        batch_adv = net(X_pgd, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def fgsm(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
         grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    严格的pgd攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    # fgsm只迭代一次
    num_steps = 1
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon

    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器
    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)

        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def rcr_pgd(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
            grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = sum_loss(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                    grid_node_map),  risk_mask_use)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def kl_pgd(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = kl_div(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre

def kl_pgd_round(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = kl_div(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        #对feature取整
        feature = atc_round(feature,X_pgd001)#攻击后攻击前
        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def JS_pgd(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = js_div(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def rand_attack(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
                grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature
        # 生成对抗样本
        for _ in range(num_steps):  # 偷懒写法
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            # 修改的地方
            eta = torch.clamp(torch.randn(
                size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map*step_size
            # 均匀分布与随机分布
            #eta = torch.clamp(torch.rand(size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def log_test_csv(adv_val_predict, val_target, val_predict,  cfgs, select_node_name, select_ack_name, file_name, risk_mask):
    '''
    节点选择方法，攻击方法
    'dataset', 'model','node_select', 'method','K' ,'batch size'
    计算并保存结果'clean_RMSE', 'clean_recaall', 'clean_MAP', 'clean_RCR',
              'adv_RMSE', 'adv_recaall', 'adv_MAP', 'adv_RCR',
              'local_adv_RMSE', 'local_adv_recaall', 'local_adv_MAP', 'local_adv_RCR'

    '''
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(data_set)
    model_name = cfgs.backbone
    metric_list.append(model_name)
    metric_list.append(select_node_name)
    metric_list.append(select_ack_name)
    metric_list.append(cfgs.K)
    metric_list.append(cfgs.batchsize)

    clean_RMSE, clean_recaall, clean_MAP, clean_RCR = mask_evaluation_np(
        val_target, val_predict, risk_mask, 0)
    adv_RMSE, adv_recaall, adv_MAP, adv_RCR = mask_evaluation_np(
        val_target, adv_val_predict, risk_mask, 0)
    local_adv_RMSE, local_adv_recaall, local_adv_MAP, local_adv_RCR = mask_evaluation_np(
        val_predict, adv_val_predict, risk_mask, 0)

    metric_list.append(clean_RMSE)
    metric_list.append(clean_recaall)
    metric_list.append(clean_MAP)
    metric_list.append(clean_RCR)
    metric_list.append(adv_RMSE)
    metric_list.append(adv_recaall)
    metric_list.append(adv_MAP)
    metric_list.append(adv_RCR)
    metric_list.append(local_adv_RMSE)
    metric_list.append(local_adv_recaall)
    metric_list.append(local_adv_MAP)
    metric_list.append(local_adv_RCR)
    # 所有batch跑完，1个epoch保存

    log_test_results(cfgs.model_dir, metric_list, file_name)


def node_map(select_name, cfgs, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
             grid_node_map, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    # select_name:节点选择方法
    # Random:随机处理
    # num_nodes：节点数量
    # K：选择攻击的节点数量
    for_random = cfgs.for_random
    num_nodes = cfgs.num_nodes
    K = cfgs.K
    step_size = cfgs.step_size
    epsilon = cfgs.epsilon
    net.train()
    map = []
    # with torch.no_grad():
    if select_name == 'none':
        a = np.ones((1080, 20, 20),)
        return a

    if select_name == 'random':
        # 返回np矩阵1080*20*20，值为1 的地方为选中的节点(K个)
        a = np.zeros((1080, 20, 20),)
        b = a.reshape(1080, -1)
        for len_future in range(1080):
            list = [i for i in range(num_nodes)]
            index = random.sample(list, K)
            b[len_future][index] = 1
        return b.reshape(1080, 20, 20)
    if select_name == 'saliency_loss_mse':
        for feature, target_time, graph_feature, label in dataloader:
            # 选择敏感节点
            X_saliency = feature
            X = feature
            saliency_steps = 5
            target_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            risk_mask_use = torch.from_numpy(risk_mask).to(device)
            X = X.to(device)
            X_saliency = X_saliency.to(device)
            X_saliency.requires_grad = True  # 先放进去在设置
            X_saliency.retain_grad()

            net.to(device)
            for _ in range(saliency_steps):
                if for_random:
                    random_noise = torch.FloatTensor(
                        *X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                    X_saliency = Variable(
                        X_saliency.data + random_noise, requires_grad=True)
                opt_saliency = optim.SGD([X_saliency], lr=1e-3)
                opt_saliency.zero_grad()
                with torch.enable_grad():
                    loss_saliency = nn.MSELoss()(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                     grid_node_map), label)
                loss_saliency.backward()

                eta_s = step_size * X_saliency.grad.data.sign()
                inputs_grad = X_saliency.grad.data
                # X_saliency随着迭代变化
                X_saliency = Variable(
                    X_saliency.data + eta_s, requires_grad=True)
                eta_s = torch.clamp(
                    X_saliency.data - X.data, -epsilon, epsilon)
                X_saliency = Variable(X.data + eta_s, requires_grad=True)
                X_saliency = Variable(torch.clamp(
                    X_saliency, 0, 1.0), requires_grad=True)
            saliency_map = item_saliency_map_zz(
                inputs_grad, K, feature.shape[0])
            map.append(saliency_map)
        select_name = np.concatenate(map, 0)
        print(select_name.shape)
        return select_name
    if select_name == 'sum_yi_loss':  # 不迭代
        for feature, target_time, graph_feature, label in dataloader:
            # 选择敏感节点
            X_saliency = feature
            X = feature
            target_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            risk_mask_use = torch.from_numpy(risk_mask).to(device)
            X = X.to(device)
            X_saliency = X_saliency.to(device)
            X_saliency.requires_grad = True  # 先放进去在设置
            X_saliency.retain_grad()

            net.to(device)

            if for_random:
                random_noise = torch.FloatTensor(
                    *X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(
                    X_saliency.data + random_noise, requires_grad=True)
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = sum_loss(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                             grid_node_map), risk_mask_use)
            loss_saliency.backward()
            inputs_grad = X_saliency.grad.data
            # 看看梯度信息
            # view_tensor(inputs_grad)
            saliency_map = item_saliency_map_zz(
                inputs_grad, K, feature.shape[0])
            map.append(saliency_map)
        select_name = np.concatenate(map, 0)
        print(select_name.shape)
        return select_name
    if select_name == 'sum_yi_loss_test':  # 不迭代
        for feature, target_time, graph_feature, label in dataloader:
            # 选择敏感节点
            X_saliency = feature
            X = feature
            target_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            risk_mask_use = torch.from_numpy(risk_mask).to(device)
            X = X.to(device)
            X_saliency = X_saliency.to(device)
            X_saliency.requires_grad = True  # 先放进去在设置
            X_saliency.retain_grad()

            net.to(device)

            if for_random:
                random_noise = torch.FloatTensor(
                    *X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(
                    X_saliency.data + random_noise, requires_grad=True)
            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = sum_risk_file(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                  grid_node_map), 40, device)
            loss_saliency.backward()
            inputs_grad = X_saliency.grad.data
            # 看看梯度信息
            # view_tensor(inputs_grad)
            saliency_map = item_saliency_map_zz(
                inputs_grad, K, feature.shape[0])
            map.append(saliency_map)
        select_name = np.concatenate(map, 0)
        print(select_name.shape)
        return select_name
