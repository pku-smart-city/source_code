from lib.metrics import mask_evaluation_np
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

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

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
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33:40] = (data[:, 33:40] - self.min[33:40]) / \
            (self.max[33:40] - self.min[33:40])
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
    print("数据类型", type(net), type(dataloader), type(
        risk_mask), type(road_adj), type(grid_node_map),)
    net.eval()
    prediction_list = []
    label_list = []
    for feature, target_time, graph_feature, label in dataloader:
        print("特征的数据类型", type(feature))
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        #print("the size of feature is".format())
        prediction_list.append(net(feature, target_time, graph_feature,
                               road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label


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
            print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            print(device)
            #road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            print("2:", feature.requires_grad)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature
            print(feature.requires_grad)
            print(type(X_pgd.grad))
            print(type(feature.grad))
            print("---------------------------------------------------------------")
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd

        X_pgd001 = X_pgd001.to(device)
        print('*********')
        print(X_pgd001.device)
        print(feature.device)
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
            print("1:", feature.requires_grad)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            print(device)
            #road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
            print("2:", feature.requires_grad)
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
            print('特征的形状：{}'.format(X_pgd.shape))
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
        print("feature.requires_grad:", feature.requires_grad)
        arget_time, graph_feature, label = target_time.to(
            device), graph_feature.to(device), label.to(device)
        feature = feature.to(device)
        print(device)
        #road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)
        print("feature.requires_grad:", feature.requires_grad)
        net.to(device)
        feature.retain_grad()
        with torch.enable_grad():
            loss = nn.MSELoss()(net(feature, arget_time, graph_feature, road_adj, risk_adj, poi_adj,
                                    grid_node_map), label)
        loss.backward()

        X_pgd = feature
        #X_fgsm = feature
        print(type(feature.grad))
        X_fgsm = Variable(torch.clamp(feature.data + epsilon *
                          feature.grad.data.sign(), 0.0, 1.0), requires_grad=True)
        feature = X_fgsm

        X_pgd001 = X_pgd001.to(device)
        print('*********')
        print(X_pgd001.device)
        print(feature.device)
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
            print('****************')
            print(feature.requires_grad)
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()

            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            arget_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)

            #road_adj, risk_adj, poi_adj = road_adj.to(device), risk_adj.to(device), poi_adj.to(device)

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
