import os

import dgl
import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GraphConv

# import keras.backend.tensorflow_backend as KTF
from keras.backend import set_session
from keras.callbacks import Callback
from keras.layers import LSTM, Dense
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential, load_model

# 设定为自增长
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)


def create_dataset(
    data, traintime_num, pertime_num, train_proportion, vehicle_count, times
):
    '''
    对数据进行处理
    首先是对于多个预测维度，全部划分为3预测1的形式
    其次多次获取到的数据分开
    '''
    dim = data.shape[0]  # time_count * times * vehicle_count
    train_X, train_Y = [], []
    j = 0
    for time in range(int(times * train_proportion)):
        if time > 0:
            j = j + traintime_num + pertime_num
        temp_X1, temp_Y1 = [], []
        while (
            dim / times * time
            <= (traintime_num + j + pertime_num) * vehicle_count
            < dim / times * (time + 1)
        ):
            vehicle_X1 = data[
                j * vehicle_count : (traintime_num + j) * vehicle_count :, 1:3
            ]
            temp_X1.append(vehicle_X1)
            vehicle_Y1 = data[
                (traintime_num + j)
                * vehicle_count : (traintime_num + j + pertime_num)
                * vehicle_count :,
                1:3,
            ]
            temp_Y1.append(vehicle_Y1)
            j += 1
        train_X.append(temp_X1)
        train_Y.append(temp_Y1)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    test_X, test_Y = [], []
    j = 0
    for time in range(int(times * train_proportion)):
        if time > 0:
            j = j + traintime_num + pertime_num
        temp_X2, temp_Y2 = [], []
        while (
            dim / times * time
            <= (traintime_num + j + pertime_num) * vehicle_count
            < dim / times * (time + 1)
        ):
            vehicle_X2 = data[
                j * vehicle_count : (traintime_num + j) * vehicle_count :, 1:3
            ]
            temp_X2.append(vehicle_X2)
            vehicle_Y2 = data[
                (traintime_num + j)
                * vehicle_count : (traintime_num + j + pertime_num)
                * vehicle_count :,
                1:3,
            ]
            temp_Y2.append(vehicle_Y2)
            j += 1
        test_X.append(temp_X2)
        test_Y.append(temp_Y2)

    return train_X, train_Y, test_X, test_Y


def NormalizeMult(data, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        if set_range == True:
            list = data[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])
        else:
            if i == 0:
                listlow = -90
                listhigh = 90
            else:
                listlow = -180
                listhigh = 180

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data, normalize


class tra_pre(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, input_size, hidden_size, output_size
    ):
        super(tra_pre, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.fc_gcn = nn.Linear(output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_gru = nn.Linear(hidden_size, output_size)

    def forward(self, points_list):
        output = []
        for point_list in points_list:
            points = torch.tensor(point_list, dtype=torch.float)
            gcn_out = []
            for one_time_points in points:
                distances = torch.norm(
                    one_time_points[:, None] - one_time_points, dim=2
                )  # 计算点之间的欧氏距离
                threshold = 0.3  # 距离阈值
                adj_matrix = (distances < threshold).float()  # 基于距离阈值构建邻接矩阵
                rowsum = torch.sum(adj_matrix, dim=1)
                D = torch.diag(1.0 / torch.sqrt(rowsum))
                normalized_adj_matrix = torch.matmul(torch.matmul(D, adj_matrix), D)
                node_features = one_time_points.clone().detach()
                # 前向传播
                normalized_adj_matrix = normalized_adj_matrix.clone().detach()
                node_features = node_features.clone().detach()
                normalized_adj_matrix = normalized_adj_matrix.float()
                x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
                x = self.relu(x)
                x = self.conv2(torch.matmul(normalized_adj_matrix, x))
                x = self.relu(x)
                x = self.fc_gcn(x.mean(dim=0, keepdim=True))
                gcn_out.append(x)
            gcn_out = torch.stack(gcn_out)
            gcn_out = gcn_out.reshape(
                gcn_out.shape[1], gcn_out.shape[0], gcn_out.shape[2]
            )
            gru_out, _ = self.gru(gcn_out)
            gru_out = self.fc_gru(gru_out)
            gru_out = gru_out.mean(dim=1, keepdim=True)
            output.append(gru_out)
        return output


if __name__ == "__main__":
    traintime_num = 3
    pertime_num = 1
    # set_range = False
    set_range = True
    train_proportion = 1  # the proportion of the vehicle to train
    times = 400
    during = 40
    agg_rate = 100
    vehicle_type = "IDM"

    # 读入时间序列的文件数据
    # npy = np.load('../npy/agg_'+ str(agg_rate) +'%_'+ str(during) +'s_'+ str(times) +'times.npy')
    npy = np.load('../npy/' + str(vehicle_type) + '.npy')
    # print(agg)
    # print(npy)

    # 转DataFrame
    data = np.array(npy)
    time_count = int(data.shape[0] / times)
    vehicle_count = data.shape[1]
    data = np.resize(data, (time_count * times * vehicle_count, 7))
    print(
        "时间维度：{0}，车辆数量：{1}, 运行次数：{2}".format(
            time_count, vehicle_count, times
        )
    )

    # 生成训练数据
    # print(data)
    train_X, train_Y, test_X, test_Y = create_dataset(
        data, traintime_num, pertime_num, train_proportion, vehicle_count, times
    )
    train_X = train_X.reshape(
        train_X.shape[0] * train_X.shape[1],
        traintime_num,
        vehicle_count,
        train_X.shape[3],
    )
    train_Y = train_Y.reshape(
        train_Y.shape[0] * train_Y.shape[1], train_Y.shape[2], train_Y.shape[3]
    )
    # train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[3])
    # test_X = test_X.reshape(test_X.shape[0], test_X.shape[2], test_X.shape[3])
    # test_Y = test_Y.reshape(test_Y.shape[0], test_Y.shape[2], test_Y.shape[3])
    # print(type(train_X))
    print("x:", train_X.shape)
    print("y:", train_Y.shape)

    model = tra_pre(
        input_dim=2,
        hidden_dim=256,
        output_dim=256,
        input_size=256,
        hidden_size=256,
        output_size=10,
    )
    # 迭代训练
    num_epochs = 10
    batch_size = 32
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        # 在每个训练迭代中，按批次处理数据
        for i in range(0, train_X.shape[0], batch_size):
            inputs = train_X[i : i + batch_size]
            targets = torch.tensor(train_Y[i : i + batch_size])
            # print(targets.shape)
            # 将输入数据和目标数据传递给模型
            outputs = model(inputs)
            outputs = torch.stack(outputs)
            # print(outputs.shape)
            outputs = outputs.reshape(
                outputs.shape[0], vehicle_count, int(outputs.shape[3] / vehicle_count)
            )
            # print(outputs)

            # 计算损失
            loss = criterion(outputs.float(), targets.float())
            # print(loss)
            # 清除之前的梯度
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
        # 打印当前迭代的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    path = "./model/" + vehicle_type + "_model.pt"
    torch.save(model, path)
