import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from utils.STFTMLP import STFTMLP


def load_data(od_path, speed_path, freq_shuffle=True, freq_method=None, seed=42):
    print('************************* Loading data ************************')

    set_seed(seed)
    speed = np.load(speed_path)
    od = np.load(od_path)

    # 获取数据长度 T
    T, N = speed.shape

    # 按顺序划分数据
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]


    # 在训练集上计算 OD 出发总量
    od_train_departures = np.sum(od_train, axis=-1)  # 形状 [T_train, N]

    # 计算每个区域在 T_train 时间步的平均速度和平均出发总量
    mean_speed = np.mean(speed_train, axis=0)  # 每个区域的平均速度 [N,]
    mean_departures = np.mean(od_train_departures, axis=0)  # 每个区域的平均出发总量 [N,]

    # 计算平均速度和平均出发总量的差值
    temporal = (mean_departures - mean_speed).astype(float)  # [N,]

    # 频域差距求解
    seq1 = torch.tensor(speed_train, dtype=torch.float32).transpose(0, 1)  # [N,T]
    seq2 = torch.tensor(od_train_departures, dtype=torch.float32).transpose(0, 1) # [N,T]

    if freq_shuffle:
        # 生成打乱索引
        perm = torch.randperm(seq1.size(1))
        # 按照相同的索引打乱两个序列
        seq1 = seq1[:, perm]  # [N, T]
        seq2 = seq2[:, perm]  # [N, T]

    if freq_method == 'STFT':
        stftmlp = STFTMLP(train_size)
        freq = stftmlp(seq1, seq2)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    val_data = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    test_data = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("Final train input:", train_data.shape, "train target:", train_target.shape)
    print("Final val input:", val_data.shape, "val target:", val_target.shape)
    print("Final test input:", test_data.shape, "test target:", test_target.shape)

    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    scaler_x_temp = MinMaxScaler()
    x_temp = scaler_x_temp.fit_transform(temporal.reshape(-1, 1))

    scaler_x_freq = MinMaxScaler()
    x_freq = scaler_x_freq.fit_transform(freq.detach().numpy().reshape(-1, 1))

    x_temp = np.squeeze(x_temp, axis=-1)
    x_freq = np.squeeze(x_freq, axis=-1)

    return train_loader, val_loader, test_loader, x_temp, x_freq, scaler

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子