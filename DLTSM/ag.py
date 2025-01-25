import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import Helper as helper
from Imputer import LRTC_TSpN


def get_missing_rate(tensor):
    return np.sum(tensor == 0) / np.prod(tensor.shape)


def generate_fiber_missing(tensor3d_true, lost_rate, mode: int):
    """
    根据模式生成丢失数据。

    Parameters:
    tensor3d_true: np.array - 输入的真实张量 (intervals * links * days)
    lost_rate: float - 丢失数据的比例 (0 到 1)
    mode: int - 丢失模式：
        - mode 0: links * days 组合
        - mode 1: intervals * days 组合
        - mode 2: intervals * links 组合

    Returns:
    tensor3d_lost_fiber: np.array - 具有丢失数据的张量
    """
    n = tensor3d_true.shape
    nn = np.delete(n, mode)
    S = np.ones(nn)

    coord = []
    for i in range(nn[0]):
        for j in range(nn[1]):
            coord.append((i, j))

    # 随机选择丢失的坐标
    mask = random.sample(coord, int(lost_rate * len(coord)))
    for coord in mask:
        S[coord[0], coord[1]] = 0

    # 扩展并应用遮罩
    fai = np.expand_dims(S, mode).repeat(n[mode], axis=mode)

    # 生成具有丢失数据的张量
    tensor3d_lost_fiber = fai * tensor3d_true
    tensor_miss_rate = get_missing_rate(tensor3d_lost_fiber)
    print(f'fiber-mode{mode} missing rate of tensor is：{100 * tensor_miss_rate:.2f}%')

    return tensor3d_lost_fiber


def generate_augmented_data(X_T_prime, k=10, lost_rate=0.3, mode=0):
    """
    使用随机遮罩生成增广数据。

    Parameters:
    X_T_prime: np.array - 初始插值数据 (intervals * links * days)
    k: int - 生成的增广数据数量
    lost_rate: float - 丢失率
    mode: int - 丢失模式（参考 `generate_fiber_missing` 的模式参数）

    Returns:
    bar_X_T: np.array - 增广后的数据，形状为 (intervals * links * days * k)
    """
    augmented_data = []

    # 生成 k 个增广数据
    for i in range(k):
        # 使用 `generate_fiber_missing` 函数来生成丢失数据
        X_m_i = generate_fiber_missing(X_T_prime, lost_rate, mode)

        # 对每个丢失数据进行LRTC-TSpN插值
        X_T_i = LRTC_TSpN(X_m_i)

        # 将每次生成的数据添加到增广数据列表中
        augmented_data.append(X_T_i)

    # 将增广后的数据合并为一个多维数组 (intervals * links * days * k)
    bar_X_T = np.stack(augmented_data, axis=-1)

    return bar_X_T


# 生成空间增广数据
augmented_data = generate_augmented_data(X_T_prime, k=5, lost_rate=0.3, mode=0)

print("Shape of augmented data:", augmented_data.shape)