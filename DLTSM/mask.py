import torch
from torch.autograd import Variable
import torch.utils.data as utils
import numpy as np
import copy
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random as random


def get_missing_rate(X_lost):
    o_channel_num = (X_lost == 0).astype(int).sum().sum()
    matrix_miss_rate = o_channel_num / (X_lost.size)

    return matrix_miss_rate


def generate_fiber_missing(tensor3d_true, lost_rate, mode: int):
    # three kinds of fiber-like missing cases, the original tensor structure is intervals*links*days.
    # mode0:links*days combination
    # mode1:intervals*days combination
    # mode2:intervals*links combination
    n = tensor3d_true.shape
    nn = np.delete(n, mode)
    S = np.ones(nn)
    coord = []
    for i in range(nn[0]):
        for j in range(nn[1]):
            coord.append((i, j))
    mask = random.sample(coord, int(lost_rate * len(coord)))
    for coord in mask:
        S[coord[0], coord[1]] = 0
    fai = np.expand_dims(S, mode).repeat(n[mode], axis=mode)
    tensor3d_lost_fiber = fai * tensor3d_true
    tensor_miss_rate = get_missing_rate(tensor3d_lost_fiber)
    print(f'fiber-mode{mode} missing rate of tensor is：{100 * tensor_miss_rate:.2f}%')

    return tensor3d_lost_fiber


def generate_tensor_random_missing(tensor3d_true, lost_rate):
    tensor3d_lost = tensor3d_true.copy()
    coord = []
    m, n, q = tensor3d_lost.shape
    for i in range(m):
        for j in range(n):
            for k in range(q):
                coord.append((i, j, k))

    mask = random.sample(coord, int(lost_rate * len(coord)))
    for coord in mask:
        tensor3d_lost[coord[0]][coord[1]][coord[2]] = 0
    return tensor3d_lost




