import torch
import torch.nn as nn
from scipy.signal import stft
import numpy as np


class STFTMLP(nn.Module):
    def __init__(self, T, fs=1.0, nperseg=64, noverlap=32, hidden_size=128):
        super(STFTMLP, self).__init__()
        self.T = T
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

        # 计算STFT后的频率和时间数
        _, _, Zxx_example = stft(np.zeros(T), fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        n_frequencies, n_times = Zxx_example.shape

        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_frequencies * n_times, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, T)
        )

        # He初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        N = x1.shape[0]
        amplitudes_diff = []

        for i in range(N):
            # 对每个样本应用STFT
            f1, t_stft1, Zxx1 = stft(x1[i], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            f2, t_stft2, Zxx2 = stft(x2[i], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)

            # 提取振幅
            amplitude1 = np.abs(Zxx1)
            amplitude2 = np.abs(Zxx2)

            # 计算振幅差值
            diff = amplitude1 - amplitude2
            amplitudes_diff.append(diff)

        # 将差值转换为张量
        amplitudes_diff = np.array(amplitudes_diff)
        amplitudes_diff = torch.tensor(amplitudes_diff, dtype=torch.float32)

        # 通过MLP
        output = self.mlp(amplitudes_diff)  # [N,T]

        output = torch.mean(output, dim=-1)  # [N,]

        return output