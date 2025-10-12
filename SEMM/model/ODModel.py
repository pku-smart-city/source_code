import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ODModel(nn.Module):
    def __init__(self, N, temp, freq,dropout_rate=0.0):
        super(ODModel, self).__init__()

        self.N = N
        self.temp = temp
        self.freq = freq

        n1 = 128
        n2 = 64

        self.weights = nn.Parameter(torch.randn(N, 3))

        # MLP for each parameter: n, p, pi
        self.mlp_n = nn.Sequential(
            nn.Linear(N, n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n2, N * N)
        )

        self.mlp_p = nn.Sequential(
            nn.Linear(N, n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n2, N * N)
        )

        self.mlp_pi = nn.Sequential(
            nn.Linear(N, n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n2, N * N)
        )

    def forward(self, x):
        '''
        :param x: 输入速度 [B, N]
        :return: 负二项分布参数 n, p, pi  --> [B, N, N]
        '''
        B, N = x.shape

        x_temp = np.tile(self.temp, (B, 1))  # [B,N]
        x_freq = np.tile(self.freq, (B, 1))  # [B,N]

        x_random = np.random.normal(loc=0.05, scale=0.01, size=(B, N)).astype(float)
        x_random = np.clip(x_random, 0, 0.1)

        x_temp_freq_rand = np.stack([x_temp, x_freq, x_random], axis=-1)  # [B,N,3]
        tensor_delta_x = torch.tensor(x_temp_freq_rand, dtype=torch.float).to(x.device)

        self.weights = self.weights.to(x.device)
        weighted_sum = x + torch.sum(tensor_delta_x * self.weights.unsqueeze(0), dim=2)  # [B, N]

        # 三个分支 MLP 分别生成 n, p, pi 参数
        n_flat = self.mlp_n(weighted_sum)     # [B, N*N]
        p_flat = self.mlp_p(weighted_sum)     # [B, N*N]
        pi_flat = self.mlp_pi(weighted_sum)   # [B, N*N]

        # reshape 到 [B, N, N]
        n = F.softplus(n_flat.view(B, N, N))  # 保证 n > 0
        p = torch.sigmoid(p_flat.view(B, N, N))  # 保证 p ∈ (0,1)
        pi = torch.sigmoid(pi_flat.view(B, N, N))  # 保证 pi ∈ (0,1)

        return n, p, pi