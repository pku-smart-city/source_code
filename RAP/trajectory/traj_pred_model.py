# traj_pred_model.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    """
    H = D^{-1/2} (A+I) D^{-1/2} X W
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, V, Fin]
        adj:[B, V, V]  
        """
        b, v, _ = x.shape
        device = x.device

        # A_hat = A + I
        I = torch.eye(v, device=device).unsqueeze(0).expand(b, v, v)
        A_hat = adj + I

        deg = A_hat.sum(dim=-1).clamp(min=1.0)  # [B, V]
        deg_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, V, V]

        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # [B, V, V]

        h = A_norm @ x  # [B, V, Fin]
        h = self.lin(h)
        return h


@dataclass
class TrajModelConfig:
    feat_dim: int = 5         
    gcn_hidden: int = 256
    gru_hidden: int = 256
    num_gcn_layers: int = 2
    num_pred_steps: int = 3    
    num_slots: int = 3         # LEFT / KEEP / RIGHT
    dropout: float = 0.0


class SlotTrajPredictor(nn.Module):
    def __init__(self, cfg: TrajModelConfig):
        super().__init__()
        self.cfg = cfg

        self.gcn1 = SimpleGCNLayer(cfg.feat_dim, cfg.gcn_hidden)
        self.gcn2 = SimpleGCNLayer(cfg.gcn_hidden, cfg.gcn_hidden)

        self.fc_pre_gru = nn.Linear(cfg.gcn_hidden, cfg.gcn_hidden)

        self.gru = nn.GRU(
            input_size=cfg.gcn_hidden,
            hidden_size=cfg.gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc_out = nn.Linear(cfg.gru_hidden, cfg.num_pred_steps * 2)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        hist_feats: torch.Tensor,
        adj: torch.Tensor,
        slot_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        hist_feats: [B, T, V, F]
        adj:        [B, V, V]   （
        slot_idx:   [B, K]     

        return:
            pred: [B, K, P, 2]
        """
        B, T, V, Fdim = hist_feats.shape
        K = slot_idx.shape[1]
        P = self.cfg.num_pred_steps

        x = hist_feats.reshape(B * T, V, Fdim)          # [B*T, V, F]
        adj_bt = adj.unsqueeze(1).expand(B, T, V, V).reshape(B * T, V, V)

        h = F.relu(self.gcn1(x, adj_bt))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, adj_bt))
        h = self.dropout(h)

        # FC
        h = F.relu(self.fc_pre_gru(h))                  # [B*T, V, H]
        h = h.reshape(B, T, V, self.cfg.gcn_hidden)     # [B, T, V, H]

        # slot_idx: [B, K] -> [B, 1, K, 1] -> broadcast to [B, T, K, H]
        idx = slot_idx[:, None, :, None].expand(B, T, K, self.cfg.gcn_hidden)
        slot_seq = torch.gather(h, dim=2, index=idx)    # [B, T, K, H]
        slot_seq = slot_seq.permute(0, 2, 1, 3).contiguous()  # [B, K, T, H]

        slot_seq = slot_seq.reshape(B * K, T, self.cfg.gcn_hidden)  # [B*K, T, H]
        out, _ = self.gru(slot_seq)                                 # [B*K, T, gru_hidden]
        last = out[:, -1, :]                                        # hidden: [B*K, gru_hidden]

        y = self.fc_out(last)                                       # [B*K, P*2]
        y = y.reshape(B, K, P, 2)                                   # [B, K, P, 2]
        return y
