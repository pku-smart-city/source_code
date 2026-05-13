# traj_dataset.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    lanes_count: int = 5
    lane_width: float = 4.0
    lane_y0: float = 0.0
    presence_th: float = 0.5


    history_s: float = 3.0      # T_P
    future_steps: Tuple[int, ...] = (1, 2, 3)  

    knn_k: int = 6
    max_dist_m: float = 50.0  


def get_lane_id(y: float, cfg: DataConfig) -> int:
    lid = int(np.round((y - cfg.lane_y0) / cfg.lane_width))
    return int(np.clip(lid, 0, cfg.lanes_count - 1))


def obs_to_node_features_ego_relative(obs: np.ndarray, cfg: DataConfig) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    ego = arr[0].copy()
    ego_x, ego_y, ego_vx, ego_vy = ego[1], ego[2], ego[3], ego[4]

    feats = arr.copy()
    # x,y,vx,vy -> dx,dy,dvx,dvy
    feats[:, 1] = feats[:, 1] - ego_x
    feats[:, 2] = feats[:, 2] - ego_y
    feats[:, 3] = feats[:, 3] - ego_vx
    feats[:, 4] = feats[:, 4] - ego_vy
    return feats  # [V, 5]


def build_adjacency_from_obs(obs: np.ndarray, cfg: DataConfig) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    V = arr.shape[0]
    pres = arr[:, 0] >= cfg.presence_th
    xy = arr[:, 1:3]  
    adj = np.zeros((V, V), dtype=np.float32)

    for i in range(V):
        if not pres[i]:
            continue
        d = np.linalg.norm(xy - xy[i], axis=1) 
        d[i] = 1e9
        cand = np.where((pres) & (d < cfg.max_dist_m))[0]
        if cand.size == 0:
            continue
        cand = cand[np.argsort(d[cand])]
        cand = cand[: cfg.knn_k]
        adj[i, cand] = 1.0

    adj = np.maximum(adj, adj.T)
    return adj


def pick_slot_indices(obs: np.ndarray, cfg: DataConfig) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    ego = arr[0]
    ego_x, ego_y = ego[1], ego[2]
    ego_lane = get_lane_id(float(ego_y), cfg)

    def lane_center(lid: int) -> float:
        return cfg.lane_y0 + lid * cfg.lane_width

    lane_veh: Dict[int, List[Tuple[int, float]]] = {lid: [] for lid in range(cfg.lanes_count)}
    for idx in range(1, arr.shape[0]):
        if arr[idx, 0] < cfg.presence_th:
            continue
        x, y = arr[idx, 1], arr[idx, 2]
        lid = get_lane_id(float(y), cfg)
        if abs(y - lane_center(lid)) <= cfg.lane_width * 0.5 + 1e-6:
            dx = x - ego_x
            lane_veh[lid].append((idx, abs(dx)))

    def nearest_in_lane(lid: int) -> int:
        if lid < 0 or lid >= cfg.lanes_count:
            return -1
        if len(lane_veh[lid]) == 0:
            return -1
        lane_veh[lid].sort(key=lambda t: t[1])
        return int(lane_veh[lid][0][0])

    left = nearest_in_lane(ego_lane - 1)
    keep = nearest_in_lane(ego_lane)
    right = nearest_in_lane(ego_lane + 1)
    return np.array([left, keep, right], dtype=np.int64)


def replace_invalid_slot(slot_idx: np.ndarray) -> np.ndarray:
    out = slot_idx.copy()
    out[out < 0] = 0
    return out


class TrajNPZDataset(Dataset):
    """
    from npz load:
    hist_feats:   [N, T, V, 5]
    adj:         [N, V, V]
    slot_idx:    [N, 3]
    target:      [N, 3, P, 2]
    slot_valid:  [N, 3]          (bool)
    future_valid:[N, 3, P]       (bool) 
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        required = ["hist_feats", "adj", "slot_idx", "target", "slot_valid", "future_valid"]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(
                f"[TrajNPZDataset] missing keys: {missing}. "
                f"available keys: {data.files}. "
            )

        self.hist_feats = data["hist_feats"].astype(np.float32)
        self.adj = data["adj"].astype(np.float32)
        self.slot_idx = data["slot_idx"].astype(np.int64)
        self.target = data["target"].astype(np.float32)
        self.slot_valid = data["slot_valid"].astype(np.bool_)
        self.future_valid = data["future_valid"].astype(np.bool_)

        self.ego_action = data["ego_action"].astype(np.int64) if "ego_action" in data.files else None
        self.meta = data["meta"] if "meta" in data.files else None

        N = self.hist_feats.shape[0]
        assert self.adj.shape[0] == N
        assert self.slot_idx.shape[0] == N
        assert self.target.shape[0] == N
        assert self.slot_valid.shape[0] == N
        assert self.future_valid.shape[0] == N

        assert self.target.shape[1] == 3 and self.target.shape[-1] == 2
        assert self.future_valid.shape[1] == 3
        assert self.future_valid.shape[2] == self.target.shape[2]

    def __len__(self):
        return self.hist_feats.shape[0]

    def __getitem__(self, idx):
        out = {
            "hist_feats": torch.from_numpy(self.hist_feats[idx]),
            "adj": torch.from_numpy(self.adj[idx]),
            "slot_idx": torch.from_numpy(self.slot_idx[idx]),
            "target": torch.from_numpy(self.target[idx]),
            "slot_valid": torch.from_numpy(self.slot_valid[idx]).bool(),
            "future_valid": torch.from_numpy(self.future_valid[idx]).bool(),  # [3,P]
        }
        if self.ego_action is not None:
            out["ego_action"] = torch.from_numpy(self.ego_action[idx])
        return out
