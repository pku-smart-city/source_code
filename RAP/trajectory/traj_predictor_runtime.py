# traj_predictor_runtime.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import numpy as np
import torch

from traj_pred_model import SlotTrajPredictor, TrajModelConfig
from traj_dataset import (
    DataConfig,
    obs_to_node_features_ego_relative,
    build_adjacency_from_obs,
    pick_slot_indices,
    replace_invalid_slot,
    get_lane_id,
)

class DeepTrajPredictor:
    def __init__(
        self,
        ckpt_path: str,
        data_cfg: DataConfig,
        history_steps: int,
        future_steps: Tuple[int, ...] = (1,2,3),
        device: Optional[str] = None,
    ):
        self.data_cfg = data_cfg
        self.history_steps = history_steps
        self.future_steps = future_steps

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        payload = torch.load(ckpt_path, map_location=self.device)
        cfg_dict = payload["cfg"]
        cfg = TrajModelConfig(**cfg_dict)

        self.model = SlotTrajPredictor(cfg).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.obs_hist = deque(maxlen=history_steps)

    def reset(self):
        self.obs_hist.clear()

    @torch.no_grad()
    def predict(self, obs: np.ndarray, dt_s: float, horizon_s: float = 3.0) -> Optional[Dict[str, Any]]:
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        self.obs_hist.append(obs)
        if len(self.obs_hist) < self.history_steps:
            return None

        # hist_feats: [T,V,5] (ego-relative)
        hist_feats = np.stack(
            [obs_to_node_features_ego_relative(o, self.data_cfg) for o in list(self.obs_hist)],
            axis=0
        )

        adj = build_adjacency_from_obs(obs, self.data_cfg)

        slot_raw = pick_slot_indices(obs, self.data_cfg)      # [-1 or idx]
        slot_valid = (slot_raw >= 0)
        slot_fixed = replace_invalid_slot(slot_raw)

        hist_t = torch.from_numpy(hist_feats[None, ...]).to(self.device)   # [1,T,V,5]
        adj_t = torch.from_numpy(adj[None, ...]).to(self.device)           # [1,V,V]
        slot_t = torch.from_numpy(slot_fixed[None, ...]).to(self.device)   # [1,3]
        pred = self.model(hist_t, adj_t, slot_t)[0].cpu().numpy()          # [3,P,2]

        ego_x, ego_y = float(obs[0, 1]), float(obs[0, 2])
        ego_lane = get_lane_id(ego_y, self.data_cfg)

        def lane_from_dy(dy: float) -> int:
            y_abs = ego_y + float(dy)
            return get_lane_id(y_abs, self.data_cfg)

        lanes = {}
        slot_names = ["left", "keep", "right"]
        for s, name in enumerate(slot_names):
            vidx = int(slot_raw[s])
            if vidx < 0 or (not slot_valid[s]):
                lanes[name] = None
                continue

            dx_now = float(obs[vidx, 1] - obs[0, 1])
            dy_now = float(obs[vidx, 2] - obs[0, 2])
            dvx_now = float(obs[vidx, 3] - obs[0, 3])
            vy = float(obs[vidx, 4])

            points = [{"t": 0.0, "dx": dx_now, "dy": dy_now, "lane": lane_from_dy(dy_now)}]
            for j, k in enumerate(self.future_steps):
                t = float(k) * float(dt_s)
                dx, dy = float(pred[s, j, 0]), float(pred[s, j, 1])
                points.append({"t": t, "dx": dx, "dy": dy, "lane": lane_from_dy(dy)})

            lanes[name] = {
                "obs_idx": int(vidx),
                "lane": int(lane_from_dy(dy_now)),
                "dx_now": dx_now,
                "dvx_now": dvx_now,
                "vy": vy,
                "y_offset_to_center": 0.0,  
                "modes": [
                    {
                        "name": "nn_pred",
                        "p": 1.0,
                        "points": points
                    }
                ]
            }

        return {
            "ego_lane": int(ego_lane),
            "horizon_s": float(horizon_s),
            "dt_s": float(dt_s),
            "lanes": lanes
        }
