import torch
import numpy as np
import os
import math
from .models import RSSMWorldModelV3
from .pid_utils import PurePID

KMH_TO_MS = 1.0 / 3.6

class WorldModelAdapter:
    def __init__(self, model_path, stats_path, device='cuda', horizon=15):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = RSSMWorldModelV3(ego_dim=10, hidden_dim=512).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[WM] Weights loaded successfully from {model_path}")
            except Exception as e:
                print(f"[WM] Failed to load weights: {e}")
        self.model.eval()
        
        self._init_stats(stats_path)
        self.reset()
        self.horizon = horizon
        self.dt = 0.05
        self.lon_pid = PurePID(K_P=1.0, K_D=0.05, K_I=0.1, dt=self.dt)

    def _init_stats(self, stats_path):
        self.cols_order = ['vx','vy','vz','ax','ay','az','sin_yaw','cos_yaw','yaw_err','lat_dist'] 
        self.cols_order += ['wp5_x','wp5_y','wp10_x','wp10_y','wp20_x','wp20_y','wp50_x','wp50_y']
        for n in range(5): self.cols_order += [f'obj{n}_dx', f'obj{n}_dy', f'obj{n}_dvx', f'obj{n}_dvy', f'obj{n}_l', f'obj{n}_w']

        dim = len(self.cols_order)
        self.vec_mean = torch.zeros(dim, device=self.device)
        self.vec_std = torch.ones(dim, device=self.device)

        if not stats_path or not os.path.exists(stats_path):
            print(f"[WM] Stats file missing: {stats_path}. Using identity normalization fallback.")
            return

        raw_stats = torch.load(stats_path)
        for i, col in enumerate(self.cols_order):
            if col in raw_stats:
                self.vec_mean[i] = raw_stats[col]['mean']
                self.vec_std[i] = raw_stats[col]['std']

    def normalize(self, raw_vec_np):
        t = torch.from_numpy(raw_vec_np).float().to(self.device)
        return ((t - self.vec_mean) / (self.vec_std + 1e-6)).unsqueeze(0)
    
    def denormalize_ego(self, norm_ego):
        return (norm_ego * self.vec_std[0:10] + self.vec_mean[0:10]).detach().cpu().numpy()

    def _denormalize_road(self, norm_road):
        return (norm_road * self.vec_std[10:18] + self.vec_mean[10:18]).detach().cpu().numpy()

    def _normalize_road(self, road_phys_np):
        t = torch.from_numpy(road_phys_np).float().to(self.device)
        return ((t - self.vec_mean[10:18]) / (self.vec_std[10:18] + 1e-6)).unsqueeze(0)

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _update_road_features(self, road_loop_norm, ego_curr_norm, ego_next_norm):
        """
        Geometric road update during imagination rollout.
        The road waypoints are stored in ego-local coordinates. As ego moves/rotates,
        update the relative waypoint coordinates accordingly.
        """
        curr_e = self.denormalize_ego(ego_curr_norm)[0]
        next_e = self.denormalize_ego(ego_next_norm)[0]

        # ego local displacement in current frame (m)
        dx = 0.5 * (float(curr_e[0]) + float(next_e[0])) * self.dt
        dy = 0.5 * (float(curr_e[1]) + float(next_e[1])) * self.dt

        # heading increment from predicted sin/cos yaw
        yaw_curr = math.atan2(float(curr_e[6]), float(curr_e[7]))
        yaw_next = math.atan2(float(next_e[6]), float(next_e[7]))
        dyaw = self._wrap_to_pi(yaw_next - yaw_curr)

        road_phys = self._denormalize_road(road_loop_norm)[0].copy()
        c = math.cos(-dyaw)
        s = math.sin(-dyaw)

        for i in range(0, 8, 2):
            px = float(road_phys[i]) - dx
            py = float(road_phys[i + 1]) - dy
            road_phys[i] = px * c - py * s
            road_phys[i + 1] = px * s + py * c

        return self._normalize_road(road_phys)

    def reset(self):
        self.h = None
        self.z = None

    def tick_update(self, raw_obs, action_prev_phys):
        n = self.normalize(raw_obs)
        act = torch.tensor([action_prev_phys], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, _, self.h, self.z, _ = self.model(
                n[:, :10], n[:, 10:18], n[:, 18:48], act, self.h, self.z
            )

    def rollout(self, raw_obs, mada_actions):
        norm_obs = self.normalize(raw_obs)
        curr_ego_norm = norm_obs[:, 0:10]
        curr_road_norm = norm_obs[:, 10:18] 
        curr_social_norm = norm_obs[:, 18:48]
        
        phys_start = self.denormalize_ego(curr_ego_norm)[0]
        v_start = math.sqrt(phys_start[0]**2 + phys_start[1]**2)
        results = {}
        
        for action_name in mada_actions:
            h_loop = self.h.clone() if self.h is not None else None
            z_loop = self.z.clone() if self.z is not None else None
            e_loop_norm = curr_ego_norm
            r_loop_norm = curr_road_norm
            s_loop_norm = curr_social_norm
            
            traj_phys = []
            social_trajs_phys = []  # Store predicted trajectories of surrounding vehicles.
            self.lon_pid.reset()
            target_speed, base_steer = self._action_profile(action_name, v_start)

            with torch.no_grad():
                for t in range(self.horizon):
                    phys_e = self.denormalize_ego(e_loop_norm)[0]
                    v_curr = math.sqrt(phys_e[0]**2 + phys_e[1]**2)
                    
                    acc = self.lon_pid.step(target_speed - v_curr)
                    if acc >= 0: thr, brk = np.clip(acc, 0, 0.8), 0.0
                    else: thr, brk = 0.0, np.clip(-acc, 0, 1.0)
                    
                    steer = base_steer
                    is_lane_change_action = (
                        ("lane_changing" in action_name)
                        or action_name.startswith("lane_left")
                        or action_name.startswith("lane_right")
                    )
                    if not is_lane_change_action:
                        # phys_e[9] is unsigned lat_dist; use signed yaw_err for stable lane-centering.
                        steer += np.clip(-0.02 * phys_e[8], -0.2, 0.2)
                    
                    act_t = torch.tensor([[thr, np.clip(steer, -0.8, 0.8), brk]], dtype=torch.float32, device=self.device)
                    
                    next_e_norm, next_s_enc, _, h_loop, z_loop, _ = self.model(
                        e_loop_norm, r_loop_norm, s_loop_norm, act_t, h_loop, z_loop
                    )
                    
                    traj_phys.append(phys_e)
                    # Denormalize social vehicle states (first 20 dims: dx, dy, dvx, dvy).
                    social_phys = (next_s_enc * self.vec_std[18:38] + self.vec_mean[18:38]).detach().cpu().numpy()
                    social_trajs_phys.append(social_phys.reshape(5, 4))
                    
                    r_loop_norm = self._update_road_features(r_loop_norm, e_loop_norm, next_e_norm)
                    e_loop_norm = next_e_norm
                    s_reshaped = s_loop_norm.view(1, 5, 6)
                    s_loop_norm = torch.cat([next_s_enc.view(1, 5, 4), s_reshaped[:, :, 4:6]], dim=2).view(1, 30)

            results[action_name] = {
                'traj': np.array(traj_phys),
                'social_trajs': np.array(social_trajs_phys)  # Return imagined surrounding-vehicle positions.
            }
        return results

    def _action_profile(self, action_name, v_start):
        target_speed = v_start
        base_steer = 0.0

        if action_name in ("stop", "stop_base", "stop_hard"):
            target_speed = 0.0
        elif action_name == "stop_soft":
            target_speed = max(0.0, v_start - 2.0 * KMH_TO_MS)
        elif action_name in ("speed_up", "speed_up_base"):
            target_speed = v_start + 0.5 * KMH_TO_MS
        elif action_name == "speed_up_soft":
            target_speed = v_start + 0.25 * KMH_TO_MS
        elif action_name == "speed_up_hard":
            target_speed = v_start + 1.0 * KMH_TO_MS
        elif action_name in ("speed_down", "speed_down_base"):
            target_speed = max(0.0, v_start - 1.5 * KMH_TO_MS)
        elif action_name == "speed_down_soft":
            target_speed = max(0.0, v_start - 0.8 * KMH_TO_MS)
        elif action_name == "speed_down_hard":
            target_speed = max(0.0, v_start - 2.5 * KMH_TO_MS)
        elif action_name in ("maintain_speed", "maintain_base"):
            target_speed = v_start
        elif action_name == "maintain_minus":
            target_speed = max(0.0, v_start - 1.0 * KMH_TO_MS)
        elif action_name == "maintain_plus":
            target_speed = v_start + 1.0 * KMH_TO_MS
        elif action_name in ("lane_changing_left", "lane_left_base"):
            base_steer = -0.15
        elif action_name == "lane_left_gentle":
            base_steer = -0.10
            target_speed = max(0.0, v_start - 3.0 * KMH_TO_MS)
        elif action_name == "lane_left_fast":
            base_steer = -0.20
            target_speed = v_start + 3.0 * KMH_TO_MS
        elif action_name in ("lane_changing_right", "lane_right_base"):
            base_steer = 0.15
        elif action_name == "lane_right_gentle":
            base_steer = 0.10
            target_speed = max(0.0, v_start - 3.0 * KMH_TO_MS)
        elif action_name == "lane_right_fast":
            base_steer = 0.20
            target_speed = v_start + 3.0 * KMH_TO_MS

        return target_speed, base_steer
