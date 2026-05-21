import os
import threading
import time
import json
import numpy as np

from agents.navigation.behavior_agent import BehaviorAgent


class BehaviorAgentWM(BehaviorAgent):
    """
    Baseline agent + WM-only sub-action refinement.
    LLM outputs Action only; WM picks the safest SubAction within that Action group.
    """

    def __init__(
        self,
        vehicle,
        behavior='normal',
        model_path=None,
        stats_path=None,
        exp_path='logs',
        openai_api_key=None,
        openai_api_base=None,
    ):
        super().__init__(
            vehicle,
            behavior=behavior,
            exp_path=exp_path,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
        )

        from .wm_core.wm_adapter import WorldModelAdapter
        from .wm_core.features import FeatureExtractor

        wm_core_dir = os.path.join(os.path.dirname(__file__), "wm_core")
        if model_path is None:
            model_path = os.path.join(wm_core_dir, "rssm_v3_best.pth")
        if stats_path is None:
            stats_path = os.path.join(wm_core_dir, "meta_stats.pth")

        self.wm = WorldModelAdapter(model_path, stats_path)
        self.feature_extractor = FeatureExtractor(vehicle, self._world)
        self.wm_lock = threading.Lock()
        self.current_raw_features = None
        self.last_phys_action = [0.0, 0.0, 0.0]

        self.refined_action_groups = {
            "normal_behavior": ["maintain_base", "speed_down_base", "speed_down_hard", "stop_base"],
            "speed_up": ["speed_up_soft", "speed_up_base", "speed_up_hard"],
            "speed_down": ["speed_down_soft", "speed_down_base", "speed_down_hard"],
            "maintain_speed": ["maintain_minus", "maintain_base", "maintain_plus"],
            "lane_changing_left": ["lane_left_gentle", "lane_left_base", "lane_left_fast"],
            "lane_changing_right": ["lane_right_gentle", "lane_right_base", "lane_right_fast"],
            "stop": ["stop_soft", "stop_base", "stop_hard"],
        }
        self.enable_refined_subaction = True
        self._cached_subaction = None

    def get_refined_actions(self, base_action):
        return self.refined_action_groups.get(base_action, [base_action])

    def _safety_score(self, pred, action_name=None):
        traj = pred.get("traj")
        social = pred.get("social_trajs")
        if traj is None or social is None or len(traj) == 0 or len(social) == 0:
            return 1e9

        min_dist = 1e9
        for t in range(social.shape[0]):
            for n in range(min(5, social.shape[1])):
                dx, dy = float(social[t, n, 0]), float(social[t, n, 1])
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist > 0.5:
                    min_dist = min(min_dist, dist)

        if min_dist == 1e9:
            min_dist = 60.0

        # Safety-first score: prioritize clearance and lane-boundary excursions.
        critical_penalty = 100.0 if min_dist < 3.0 else 0.0
        dist_penalty = max(0.0, 8.0 - min_dist) ** 2
        max_lat = float(np.max(np.abs(traj[:, 9])))

        # During lane-change actions, lateral offset increase is expected.
        # Use a looser threshold and lower weight to avoid over-penalizing valid lane changes.
        is_lane_change_action = bool(action_name) and (
            ("lane_changing" in action_name)
            or action_name.startswith("lane_left")
            or action_name.startswith("lane_right")
        )
        if is_lane_change_action:
            lane_penalty = max(0.0, max_lat - 2.8) * 2.0
        else:
            lane_penalty = max(0.0, max_lat - 1.5) * 8.0
        return critical_penalty + dist_penalty + lane_penalty

    def _estimate_min_social_distance(self, raw_obs):
        # social starts at index 18, each object has [dx, dy, dvx, dvy, l, w]
        min_dist = 1e9
        for i in range(5):
            base = 18 + i * 6
            dx = float(raw_obs[base])
            dy = float(raw_obs[base + 1])
            d = float(np.sqrt(dx * dx + dy * dy))
            if d > 0.5:
                min_dist = min(min_dist, d)
        return min_dist if min_dist < 1e9 else 100.0

    def _estimate_front_collision_risk(self, raw_obs):
        """
        Estimate front-risk using relative position and relative speed.
        This avoids hard-stop deadlocks when both cars are already stopped
        at short distance (low closing risk).
        """
        ego_half_len = 2.4
        front_lane_half_width = 2.5

        min_gap = 1e9
        min_ttc = 1e9
        max_closing = 0.0
        has_front = False

        for i in range(5):
            base = 18 + i * 6
            dx = float(raw_obs[base])
            dy = float(raw_obs[base + 1])
            dvx = float(raw_obs[base + 2])  # target_vx - ego_vx in ego frame
            obj_l = float(raw_obs[base + 4])

            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist <= 0.5:
                continue
            if dx <= -1.0:
                continue
            if abs(dy) > front_lane_half_width:
                continue

            has_front = True
            gap = max(0.0, dx - 0.5 * obj_l - ego_half_len)
            closing = max(0.0, -dvx)
            ttc = (gap / closing) if closing > 1e-3 else 1e9

            if gap < min_gap:
                min_gap = gap
            if ttc < min_ttc:
                min_ttc = ttc
            if closing > max_closing:
                max_closing = closing

        return {
            "has_front": has_front,
            "min_gap": min_gap if has_front else 1e9,
            "min_ttc": min_ttc if has_front else 1e9,
            "max_closing": max_closing if has_front else 0.0,
        }

    def _select_subaction(self, base_action, raw_obs):
        candidates = self.get_refined_actions(base_action)
        if len(candidates) == 1:
            return candidates[0]

        with self.wm_lock:
            preds = self.wm.rollout(raw_obs, candidates)
        if not preds:
            return candidates[1] if len(candidates) >= 2 else candidates[0]

        best = None
        best_score = 1e9
        for name in candidates:
            pred = preds.get(name)
            if pred is None:
                continue
            s = self._safety_score(pred, action_name=name)
            if s < best_score:
                best_score = s
                best = name

        if best is None:
            return candidates[1] if len(candidates) >= 2 else candidates[0]
        return best

    def _update_wm_prompt_hint(self):
        """
        Export compact WM risk hints for downstream TM logic.
        Stored as JSON string in self.extra_prompt_hint.
        """
        raw_obs = self.current_raw_features
        if raw_obs is None or len(raw_obs) < 10:
            self.extra_prompt_hint = ""
            return

        front_risk = self._estimate_front_collision_risk(raw_obs)
        has_front = bool(front_risk.get("has_front", False))
        front_gap = float(front_risk.get("min_gap", 1e9))
        front_ttc = float(front_risk.get("min_ttc", 1e9))

        if (not has_front) or (not np.isfinite(front_gap)) or front_gap >= 1e8:
            front_gap = -1.0
        if (not has_front) or (not np.isfinite(front_ttc)) or front_ttc >= 1e8:
            front_ttc = -1.0

        # raw_obs[9] is absolute lateral distance to lane center (meters).
        lat_dist_m = float(abs(raw_obs[9]))
        lane_half_width_m = 1.75  # fallback for typical 3.5m lane
        try:
            ego_wp = self._map.get_waypoint(
                self._vehicle.get_location(), project_to_road=True
            )
            if ego_wp is not None and float(ego_wp.lane_width) > 0.1:
                lane_half_width_m = max(1.0, 0.5 * float(ego_wp.lane_width))
        except Exception:
            pass
        lane_offset_norm = lat_dist_m / max(1e-3, lane_half_width_m)

        static_ahead_m = -1.0
        try:
            hit = self._static_obstacle_ahead(max_distance=30.0, fov_deg=45.0, lateral_limit=3.0)
            if isinstance(hit, dict):
                d = float(hit.get("distance", -1.0))
                if np.isfinite(d) and d >= 0.0:
                    static_ahead_m = d
        except Exception:
            pass

        hint = {
            "wm_front_has": has_front,
            "wm_front_gap_m": round(front_gap, 2) if front_gap >= 0.0 else -1.0,
            "wm_front_ttc_s": round(front_ttc, 2) if front_ttc >= 0.0 else -1.0,
            "wm_lane_offset_norm": round(max(0.0, lane_offset_norm), 3),
            "wm_static_ahead_m": round(static_ahead_m, 2) if static_ahead_m >= 0.0 else -1.0,
        }
        self.extra_prompt_hint = json.dumps(hint, ensure_ascii=False)

    def _execute_refined_action(self, action_name, debug=False):
        if action_name == "speed_up_soft":
            return self.speed_up(increment=0.25, debug=debug)
        if action_name == "speed_up_base":
            return self.speed_up(increment=0.5, debug=debug)
        if action_name == "speed_up_hard":
            return self.speed_up(increment=1.0, debug=debug)

        if action_name == "speed_down_soft":
            return self.speed_down(decrement=0.8, debug=debug)
        if action_name == "speed_down_base":
            return self.speed_down(decrement=1.5, debug=debug)
        if action_name == "speed_down_hard":
            return self.speed_down(decrement=2.5, debug=debug)

        if action_name == "maintain_minus":
            self._local_planner.set_speed(max(0.0, self.last_speed - 1.0))
            return self._local_planner.run_step(debug=debug)
        if action_name == "maintain_base":
            return self.maintain_speed(debug=debug)
        if action_name == "maintain_plus":
            self._local_planner.set_speed(self.last_speed + 1.0)
            return self._local_planner.run_step(debug=debug)

        if action_name == "lane_left_gentle":
            self._local_planner.set_speed(max(10.0, self._speed - 3.0))
            return self.lane_changing_left(debug=debug)
        if action_name == "lane_left_base":
            return self.lane_changing_left(debug=debug)
        if action_name == "lane_left_fast":
            self._local_planner.set_speed(min(45.0, self._speed + 3.0))
            return self.lane_changing_left(debug=debug)

        if action_name == "lane_right_gentle":
            self._local_planner.set_speed(max(10.0, self._speed - 3.0))
            return self.lane_changing_right(debug=debug)
        if action_name == "lane_right_base":
            return self.lane_changing_right(debug=debug)
        if action_name == "lane_right_fast":
            self._local_planner.set_speed(min(45.0, self._speed + 3.0))
            return self.lane_changing_right(debug=debug)

        if action_name == "stop_soft":
            control = self.stop()
            control.brake = min(self._max_brake, 0.4)
            return control
        if action_name == "stop_base":
            return self.stop()
        if action_name == "stop_hard":
            return self.stop()

        return None

    def translate_gpt_command(self, response, debug=False):
        if not (isinstance(response, dict) and 'Action' in response):
            return self.normal_behavior(debug=debug)

        sub_action = response.get("SubAction")
        if self.enable_refined_subaction and sub_action:
            control = self._execute_refined_action(sub_action, debug=debug)
            if control is not None:
                return control
        return super().translate_gpt_command(response)

    def run_step(self, debug=False):
        self._update_information()

        self.current_raw_features = self.feature_extractor.get_features_vector()
        with self.wm_lock:
            self.wm.tick_update(self.current_raw_features, self.last_phys_action)
        self._update_wm_prompt_hint()

        if not self.gpt3_thread.is_alive():
            self.start_gpt3_thread()
        if not self.log_thread.is_alive():
            self.start_log_thread()
        while self.command_dict is None:
            time.sleep(0.1)

        cmd = self.command_dict
        if self.enable_refined_subaction and isinstance(cmd, dict) and "Action" in cmd:
            base_action = cmd["Action"]
            if base_action in self.refined_action_groups:
                # Hard safety guard: front-risk based (gap + closing speed + TTC),
                # instead of all-direction distance thresholding.
                front_risk = self._estimate_front_collision_risk(self.current_raw_features)
                forced_subaction = None

                if front_risk["has_front"]:
                    gap = front_risk["min_gap"]
                    ttc = front_risk["min_ttc"]
                    closing = front_risk["max_closing"]

                    # Immediate very-close protection.
                    if gap < 1.5:
                        forced_subaction = "stop_base"
                    # Aggressive braking only when there is real closing risk.
                    elif gap < 3.0 and closing > 0.5:
                        forced_subaction = "stop_base"
                    elif gap < 6.0 and ttc < 2.0 and base_action in (
                        "normal_behavior",
                        "speed_up",
                        "maintain_speed",
                        "lane_changing_left",
                        "lane_changing_right",
                    ):
                        forced_subaction = "speed_down_hard"

                if forced_subaction is None and base_action in (
                    "normal_behavior",
                    "speed_up",
                    "maintain_speed",
                    "lane_changing_left",
                    "lane_changing_right",
                ):
                    # Keep a weak all-direction fallback only for extreme proximity.
                    min_dist = self._estimate_min_social_distance(self.current_raw_features)
                    if min_dist < 2.0:
                        forced_subaction = "speed_down_hard"

                if forced_subaction is not None:
                    self._cached_subaction = forced_subaction
                else:
                    # Re-evaluate every tick to keep WM refinement fully closed-loop.
                    self._cached_subaction = self._select_subaction(base_action, self.current_raw_features.copy())
                cmd = dict(cmd)
                cmd["SubAction"] = self._cached_subaction

        control = self.translate_gpt_command(cmd, debug=debug)
        if control is None:
            control = self.normal_behavior(debug=debug)

        self.last_phys_action = [control.throttle, control.steer, control.brake]
        return control

    def reset_wm_state(self):
        with self.wm_lock:
            self.wm.reset()
        self.current_raw_features = None
        self.last_phys_action = [0.0, 0.0, 0.0]
        self._cached_subaction = None
        self.extra_prompt_hint = ""
