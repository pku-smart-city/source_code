# collect_traj_data.py
# -*- coding: utf-8 -*-

import os
import argparse
import random
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym

from traj_dataset import (
    DataConfig,
    obs_to_node_features_ego_relative,
    build_adjacency_from_obs,
    pick_slot_indices,
    replace_invalid_slot,
    get_lane_id,
)


def get_dt_from_env(env) -> float:
    cfg = getattr(env.unwrapped, "config", {}) or {}
    pf = float(cfg.get("policy_frequency", 1))
    pf = max(pf, 1.0)
    return 1.0 / pf


def _get_action_name_to_id(env) -> Dict[str, int]:
    cand = []
    for obj in [getattr(env, "action_type", None), getattr(env.unwrapped, "action_type", None)]:
        if obj is None:
            continue
        actions = getattr(obj, "actions", None)
        if actions is not None:
            cand.append(actions)

    if not cand:
        return {"IDLE": 0, "SLOWER": 4, "FASTER": 3, "LANE_LEFT": 1, "LANE_RIGHT": 2}

    actions = cand[0]

    # list/tuple
    if isinstance(actions, (list, tuple)):
        return {str(name): i for i, name in enumerate(actions)}

    # dict
    if isinstance(actions, dict):
        # name->int
        ok = True
        out = {}
        for k, v in actions.items():
            if isinstance(v, (int, np.integer)):
                out[str(k)] = int(v)
            elif isinstance(v, str) and v.isdigit():
                out[str(k)] = int(v)
            else:
                ok = False
                break
        if ok and out:
            return out

        # int->name / "int"->name
        out = {}
        rev_ok = True
        for k, v in actions.items():
            if isinstance(k, (int, np.integer)):
                kid = int(k)
            elif isinstance(k, str) and k.isdigit():
                kid = int(k)
            else:
                rev_ok = False
                break
            if not isinstance(v, str):
                rev_ok = False
                break
            out[str(v)] = kid
        if rev_ok and out:
            return out

    # fallback
    return {"IDLE": 0, "SLOWER": 4, "FASTER": 3, "LANE_LEFT": 1, "LANE_RIGHT": 2}


def choose_action_keep_lane_speed_control(
    obs: np.ndarray,
    cfg: DataConfig,
    action_map: Dict[str, int],
    desired_speed: float = 28.0,
    safe_dist: float = 15.0,
    ttc_min: float = 2.0,
) -> int:
    arr = np.asarray(obs, dtype=np.float32)
    ego_x, ego_y, ego_vx = float(arr[0, 1]), float(arr[0, 2]), float(arr[0, 3])
    ego_lane = get_lane_id(ego_y, cfg)

    lane_center = cfg.lane_y0 + ego_lane * cfg.lane_width
    front_idx = None
    front_dx = 1e9
    for i in range(1, arr.shape[0]):
        if arr[i, 0] < cfg.presence_th:
            continue
        x, y = float(arr[i, 1]), float(arr[i, 2])
        if abs(y - lane_center) > cfg.lane_width * 0.5 + 1e-6:
            continue
        dx = x - ego_x
        if dx > 0 and dx < front_dx:
            front_dx = dx
            front_idx = i

    if ego_vx < desired_speed - 0.5:
        base = "FASTER"
    elif ego_vx > desired_speed + 0.5:
        base = "SLOWER"
    else:
        base = "IDLE"

    if front_idx is not None:
        fx, fv = float(arr[front_idx, 1]), float(arr[front_idx, 3])
        dx = fx - ego_x
        rel_speed = ego_vx - fv 

        if rel_speed > 0.1:
            ttc = dx / max(rel_speed, 1e-3)
        else:
            ttc = 1e9

        if dx < safe_dist or ttc < ttc_min:
            return action_map.get("SLOWER", action_map.get("IDLE", 0))

        if rel_speed > 0.5 and dx < safe_dist * 2.0:
            return action_map.get("IDLE", 0)

    return action_map.get(base, 0)


def _nearest_vehicle_in_lane_by_x(obs: np.ndarray, lane_id: int, x_anchor: float, cfg: DataConfig) -> Optional[int]:
    arr = np.asarray(obs, dtype=np.float32)
    lane_center = cfg.lane_y0 + lane_id * cfg.lane_width

    best_i = None
    best_dx = 1e9
    for i in range(1, arr.shape[0]):
        if arr[i, 0] < cfg.presence_th:
            continue
        x, y = float(arr[i, 1]), float(arr[i, 2])
        if abs(y - lane_center) > cfg.lane_width * 0.5 + 1e-6:
            continue
        d = abs(x - x_anchor)
        if d < best_dx:
            best_dx = d
            best_i = i

    return best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="highway-v0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="traj_dataset.npz")
    parser.add_argument("--vehicles_count", type=int, default=20)
    parser.add_argument("--lanes_count", type=int, default=5)
    parser.add_argument("--duration", type=int, default=120)  
    parser.add_argument("--history_s", type=float, default=3.0)
    parser.add_argument("--future_steps", type=str, default="1,2,3")

    parser.add_argument("--desired_speed", type=float, default=28.0)
    parser.add_argument("--safe_dist", type=float, default=15.0)
    parser.add_argument("--ttc_min", type=float, default=2.0)

    parser.add_argument("--no_car_patience", type=int, default=10, help="episode end")

    parser.add_argument("--print_every", type=int, default=50)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "vehicles_count": args.vehicles_count,
            "see_behind": True,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": args.lanes_count,
        "duration": args.duration,
    })

    dt = get_dt_from_env(env)
    future_steps = tuple(int(x) for x in args.future_steps.split(",") if x.strip())
    max_future = max(future_steps)
    T = int(np.ceil(args.history_s / dt)) + 1

    cfg = DataConfig(
        lanes_count=args.lanes_count,
        history_s=args.history_s,
        future_steps=future_steps,
    )



    action_map = _get_action_name_to_id(env)
    print("[collect] action_map:", action_map)
    assert "IDLE" in action_map and "SLOWER" in action_map and "FASTER" in action_map, \
        f"Action map missing required keys: {action_map.keys()}"
    print(f"[collect] dt={dt:.3f}s, history_steps(T)={T}, future_steps={future_steps}, max_future={max_future}")
    print(f"[collect] vehicles_count={args.vehicles_count}, lanes_count={args.lanes_count}, duration={args.duration}")
    print(f"[collect] save_condition: slot_valid.any() ")
    print(f"[collect] early_break_condition")


    hist_feats_list: List[np.ndarray] = []
    adj_list: List[np.ndarray] = []
    slot_idx_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    slot_valid_list: List[np.ndarray] = []
    ego_action_list: List[np.ndarray] = []
    future_valid_list: List[np.ndarray] = []  # [N, 3, P]

    total_steps = 0
    total_early_break = 0
    total_match_fail = 0

    for ep in range(args.episodes):
        reset_seed = args.seed + 10007 * ep + random.randint(0, 10_000_000)
        obs, info = env.reset(seed=reset_seed)

        obs_buf = deque(maxlen=T + max_future + 5)
        act_buf = deque(maxlen=T + max_future + 5)

        obs = np.array(obs, dtype=np.float32)
        obs_buf.append(obs)
        act_buf.append(-1)

        done = False
        step = 0
        no_car_steps = 0
        saved_this_ep = 0
        match_fail_this_ep = 0

        ego_lane0 = get_lane_id(float(obs[0, 2]), cfg)
        #print(f"\n[collect][ep={ep+1}/{args.episodes}] reset_seed={reset_seed} ego_lane={ego_lane0} ego_x={obs[0,1]:.1f} ego_y={obs[0,2]:.1f} ego_vx={obs[0,3]:.2f}")

        while not done and step < args.duration:
            a = choose_action_keep_lane_speed_control(
                obs=obs,
                cfg=cfg,
                action_map=action_map,
                desired_speed=args.desired_speed,
                safe_dist=args.safe_dist,
                ttc_min=args.ttc_min,
            )

            obs2, r, done, info, _ = env.step(a)
            obs2 = np.array(obs2, dtype=np.float32)

            obs_buf.append(obs2)
            act_buf.append(int(a))

            if len(obs_buf) >= (T + max_future):
                cur_index = len(obs_buf) - 1 - max_future
                obs_t = obs_buf[cur_index]

                slot_idx = pick_slot_indices(obs_t, cfg)      # [-1 or idx]
                slot_valid = (slot_idx >= 0)

                has_any = bool(slot_valid.any())
                if not has_any:
                    no_car_steps += 1
                else:
                    no_car_steps = 0

                if no_car_steps >= args.no_car_patience:
                    total_early_break += 1
                    print(f"[collect][ep={ep+1}] early break at step={step} (no_any_slot_steps={no_car_steps})")
                    break

                if not has_any:
                    if (step % args.print_every) == 0:
                        ego_lane = get_lane_id(float(obs_t[0, 2]), cfg)
                        #print(f"[collect][ep={ep+1}] step={step} skip_save: slot_valid={slot_valid.tolist()} ego_lane={ego_lane} no_any_slot_steps={no_car_steps}")
                    obs = obs2
                    step += 1
                    total_steps += 1
                    continue

                hist_obs = [obs_buf[cur_index - (T - 1) + i] for i in range(T)]
                hist_feats = np.stack([obs_to_node_features_ego_relative(o, cfg) for o in hist_obs], axis=0)  # [T,V,5]
                adj = build_adjacency_from_obs(obs_t, cfg)

                P = len(future_steps)
                target = np.zeros((3, P, 2), dtype=np.float32)
                future_valid = np.zeros((3, P), dtype=np.bool_)  

                anchors = []
                for s in range(3):
                    vidx = int(slot_idx[s])
                    if vidx < 0:
                        anchors.append(None)
                        continue
                    x_abs = float(obs_t[vidx, 1])
                    y_abs = float(obs_t[vidx, 2])
                    lane_id = get_lane_id(y_abs, cfg)
                    anchors.append((lane_id, x_abs))

                for s in range(3):
                    if anchors[s] is None:
                        continue
                    lane_id, x_anchor = anchors[s]
                    for j, k in enumerate(future_steps):
                        obs_tk = obs_buf[cur_index + k]
                        matched = _nearest_vehicle_in_lane_by_x(obs_tk, lane_id, x_anchor, cfg)
                        if matched is None:
                            match_fail_this_ep += 1
                            total_match_fail += 1
                            continue

                        ego_x, ego_y = float(obs_tk[0, 1]), float(obs_tk[0, 2])
                        ox, oy = float(obs_tk[matched, 1]), float(obs_tk[matched, 2])
                        target[s, j, 0] = ox - ego_x
                        target[s, j, 1] = oy - ego_y
                        future_valid[s, j] = True

                slot_idx_fixed = replace_invalid_slot(slot_idx)

                hist_feats_list.append(hist_feats)
                adj_list.append(adj)
                slot_idx_list.append(slot_idx_fixed)
                target_list.append(target)
                slot_valid_list.append(slot_valid)
                ego_act_t = act_buf[cur_index]
                ego_action_list.append(np.array([ego_act_t], dtype=np.int64))
                future_valid_list.append(future_valid)

                saved_this_ep += 1

                if (saved_this_ep <= 3) or ((step % args.print_every) == 0):
                    ego_lane = get_lane_id(float(obs_t[0, 2]), cfg)
                    ego_x = float(obs_t[0, 1])
                    dxs = []
                    for name, idx in zip(["L", "K", "R"], slot_idx.tolist()):
                        if idx < 0:
                            dxs.append(f"{name}:None")
                        else:
                            dxs.append(f"{name}:{float(obs_t[int(idx),1]) - ego_x:+.1f}")
                    fv = future_valid.astype(int).tolist()  # 0/1 更直观
                    # print(
                    #     f"[collect][ep={ep+1}] step={step} SAVE#{saved_this_ep} ego_lane={ego_lane} "
                    #     f"slot_valid={slot_valid.tolist()} slots={slot_idx.tolist()} dxs({', '.join(dxs)}) "
                    #     f"future_valid={fv} action={ego_act_t} match_fail_ep={match_fail_this_ep}"
                    # )

            obs = obs2
            step += 1
            total_steps += 1

        print(f"[collect][ep={ep+1}] done steps={step}, saved_samples={saved_this_ep}, match_fail={match_fail_this_ep}, early_break={'YES' if (no_car_steps>=args.no_car_patience) else 'NO'}")

    if len(hist_feats_list) == 0:
        raise RuntimeError("No samples collected! Try increasing vehicles_count or relaxing filters.")

    # 保存
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        hist_feats=np.stack(hist_feats_list, axis=0),      # [N,T,V,5]
        adj=np.stack(adj_list, axis=0),                    # [N,V,V]
        slot_idx=np.stack(slot_idx_list, axis=0),          # [N,3]  
        target=np.stack(target_list, axis=0),              # [N,3,P,2]
        slot_valid=np.stack(slot_valid_list, axis=0),      # [N,3]  
        future_valid=np.stack(future_valid_list, axis=0),  # [N,3,P] 
        ego_action=np.stack(ego_action_list, axis=0),      # [N,1]
        meta=np.array([{
            "dt": dt,
            "T": T,
            "history_s": args.history_s,
            "future_steps": future_steps,
            "lanes_count": args.lanes_count,
            "vehicles_count": args.vehicles_count,
            "action_map": action_map,
            "collector": "keep_lane_speed_control",
            "desired_speed": args.desired_speed,
            "safe_dist": args.safe_dist,
            "ttc_min": args.ttc_min,
            "save_condition": "slot_valid_any",
            "no_car_patience": args.no_car_patience,
            "print_every": args.print_every,
            "total_steps": total_steps,
            "total_early_break": total_early_break,
            "total_match_fail": total_match_fail,
            "total_samples": len(hist_feats_list),
            "note": "targets use lane+nearest-x matching to reduce vehicle reindexing noise"
        }], dtype=object)
    )

    print("\n[collect] ===== summary =====")
    print(f"[collect] saved dataset to {args.out}")
    print(f"[collect] dt={dt:.3f}s, history_steps(T)={T}, future_steps={future_steps}")
    print(f"[collect] total_samples={len(hist_feats_list)}")
    print(f"[collect] total_steps_simulated={total_steps}")
    print(f"[collect] early_break_episodes={total_early_break}/{args.episodes}")
    print(f"[collect] total_match_fail={total_match_fail}")


if __name__ == "__main__":
    main()
