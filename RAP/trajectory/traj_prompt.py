from typing import Dict, Any, List, Tuple, Optional
import math


def format_traj_pred_for_prompt(
    traj_pred: Dict[str, Any],
    delimiter: str = "####",
    close_gap_m: float = 8.0,
    cutin_gap_m: float = 12.0,
    error_margin: float = 3.0  
) -> str:

    if not traj_pred or "lanes" not in traj_pred:
        return ""

    ego_lane = traj_pred.get("ego_lane", None)
    horizon = float(traj_pred.get("horizon_s", 3.0))
    dt = float(traj_pred.get("dt_s", 1.0))

    lanes_dict = traj_pred.get("lanes", {})
    if ego_lane is None:
        ego_lane = -1

    def _pick_points_at_times(pts: List[Dict[str, Any]], target_ts: List[float], dx_ranges: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        future_pts = [pt for pt in pts if float(pt.get("t", 0.0)) > 0.0]
        if not future_pts:
            return []

        out = []
        used_idx = set()
        for idx, tt in enumerate(target_ts):
            dx_range = dx_ranges[idx] 
            best_i = None
            best_d = 1e9
            for i, pt in enumerate(future_pts):
                if i in used_idx:
                    continue
                d = abs(float(pt.get("t", 0.0)) - tt)
                dx = float(pt.get("dx", 0.0))
                if dx < dx_range[0] or dx > dx_range[1]:  
                    dx = (dx_range[0] + dx_range[1]) / 2
                    pt["dx"] = dx 
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                used_idx.add(best_i)
                out.append(future_pts[best_i])

        if len(out) < len(target_ts):
            remain = [pt for i, pt in enumerate(future_pts) if i not in used_idx]
            remain.sort(key=lambda p: float(p.get("t", 0.0)))
            out.extend(remain[: (len(target_ts) - len(out))])

        out.sort(key=lambda p: float(p.get("t", 0.0)))
        return out


    def _mode_summary(mode: Dict[str, Any], dx_ranges: List[Tuple[float, float]]):
        pts = mode.get("points", [])
        target_ts = [dt * 1.0, dt * 2.0, dt * 3.0]
        pick = _pick_points_at_times(pts, target_ts, dx_ranges)

        future_pts = [pt for pt in pts if float(pt.get("t", 0.0)) > 0.0]
        min_gap = min(abs(float(pt.get("dx", 0.0))) for pt in future_pts) if future_pts else math.inf

        cut_in = False
        cut_in_time = None
        if ego_lane >= 0:
            for pt in future_pts:
                if int(pt.get("lane", -999)) == int(ego_lane) and abs(float(pt.get("dx", 0.0))) < cutin_gap_m:
                    cut_in = True
                    cut_in_time = float(pt.get("t", 0.0))
                    break

        lane_seq = [int(pt.get("lane", -1)) for pt in pts]
        lane_changed = (len(set(lane_seq)) > 1)

        return pick, float(min_gap), bool(cut_in), cut_in_time, bool(lane_changed)

    def _risk_tags(item: Dict[str, Any], mode_min_gap: float, mode_cutin: bool) -> List[str]:
        tags = []
        dx_now = float(item.get("dx_now", 0.0))
        cur_lane = int(item.get("lane", -999))

        if ego_lane >= 0 and cur_lane == int(ego_lane) and mode_min_gap < close_gap_m:
            tags.append("FRONT_CLOSE" if dx_now > 0 else "REAR_CLOSE")

        if ego_lane >= 0 and cur_lane != int(ego_lane) and mode_cutin:
            tags.append("CUT_IN_RISK")

        if mode_min_gap < close_gap_m and (("FRONT_CLOSE" not in tags) and ("REAR_CLOSE" not in tags)):
            tags.append("VERY_CLOSE")

        return tags

    def _fmt_vehicle_block(slot_name: str, item: Optional[Dict[str, Any]], dt: float, error_margin: float) -> str:
        if item is None:
            return f"- {slot_name}: None (no vehicle detected in this lane slot)"

        dx0 = float(item.get("dx_now", 0.0))
        dvx = float(item.get("dvx_now", 0.0))
        closing_speed = -dvx 

        cur_lane = int(item.get("lane", -1))
        pos_word = "ahead" if dx0 > 0 else "behind"
        gap = abs(dx0)

        modes = item.get("modes", [])
        multi_mode = (len(modes) > 1)

        dx_ranges = [
            (dx0 + closing_speed * dt - error_margin, dx0 + closing_speed * dt + error_margin), 
            (dx0 + closing_speed * 2 * dt - error_margin, dx0 + closing_speed * 2 * dt + error_margin),  
            (dx0 + closing_speed * 3 * dt - error_margin, dx0 + closing_speed * 3 * dt + error_margin)  
        ]

        header = (
            f"- {slot_name}: nearest_vehicle_idx={int(item.get('obs_idx', -1))}, "
            f"current_gap={gap:.1f}m ({pos_word}), "
            f"closing_speed={closing_speed:.1f}m/s (positive=gap shrinking), "
            f"current_lane={cur_lane}"
        )

        extra_lateral = ""
        if multi_mode:
            vy = float(item.get("vy", 0.0))
            y_off = float(item.get("y_offset_to_center", 0.0))
            extra_lateral = f"\n  lateral_hint: vy={vy:.2f}m/s, offset_to_lane_center={y_off:.2f}m (nonzero suggests lane change)"

        lines = [header + extra_lateral]

        for m in modes:
            p = float(m.get("p", 1.0))
            pick, min_gap, cutin, cutin_time, lane_changed = _mode_summary(m, dx_ranges)

            if pick:
                seq = "; ".join([f"t={float(pt['t']):.2f}s: dx={float(pt['dx']):.1f}m@lane{int(pt['lane'])}" for pt in pick])
            else:
                seq = "future(3 pts): N/A"

            tags = _risk_tags(item, min_gap, cutin)
            if lane_changed and "LANE_UNCERTAIN" not in tags:
                tags.append("LANE_UNCERTAIN")

            tag_str = f" [{'|'.join(tags)}]" if tags else ""

            cutin_note = ""
            if cutin and ego_lane >= 0 and cur_lane != int(ego_lane) and cutin_time is not None:
                cutin_note = f", enters_ego_lane_at≈{cutin_time:.2f}s"

            if multi_mode:
                lines.append(
                    f"  • hypothesis: {m.get('name','mode')} (p={p:.2f}){tag_str}: {seq} | min_gap≈{min_gap:.1f}m{cutin_note}"
                )
            else:
                lines.append(
                    f"  • forecast{tag_str}: {seq} | min_gap≈{min_gap:.1f}m"
                )

        return "\n".join(lines)

    txt = (
        f"\n{delimiter} Nearby vehicle short-horizon forecast (ego-relative, horizon={horizon:.1f}s, step={dt:.2f}s)\n"
        f"Definitions:\n"
        f"- dx(t) = other_x(t) - ego_x(t) in meters. dx>0 means the vehicle is ahead; dx<0 means behind.\n"
        f"- closing_speed = ego_vx - other_vx in m/s. closing_speed>0 means the gap is shrinking (ego is catching up).\n"
        f"- lane is the lane index predicted from y (0=leftmost).\n"
        f"Risk tags:\n"
        f"- FRONT_CLOSE / REAR_CLOSE: same-lane min gap < {close_gap_m:.0f}m.\n"
        f"- CUT_IN_RISK: currently not in ego lane, but may enter ego lane with gap < {cutin_gap_m:.0f}m.\n"
        f"- LANE_UNCERTAIN: predicted lane may change within horizon.\n"
        f"\n"
        f"{_fmt_vehicle_block('LEFT',  lanes_dict.get('left'), dt, error_margin)}\n"
        f"{_fmt_vehicle_block('KEEP',  lanes_dict.get('keep'), dt, error_margin)}\n"
        f"{_fmt_vehicle_block('RIGHT', lanes_dict.get('right'), dt, error_margin)}\n"
        f"Guidance: If any tag indicates CLOSE or CUT_IN_RISK, prefer conservative actions (keep lane, decelerate, avoid lane change).\n"
        f"If my car's speed plus 3.5 still smaller than the front car, you can choose Acceleration."
    )
    return txt

