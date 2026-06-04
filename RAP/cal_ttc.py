import numpy as np
from typing import Dict, Any, List, Tuple
import math

def compute_ttc_from_obs(
    obs: np.ndarray,
    lanes_count: int = 5,
    lane_width: float = 4.0,
    lane_y0: float = 0.0,
    presence_th: float = 0.5,
    rel_v_eps: float = 0.1,
    max_ttc: float = 1e6,
) -> Dict[str, Any]:
    """
    Notes (kept consistent with your original return format):
    - keep: TTC is still computed as min(TTC_front, TTC_rear) on the ego lane.
    - left/right: TTC is computed from the *nearest* vehicle on that target lane
      (nearest in longitudinal distance |dx|), using ego/target coordinates directly.
    """

    arr = np.array(obs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    ego = arr[0]
    ego_x, ego_y, ego_vx = float(ego[1]), float(ego[2]), float(ego[3])

    def lane_center(lane_id: int) -> float:
        return lane_y0 + lane_id * lane_width

    def get_lane_id(y: float) -> int:
        lid = int(np.round((y - lane_y0) / lane_width))
        return int(np.clip(lid, 0, lanes_count - 1))

    ego_lane = get_lane_id(ego_y)

    # Group vehicles by lane (same as before, but TTC will use x/y directly)
    lane_vehicles = {lid: [] for lid in range(lanes_count)}
    for idx, row in enumerate(arr[1:], start=1):
        if row[0] < presence_th:
            continue
        x, y, vx = float(row[1]), float(row[2]), float(row[3])
        lid = get_lane_id(y)
        if abs(y - lane_center(lid)) <= lane_width * 0.5 + 1e-6:
            lane_vehicles[lid].append((idx, x, y, vx))

    def lane_ttc(target_lane: int, mode: str = "keep") -> Dict[str, Any]:
        """
        mode:
          - "keep": return TTC as min(front, rear) for ego lane.
          - "side": return TTC as TTC to the nearest vehicle (min |dx|) on target lane.
        """
        vehicles = lane_vehicles.get(target_lane, [])

        # Identify nearest front and nearest rear (for reporting)
        front = None  # (idx, dx, vx)
        rear = None   # (idx, dx_back, vx)

        for idx, x, y, vx in vehicles:
            dx = x - ego_x
            if dx > 0:
                if front is None or dx < front[1]:
                    front = (idx, dx, vx)
            else:
                dx_back = ego_x - x
                if dx_back > 0:
                    if rear is None or dx_back < rear[1]:
                        rear = (idx, dx_back, vx)

        # Standard longitudinal TTCs (for reporting / keep-lane min)
        ttc_front = float("inf")
        if front is not None:
            _, dx, v_front = front
            rel_v = ego_vx - v_front  # closing speed to front
            if rel_v > rel_v_eps:
                ttc_front = dx / rel_v

        ttc_rear = float("inf")
        if rear is not None:
            _, dx_back, v_rear = rear
            rel_v_back = v_rear - ego_vx  # closing speed from rear
            if rel_v_back > rel_v_eps:
                ttc_rear = dx_back / rel_v_back

        # --- Key fix: for LEFT/RIGHT (side lanes), compute TTC w.r.t. the NEAREST vehicle ---
        ttc_nearest = float("inf")
        nearest = None  # (idx, dx_signed, vx)

        if vehicles:
            # nearest in longitudinal distance |dx|
            for idx, x, y, vx in vehicles:
                dx_signed = x - ego_x
                if nearest is None or abs(dx_signed) < abs(nearest[1]):
                    nearest = (idx, dx_signed, vx)

            if nearest is not None:
                _, dx_signed, v_other = nearest
                if dx_signed >= 0:
                    # other ahead: collision only if ego is faster
                    closing = ego_vx - v_other
                    if closing > rel_v_eps:
                        ttc_nearest = dx_signed / closing
                else:
                    # other behind: collision only if other is faster
                    closing = v_other - ego_vx
                    if closing > rel_v_eps:
                        ttc_nearest = (-dx_signed) / closing

        # choose TTC depending on mode
        if mode == "side":
            ttc_min = ttc_nearest
        else:
            ttc_min = min(ttc_front, ttc_rear)

        if ttc_min == float("inf"):
            ttc_min = float("inf")
        else:
            ttc_min = float(ttc_min)

        return {
            "lane": target_lane,
            "ttc": ttc_min,
            "ttc_front": float(ttc_front),
            "ttc_rear": float(ttc_rear),
            "front": front,  # (veh_idx, dx, vx) or None
            "rear": rear,    # (veh_idx, dx_back, vx) or None
        }

    # keep/left/right
    keep_info = lane_ttc(ego_lane, mode="keep")

    left_info = None
    if ego_lane - 1 >= 0:
        left_info = lane_ttc(ego_lane - 1, mode="side")

    right_info = None
    if ego_lane + 1 < lanes_count:
        right_info = lane_ttc(ego_lane + 1, mode="side")

    # overall min TTC
    ttc_candidates = [keep_info["ttc"]]
    if left_info is not None:
        ttc_candidates.append(left_info["ttc"])
    if right_info is not None:
        ttc_candidates.append(right_info["ttc"])

    tmp = [t if np.isfinite(t) else max_ttc for t in ttc_candidates]
    ttc_min_all = min(tmp)
    if ttc_min_all >= max_ttc:
        ttc_min_all = float("inf")

    return {
        "ego_lane": ego_lane,
        "keep": keep_info,
        "left": left_info,
        "right": right_info,
        "ttc_min_all": ttc_min_all,
    }

