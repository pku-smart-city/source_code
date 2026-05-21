import numpy as np

class CarlaTextInterpreter:
    def __init__(self):
        # Feature indices
        self.IDX_V = 0 
        self.IDX_LAT_DIST = 9

    def describe_current_state(self, raw_vec):
        v = raw_vec[self.IDX_V]
        lat_dist = raw_vec[self.IDX_LAT_DIST]
        return f"Current Speed: {v*3.6:.1f} km/h. Lane Offset: {lat_dist:.2f}m. "

    def interpret_wm_prediction(self, action_name, ego_traj, social_trajs):
        """Provide objective future-prediction summaries only."""
        if ego_traj is None or len(ego_traj) == 0: return ""
        
        horizon = len(ego_traj)
        # Compute predicted average speed.
        avg_v = np.mean(np.sqrt(ego_traj[:, 0]**2 + ego_traj[:, 1]**2)) * 3.6
        
        # 1) Compute predicted distance to surrounding vehicles.
        min_social_dist = 100.0
        for t in range(horizon):
            for n in range(5):
                dx, dy = social_trajs[t, n, 0], social_trajs[t, n, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 0.5: continue # Ignore ego/self artifact.
                min_social_dist = min(min_social_dist, dist)

        # 2) Compute predicted lane deviation.
        max_lat_dev = np.max(np.abs(ego_traj[:, self.IDX_LAT_DIST]))
        
        # Build descriptive text only.
        res = f"- If {action_name.upper()}: Pred. Speed {avg_v:.1f} km/h, "
        res += f"Pred. Min Dist to others: {min_social_dist:.1f}m, "
        res += f"Pred. Max Lane Offset: {max_lat_dev:.2f}m.\n"
        
        # Use risk as a label only.
        if min_social_dist < 4.5 or max_lat_dev > 1.2:
            res = res.replace("\n", " [!] Potential Risk detected.\n")
            
        return res
