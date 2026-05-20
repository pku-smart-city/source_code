# --- START OF FILE pid_utils.py ---
import numpy as np
import math

class PurePID:
    """
    Pure mathematical PID controller.
    Used in both actions.py (real execution) and wm_adapter.py (imagination rollout).
    Keeps "imagined" and "executed" dynamics consistent.
    """
    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.05):
        self.Kp = K_P
        self.Kd = K_D
        self.Ki = K_I
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, error):
        self.integral += error * self.dt
        # Clamp integral term to avoid windup.
        self.integral = np.clip(self.integral, -5.0, 5.0)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

def calculate_steering_pure(current_yaw_rad, current_lat_dist, target_lat_dist):
    """
    Simplified lateral controller used inside WM.
    Goal: minimize lateral deviation.
    """
    # Simple P-control to produce a target heading.
    # If we are to the left of target (lat > target), we should steer right (negative yaw).
    lat_error = target_lat_dist - current_lat_dist
    
    # Desired heading correction (gain 0.15).
    target_yaw = np.clip(0.15 * lat_error, -0.5, 0.5)
    
    # Heading error.
    angle_diff = target_yaw - current_yaw_rad
    
    return angle_diff
