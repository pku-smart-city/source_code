import math
import numpy as np
import carla

class FeatureExtractor:
    def __init__(self, ego_vehicle, world):
        self.ego = ego_vehicle
        self.world = world
        self.map = world.get_map()
        self.max_neighbors = 5

    def get_features_vector(self):
        # 1) Ego physical features (10 dims)
        vf, vl, vz = self._get_local_velocity(self.ego)
        acc = self.ego.get_acceleration()
        t = self.ego.get_transform()
        yaw_rad = math.radians(t.rotation.yaw)
        
        yaw_err, lat_dist, road_wps = self._get_road_features(t)
        
        # 2) Build the first 10 dims: [vf, vl, vz, ax, ay, az, sin, cos, yaw_err, lat_dist]
        ego_row = [vf, vl, vz, acc.x, acc.y, acc.z, 
                   math.sin(yaw_rad), math.cos(yaw_rad), 
                   yaw_err, lat_dist]
        
        # 3) Road features (8 dims: relative dx/dy for 4 future waypoints)
        # 4) Neighbor features (30 dims: 5 vehicles * 6 features)
        neighbors = self._get_neighbor_features(t, vf, vl)
        
        full_vec = np.array(ego_row + road_wps + neighbors, dtype=np.float32)
        return full_vec

    def _get_local_velocity(self, vehicle):
        v = vehicle.get_velocity()
        t = vehicle.get_transform()
        yaw = math.radians(t.rotation.yaw)
        forward_v = v.x * math.cos(yaw) + v.y * math.sin(yaw)
        lateral_v = -v.x * math.sin(yaw) + v.y * math.cos(yaw)
        return forward_v, lateral_v, v.z

    def _get_road_features(self, t):
        try:
            wp = self.map.get_waypoint(t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except:
            wp = self.map.get_waypoint(t.location)

        # Yaw Error
        yaw_err = t.rotation.yaw - wp.transform.rotation.yaw
        if yaw_err > 180: yaw_err -= 360
        elif yaw_err < -180: yaw_err += 360
        
        # Lat Dist: keep consistent with training data collection (unsigned distance).
        lat_dist = t.location.distance(wp.transform.location)
        
        # Future waypoints (relative coordinates)
        road_wps = []
        cos_y, sin_y = math.cos(math.radians(t.rotation.yaw)), math.sin(math.radians(t.rotation.yaw))
        for d in [5, 10, 20, 50]:
            target_wps = wp.next(d)
            target_wp = target_wps[0] if target_wps else wp
            dx_g, dy_g = target_wp.transform.location.x - t.location.x, target_wp.transform.location.y - t.location.y
            road_wps.extend([dx_g * cos_y + dy_g * sin_y, -dx_g * sin_y + dy_g * cos_y])
            
        return yaw_err, lat_dist, road_wps

    def _get_neighbor_features(self, ego_t, ego_vf, ego_vl):
        all_v = self.world.get_actors().filter('vehicle.*')
        neighbors = []
        for v in all_v:
            if v.id == self.ego.id: continue
            v_t = v.get_transform()
            dist = ego_t.location.distance(v_t.location)
            if dist < 60.0:
                # Relative position and relative velocity
                cos_y, sin_y = math.cos(math.radians(ego_t.rotation.yaw)), math.sin(math.radians(ego_t.rotation.yaw))
                dx_g, dy_g = v_t.location.x - ego_t.location.x, v_t.location.y - ego_t.location.y
                dx, dy = dx_g * cos_y + dy_g * sin_y, -dx_g * sin_y + dy_g * cos_y
                
                v_target = v.get_velocity()
                ego_vx_g, ego_vy_g = ego_vf * cos_y - ego_vl * sin_y, ego_vf * sin_y + ego_vl * cos_y
                dvx, dvy = (v_target.x - ego_vx_g) * cos_y + (v_target.y - ego_vy_g) * sin_y, \
                           -(v_target.x - ego_vx_g) * sin_y + (v_target.y - ego_vy_g) * cos_y
                
                neighbors.append((dist, [dx, dy, dvx, dvy, v.bounding_box.extent.x*2, v.bounding_box.extent.y*2]))
        
        neighbors = sorted(neighbors, key=lambda x: x[0])[:self.max_neighbors]
        flat = []
        for i in range(self.max_neighbors):
            if i < len(neighbors): flat.extend(neighbors[i][1])
            else: flat.extend([100.0, 100.0, 0.0, 0.0, 4.8, 1.8])
        return flat
