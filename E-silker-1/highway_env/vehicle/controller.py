import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [
                (path[i], path[i + 1], None) for i in range(len(path) - 1)
            ]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads))


class MDPVehicle(ControlledVehicle):   #默认的被控车辆
    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)

    #########################################新增规则控制#########################################
    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    #########################################新增规则控制#########################################
    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
    ) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)


    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(
            np.clip(self.speed_index, 0, self.target_speeds.size - 1)
        )
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    #########################################新增规则控制#########################################
    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    def act_v2(self, action: Union[dict, str] = None):
        """
                Execute an action.

                For now, no action is supported because the vehicle takes all decisions
                of acceleration and lane changes on its own, based on the IDM and MOBIL models.

                :param action: the action
                """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )

        # if self.v_id == 3 or self.v_id == 0:
        #     print(f'------------------{self.v_id}-----------------')
        #     print(self.speed)
        #     print(self.position)
        #     print(action)
        #     print(front_vehicle.v_id if front_vehicle else None)
        #
        #     print(rear_vehicle.v_id if rear_vehicle else None)
        #
        #     # print(front_vehicle.speed if front_vehicle else None)
        #     # print(front_vehicle.position if front_vehicle else None)
        #     # print(rear_vehicle.speed if rear_vehicle else None)
        #     # print(rear_vehicle.position if rear_vehicle else None)
        #     print('-----------------------------------')

        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        # Skip ControlledVehicle.act(), or the command will be overridden.
        Vehicle.act(self, action)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change is already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if (
                        v is not self
                        and v.lane_index != self.target_lane_index
                        and isinstance(v, ControlledVehicle)
                        and v.target_lane_index == self.target_lane_index
                    ):
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # # else, at a given frequency,
        # if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
        #     return
        # self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                self.position
            ):
                continue
            # Only change lane when the vehicle is moving
            if np.abs(self.speed) < 1:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
            )
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True



    #########################################新增规则控制#########################################

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
