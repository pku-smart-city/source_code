#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""
from __future__ import print_function

import argparse
import collections
import datetime
import glob
import json
import logging
import math
import os
import threading
import time

import numpy.random as random
import re
import sys
import weakref

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import cv2
except ImportError:
    cv2 = None

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent_wm import BehaviorAgentWM  # pylint: disable=import-error

waypoints = []

# the log file of collisions and times
collision_path = '' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.txt'
times_path = '' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.txt'
video_path = '' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.mp4'
run_path = ''
status_path = ''
run_args_path = ''
route_live_path = ''
tm_event_path = ''

world_tick = 0
run_times = 0
waypoints_number = 0
last_action = "None"
first_collision = True
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._init_point = None
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        # self.road_start_id = 20
        # self.road_end_id = 4

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0


        # use a fixed type of car
        vehicle_bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        vehicle_bp.set_attribute('role_name', 'hero')

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self._init_point = spawn_point
            self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            self.modify_vehicle_physics(self.player)
        # while self.player is None:
        #     if not self.map.get_spawn_points():
        #         print('There are no spawn points available in your map/town.')
        #         print('Please add some Vehicle Spawn Point to your UE4 scene.')
        #         sys.exit(1)
        #     # spawn_points = self.map.get_spawn_points()
        #     for waypoint in waypoints:
        #
        #         if (waypoint.road_id == 20):
        #             world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=0, g=255, b=255),
        #                                     persistent_lines=True)
        #
        #     # draw_waypoints(waypoints, road_id=4)
        #     filtered_waypoint = []
        #     for waypoint in waypoints:
        #         if (waypoint.road_id == 5):
        #             filtered_waypoint.append(waypoint)
        #
        #     spawn_point = filtered_waypoint[0].transform
        #     spawn_point.location.z += 2
        #
        #     self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        #     self.modify_vehicle_physics(self.player)



        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as exc:
                    logging.warning("Destroy actor failed: %s", exc)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display, spawn_points=None):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

        # # render the mark points
        # if spawn_points:
        #     for index, spawn_point in enumerate(spawn_points):
        #         position = spawn_point.location
        #         x, y = position.x, position.y
        #         marker_text = str(index)
        #         font = pygame.font.Font(None, 24)
        #         text = font.render(marker_text, True, (255, 255, 255))
        #         text_rect = text.get_rect()
        #         text_rect.center = (int(x), int(y))
        #         display.blit(text, text_rect)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self.collided = False
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        self.collided = True

        # write collision information into the log file
        def read_last_line(file_name):
            if not os.path.exists(file_name):
                open(file_name, 'w').close()
                return None
            else:
                with open(file_name, 'r') as file:
                    lines = file.readlines()
                    return lines[-1] if lines else None
        run_times_current = str(run_times)  # run_time_number
        waypoints_number_current = str(waypoints_number)  # waypoints
        other_actor = event.other_actor
        other_type = other_actor.type_id if other_actor is not None else "unknown"
        other_id = other_actor.id if other_actor is not None else "unknown"
        other_loc = None
        if other_actor is not None:
            try:
                loc = other_actor.get_location()
                other_loc = f"({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})"
            except Exception:
                other_loc = "unknown"
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        collision_record_current = "collision time ticks: " + str(
            world_tick) + " run_time_number: " + run_times_current + " waypoints: " + waypoints_number_current + " last_action: " + str(
            last_action) + " other_actor_type: " + str(other_type) + " other_actor_id: " + str(other_id) + " other_actor_loc: " + str(
            other_loc) + " impulse: " + "{:.4f}".format(intensity) + "\n"
        last_line = read_last_line(collision_path)
        should_write = True

        if last_line:
            parts = last_line.split()
            run_times_last = parts[parts.index("run_time_number:") + 1]
            waypoints_last = parts[parts.index("waypoints:") + 1]

            # if the run time and waypoints is the same with the last line, skip writing
            if run_times_last == run_times_current and waypoints_last == waypoints_number_current:
                should_write = False

        if should_write:
            with open(collision_path, 'a') as file:
                file.write(collision_record_current)

        # with open(collision_path, 'a') as file:
        #     collision_record = "collision " + " time ticks: " + str(world_tick) + " run_time_number: " + str(run_times) + " waypoints: " + str(waypoints_number) + " last_action: " + str(last_action) + "\n"
        #     file.write(collision_record)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        event.other_actor.destroy()

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        self.event_count = 0
        self.solid_cross_count = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        self.event_count += 1
        if any('Solid' in str(x) for x in lane_types):
            self.solid_cross_count += 1
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = os.getenv("MADA_DISABLE_VIDEO_RECORDING", "0") != "1"
        self.video_writer = None
        self.route_video_writer = None
        self.route_video_path = None
        self.route_output_dir = None
        self.route_image_output_dir = None
        self.current_route_index = 1
        self.image_output_dir = os.path.join(os.path.dirname(video_path), "frames")
        self.image_count = 0
        self._recent_frames = collections.deque(maxlen=24)
        self._frame_seq = 0
        self._camera_lock = threading.Lock()
        if not self.recording:
            logging.info("Continuous video recording disabled via MADA_DISABLE_VIDEO_RECORDING=1")
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def destroy(self):
        with self._camera_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if self.route_video_writer is not None:
                self.route_video_writer.release()
                self.route_video_writer = None
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except Exception as exc:
                logging.warning("Destroy camera sensor failed: %s", exc)
            self.sensor = None

    def start_route_recording(self, route_index):
        with self._camera_lock:
            self.current_route_index = int(route_index)
            run_dir = os.path.dirname(video_path) or '.'
            self.route_output_dir = os.path.join(run_dir, "routes", f"route_{self.current_route_index:02d}")
            self.route_video_path = os.path.join(self.route_output_dir, "driving.mp4")
            self.route_image_output_dir = os.path.join(self.route_output_dir, "frames")
            os.makedirs(self.route_output_dir, exist_ok=True)
            if self.route_video_writer is not None:
                self.route_video_writer.release()
                self.route_video_writer = None
            self._recent_frames.clear()
            self._frame_seq = 0

    def dump_recent_frames(self, reason, tick_value):
        if cv2 is None:
            return
        with self._camera_lock:
            if not self._recent_frames:
                return
            recent_frames = list(self._recent_frames)
            base_dir = self.route_output_dir if self.route_output_dir else (os.path.dirname(video_path) or '.')
            if not recent_frames:
                return
            keyframe_dir = os.path.join(base_dir, "keyframes")
            os.makedirs(keyframe_dir, exist_ok=True)
            reason = str(reason).replace(" ", "_")
            for i, frame in enumerate(recent_frames):
                out_name = f"{reason}_tick{int(tick_value):05d}_{i:02d}.jpg"
                cv2.imwrite(os.path.join(keyframe_dir, out_name), frame)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            frame_bgr = array[:, :, :3]
            frame_rgb = frame_bgr[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            with self._camera_lock:
                self._frame_seq += 1
                if self._frame_seq % 5 == 0:
                    self._recent_frames.append(frame_bgr.copy())
        if self.recording:
            if self.sensors[self.index][0].startswith('sensor.camera'):
                if cv2 is not None:
                    with self._camera_lock:
                        if self.video_writer is None:
                            os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.video_writer = cv2.VideoWriter(
                                video_path, fourcc, 20.0, (image.width, image.height))
                        if self.video_writer is not None and self.video_writer.isOpened():
                            self.video_writer.write(frame_bgr)
                        if self.route_video_path:
                            if self.route_video_writer is None:
                                os.makedirs(os.path.dirname(self.route_video_path), exist_ok=True)
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.route_video_writer = cv2.VideoWriter(
                                    self.route_video_path, fourcc, 20.0, (image.width, image.height))
                            if self.route_video_writer is not None and self.route_video_writer.isOpened():
                                self.route_video_writer.write(frame_bgr)
                else:
                    os.makedirs(self.image_output_dir, exist_ok=True)
                    image.save_to_disk(os.path.join(self.image_output_dir, f'{self.image_count:08d}.png'))
                    if self.route_image_output_dir:
                        os.makedirs(self.route_image_output_dir, exist_ok=True)
                        image.save_to_disk(
                            os.path.join(self.route_image_output_dir, f'{self.image_count:08d}.png'))
                    self.image_count += 1

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    traffic_manager = None

    try:
        with open(status_path, 'w', encoding='utf-8') as f:
            f.write('status: running\n')
            f.write('start_time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n')

        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(float(args.carla_timeout))

        tm_ports = [args.tm_port, args.tm_port + 10, args.tm_port + 20]
        tm_error = None
        traffic_manager = None
        for tm_port in tm_ports:
            try:
                traffic_manager = client.get_trafficmanager(tm_port)
                if tm_port != args.tm_port:
                    logging.warning("Traffic manager port %d unavailable, fallback to %d", args.tm_port, tm_port)
                break
            except RuntimeError as exc:
                tm_error = exc
                if "bind error" not in str(exc):
                    raise
        if traffic_manager is None:
            raise tm_error
        traffic_manager.global_percentage_speed_difference(-50)
        sim_world = client.get_world()
        settings = sim_world.get_settings()
        # Preserve original MADA default world timestep.
        settings.fixed_delta_seconds = 0.0014
        sim_world.apply_settings(settings)

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        traffic_manager.vehicle_percentage_speed_difference(world.player, -50)

        # Standalone hybrid eval script: fixed to BehaviorAgentWM(cautious) + TM control.
        agent = BehaviorAgentWM(
            world.player,
            behavior="cautious",
            exp_path=args.log_root,
            openai_api_key=args.openai_api_key,
            openai_api_base=args.openai_api_base,
        )
        if args.ablate_wm:
            # Ablation mode: keep LLM loop active, but disable WM contribution.
            agent.enable_refined_subaction = False
            if hasattr(agent, "extra_prompt_hint"):
                agent.extra_prompt_hint = ""
            if hasattr(agent, "_update_wm_prompt_hint"):
                def _disable_wm_prompt_hint():
                    agent.extra_prompt_hint = ""
                agent._update_wm_prompt_hint = _disable_wm_prompt_hint
            try:
                if hasattr(agent, "wm") and hasattr(agent.wm, "tick_update"):
                    agent.wm.tick_update = (lambda *unused_args, **unused_kwargs: None)
            except Exception as exc:
                logging.warning("WM ablation patch failed: %s", exc)
            logging.info("wm ablation enabled: WM subaction/prompt hint disabled, LLM loop remains active")
        else:
            agent.enable_refined_subaction = True
            logging.info("wm refined subaction enabled: true (default)")
        use_tm_control = True
        if use_tm_control:
            traffic_manager.vehicle_percentage_speed_difference(world.player, 0.0)

        # Set the agent destination
        # fixed destination
        # filtered_waypoint = []
        # for waypoint in waypoints:
        #     if (waypoint.road_id == 11):
        #         filtered_waypoint.append(waypoint)
        #
        # target_waypoint = filtered_waypoint[0]
        # target_location = target_waypoint.transform.location
        # target_location.z += 2
        # agent.set_destination(target_location)
        #
        # clock = pygame.time.Clock()
        #
        # while True:
        #     clock.tick()
        #     if args.sync:
        #         world.world.tick()
        #     else:
        #         world.world.wait_for_tick()
        #     if controller.parse_events():
        #         return
        #
        #     # hud.render(display, spawn_points)
        #
        #     world.tick(clock)
        #     world.render(display)
        #     pygame.display.flip()
        #
        #     if agent.done():
        #         # if args.loop:
        #         #     agent.set_destination(random.choice(spawn_points).location)
        #         #     world.hud.notification("Target reached", seconds=4.0)
        #         #     print("The target has been reached, searching for another target")
        #         # else:
        #         print("The target has been reached, stopping the simulation")
        #         break
        #         #
        #         #     control = agent.run_step()
        #         #     agent.set_last_speed()
        #         #     agent.set_last_action()
        #         #     if control is not None:
        #         #         control.manual_gear_shift = False
        #         world.player.apply_control(control)


        # Set the first route destination.
        spawn_points = world.map.get_spawn_points()

        def build_route_points(end_location):
            start_wp = world.map.get_waypoint(world.player.get_location())
            end_wp = world.map.get_waypoint(end_location)
            route = agent.trace_route(start_wp, end_wp)
            points = [wp.transform.location for wp, _ in route]
            if len(points) < 2:
                points = [start_wp.transform.location, end_wp.transform.location]
            return points

        def init_route_progress_state(end_location):
            start_distance = world.player.get_location().distance(end_location)
            start_distance = max(1.0, float(start_distance))
            return start_distance, start_distance

        def set_route_destination(end_location):
            if use_tm_control:
                start_wp = world.map.get_waypoint(
                    world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                )
                end_wp = world.map.get_waypoint(
                    end_location, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if start_wp is None or end_wp is None:
                    return
                route = agent.trace_route(start_wp, end_wp)
                path = [wp.transform.location for wp, _ in route]
                if len(path) < 2:
                    path = [end_location]
                try:
                    traffic_manager.set_path(world.player, path)
                except RuntimeError as exc:
                    logging.warning("TrafficManager set_path failed: %s", exc)
                try:
                    traffic_manager.auto_lane_change(world.player, True)
                except RuntimeError as exc:
                    logging.warning("TrafficManager auto_lane_change failed: %s", exc)
                world.player.set_autopilot(True, traffic_manager.get_port())
            else:
                agent.set_destination(end_location)

        def safe_tm_call(method_name, *method_args):
            fn = getattr(traffic_manager, method_name, None)
            if fn is None:
                return False
            try:
                fn(*method_args)
                return True
            except RuntimeError as exc:
                logging.warning("TM call %s failed: %s", method_name, exc)
                return False

        def apply_tm_profile(
            lead_distance,
            speed_diff,
            lane_change_pct,
            ignore_lights_pct,
            ignore_signs_pct,
            ignore_vehicles_pct,
            ignore_walkers_pct,
            auto_lane_change=True,
        ):
            safe_tm_call('distance_to_leading_vehicle', world.player, float(lead_distance))
            safe_tm_call('vehicle_percentage_speed_difference', world.player, float(speed_diff))
            safe_tm_call('random_left_lanechange_percentage', world.player, float(lane_change_pct))
            safe_tm_call('random_right_lanechange_percentage', world.player, float(lane_change_pct))
            safe_tm_call('ignore_lights_percentage', world.player, float(ignore_lights_pct))
            safe_tm_call('ignore_signs_percentage', world.player, float(ignore_signs_pct))
            safe_tm_call('ignore_vehicles_percentage', world.player, float(ignore_vehicles_pct))
            safe_tm_call('ignore_walkers_percentage', world.player, float(ignore_walkers_pct))
            safe_tm_call('auto_lane_change', world.player, bool(auto_lane_change))

        def set_tm_unblock_mode(enabled, high_progress=False, stalled_start=False, unblock_attempt=1):
            if not use_tm_control:
                return
            if not enabled:
                apply_tm_profile(
                    lead_distance=2.5,
                    speed_diff=0.0,
                    lane_change_pct=0.0,
                    ignore_lights_pct=0.0,
                    ignore_signs_pct=0.0,
                    ignore_vehicles_pct=0.0,
                    ignore_walkers_pct=0.0,
                    auto_lane_change=True,
                )
                return

            attempt = max(1, int(unblock_attempt))
            high_stage = attempt >= 2
            if stalled_start:
                # Start-stall uses the strongest escape profile.
                if high_stage:
                    apply_tm_profile(0.3, -35.0, 95.0, 3.0, 65.0, 18.0, 25.0, True)
                else:
                    apply_tm_profile(0.5, -26.0, 80.0, 0.0, 40.0, 8.0, 15.0, True)
            elif high_progress:
                # Near-goal stall allows more light/sign bypass than mid-route.
                if high_stage:
                    apply_tm_profile(0.45, -30.0, 72.0, 55.0, 35.0, 20.0, 25.0, True)
                else:
                    apply_tm_profile(0.65, -22.0, 55.0, 32.0, 18.0, 12.0, 15.0, True)
            else:
                # Mid-route keeps first try conservative, escalates on repeated failure.
                if high_stage:
                    apply_tm_profile(0.75, -24.0, 58.0, 18.0, 30.0, 10.0, 20.0, True)
                else:
                    apply_tm_profile(1.0, -12.0, 22.0, 5.0, 10.0, 0.0, 8.0, True)

        def try_tm_soft_reset(end_location):
            if not use_tm_control:
                return False
            try:
                world.player.set_autopilot(False, traffic_manager.get_port())
            except Exception:
                pass

            try:
                world.player.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False
                ))
            except Exception:
                pass
            for _ in range(max(1, args.tm_soft_reset_brake_ticks)):
                tick_world_once()

            cur_wp = world.map.get_waypoint(
                world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
            end_wp = world.map.get_waypoint(
                end_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if cur_wp is not None and end_wp is not None:
                route = agent.trace_route(cur_wp, end_wp)
                path = [wp.transform.location for wp, _ in route]
                if len(path) < 2:
                    path = [end_location]
                try:
                    traffic_manager.set_path(world.player, path)
                    traffic_manager.auto_lane_change(world.player, True)
                except RuntimeError as exc:
                    logging.warning("TM soft reset set_path failed: %s", exc)
                    try:
                        world.player.set_autopilot(True, traffic_manager.get_port())
                    except Exception:
                        pass
                    return False
            try:
                world.player.set_autopilot(True, traffic_manager.get_port())
            except Exception:
                return False
            set_tm_unblock_mode(False)
            return True

        def reset_wm_state_if_supported():
            if hasattr(agent, "reset_wm_state"):
                agent.reset_wm_state()

        def tick_world_once():
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()

        def pick_destination(min_distance=25.0, max_attempts=50):
            ego_loc = world.player.get_location()
            for _ in range(max_attempts):
                candidate = random.choice(spawn_points).location
                if ego_loc.distance(candidate) >= min_distance:
                    return candidate
            return random.choice(spawn_points).location

        def relocate_ego_for_new_route(max_attempts=80, min_clearance=12.0):
            if world.player is None or not spawn_points:
                return False

            if use_tm_control:
                try:
                    world.player.set_autopilot(False, traffic_manager.get_port())
                except Exception:
                    pass

            if world.collision_sensor is not None:
                world.collision_sensor.collided = False
                world.collision_sensor.history.clear()

            for _ in range(max_attempts):
                base_sp = random.choice(spawn_points)
                spawn_point = carla.Transform(
                    carla.Location(base_sp.location.x, base_sp.location.y, base_sp.location.z + 0.3),
                    carla.Rotation(base_sp.rotation.pitch, base_sp.rotation.yaw, base_sp.rotation.roll),
                )
                try:
                    world.player.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    world.player.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass

                world.player.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                world.player.set_transform(spawn_point)
                tick_world_once()
                tick_world_once()

                # Give physics/collision callbacks time to settle after teleport.
                had_settle_collision = False
                for _ in range(6):
                    tick_world_once()
                    if world.collision_sensor is not None and world.collision_sensor.collided:
                        had_settle_collision = True
                        world.collision_sensor.collided = False
                        world.collision_sensor.history.clear()
                        break
                if had_settle_collision:
                    continue

                ego_loc = world.player.get_location()
                ego_tf = world.player.get_transform()
                try:
                    ego_wp = world.map.get_waypoint(
                        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                except Exception:
                    ego_wp = None
                if ego_wp is None:
                    continue

                # Reject teleports that end up too far from lane center (common for curb/sidewalk starts).
                right = ego_tf.get_right_vector()
                dx = ego_loc.x - ego_wp.transform.location.x
                dy = ego_loc.y - ego_wp.transform.location.y
                signed_lat = dx * right.x + dy * right.y
                if abs(signed_lat) > max(0.45, ego_wp.lane_width * 0.18):
                    continue

                yaw_diff = abs((ego_tf.rotation.yaw - ego_wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
                if yaw_diff > 20.0:
                    continue

                overlap = False
                for other in world.world.get_actors().filter('vehicle.*'):
                    if other.id == world.player.id:
                        continue
                    dist = ego_loc.distance(other.get_location())
                    other_speed = other.get_velocity().length() * 3.6
                    dynamic_clearance = min_clearance + min(6.0, other_speed * 0.25)
                    if dist < dynamic_clearance:
                        overlap = True
                        break
                if overlap:
                    continue

                if hasattr(agent, "_static_obstacle_ahead"):
                    try:
                        nearby_static = agent._static_obstacle_ahead(
                            max_distance=8.0, fov_deg=80.0, lateral_limit=3.5
                        )
                    except Exception:
                        nearby_static = None
                    if nearby_static is not None and nearby_static.get("distance", 999.0) < 4.0:
                        continue

                if world.collision_sensor is not None:
                    if world.collision_sensor.collided:
                        world.collision_sensor.collided = False
                        world.collision_sensor.history.clear()
                        continue
                    world.collision_sensor.collided = False
                    world.collision_sensor.history.clear()

                # Reject starts with an immediate lead vehicle too close in-lane.
                try:
                    ego_wp = world.map.get_waypoint(world.player.get_location())
                    v_state, lead_v, lead_dist = agent.collision_and_car_avoid_manager(ego_wp)
                except Exception:
                    v_state, lead_v, lead_dist = False, None, 0.0
                if v_state and lead_v is not None:
                    lead_gap = lead_dist - max(
                        lead_v.bounding_box.extent.y, lead_v.bounding_box.extent.x) - max(
                        world.player.bounding_box.extent.y, world.player.bounding_box.extent.x)
                    if lead_gap < 6.0:
                        continue

                # Keep ego braked for a few settle ticks to avoid instant start overlap.
                clear_start = True
                for _ in range(4):
                    tick_world_once()
                    if world.collision_sensor is not None and world.collision_sensor.collided:
                        world.collision_sensor.collided = False
                        world.collision_sensor.history.clear()
                        clear_start = False
                        break
                    try:
                        ego_wp = world.map.get_waypoint(world.player.get_location())
                        v_state, lead_v, lead_dist = agent.collision_and_car_avoid_manager(ego_wp)
                    except Exception:
                        v_state, lead_v, lead_dist = False, None, 0.0
                    if v_state and lead_v is not None:
                        lead_gap = lead_dist - max(
                            lead_v.bounding_box.extent.y, lead_v.bounding_box.extent.x) - max(
                            world.player.bounding_box.extent.y, world.player.bounding_box.extent.x)
                        if lead_gap < 5.0:
                            clear_start = False
                            break
                if not clear_start:
                    continue

                world.player.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
                return True

            return False

        reset_wm_state_if_supported()
        if not relocate_ego_for_new_route():
            logging.warning("Could not relocate ego to a clearly separated spawn point.")
        destination = pick_destination()
        set_route_destination(destination)
        set_tm_unblock_mode(False)
        current_route_points = build_route_points(destination)
        initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
        current_route_best_ratio = 0.0
        total_routes = args.num_routes
        finished_routes = 0
        success_routes = 0
        collision_routes = 0
        timeout_routes = 0
        high_progress_timeout_routes = 0
        high_progress_timeout_goal_distance_sum = 0.0
        near_goal_timeout_routes = 0
        near_goal_timeout_goal_distance_sum = 0.0
        sum_route_completion = 0.0
        route_ticks = 0
        stuck_ticks = 0
        red_light_hold_ticks = 0
        tm_no_progress_ticks = 0
        tm_replans = 0
        tm_unblock_active = False
        tm_unblock_ticks_left = 0
        tm_unblock_used = 0
        tm_soft_resets = 0
        tm_last_force_lane_change_tick = -1000000
        tm_loop_escape_used = 0
        tm_loop_escape_lock_ticks = 0
        tm_last_loop_escape_tick = -1000000
        tm_post_red_release_countdown_ticks = 0
        tm_red_wait_escape_armed = False
        was_red_light_tick = False
        tm_event_seq = 0
        tm_agent_loop_countdown = 0
        tm_agent_last_action = "normal_behavior"
        tm_wm_hint_missing_ticks = 0
        metric_dt = float(world.world.get_settings().fixed_delta_seconds or 0.05)

        # Extra eval metrics (docs/指标)
        near_miss_ttc_threshold_s = 1.0
        hard_brake_acc_threshold_mps2 = 4.0
        jerk_threshold_mps3 = 8.0
        red_light_violation_speed_kmh = max(0.0, float(args.red_light_violation_speed_kmh))
        red_light_violation_min_ticks = max(1, int(args.red_light_violation_min_ticks))
        stop_sign_detect_distance_m = 16.0
        stop_sign_lateral_limit_m = 4.0
        stop_sign_min_observe_ticks = 3
        stop_sign_full_stop_speed_kmh = 1.0
        offroad_dist_threshold_m = 1.0

        route_timeout_only_routes = 0
        stuck_timeout_routes = 0
        red_light_violation_events = 0
        red_light_violation_routes = 0
        stop_sign_violation_events = 0
        stop_sign_violation_routes = 0
        lane_invasion_events = 0
        lane_invasion_routes = 0
        solid_line_cross_events = 0
        solid_line_cross_routes = 0
        offroad_events = 0
        offroad_ticks = 0
        offroad_routes = 0
        near_miss_events = 0
        near_miss_ticks = 0
        near_miss_routes = 0
        hard_brake_events = 0
        jerk_events = 0
        jerk_abs_sum = 0.0
        jerk_samples = 0
        intervention_events = 0
        intervention_routes = 0
        min_ttc_global = float("inf")
        route_min_ttc_sum = 0.0
        route_min_ttc_valid_count = 0
        metric_total_ticks = 0

        stop_sign_actors = []
        try:
            stop_sign_actors = list(world.world.get_actors().filter("traffic.stop*"))
        except Exception:
            stop_sign_actors = []

        route_min_ttc = float("inf")
        route_had_near_miss = False
        route_had_red_light_violation = False
        route_had_stop_sign_violation = False
        route_had_offroad = False
        route_had_intervention = False
        route_lane_invasion_base = 0
        route_solid_cross_base = 0
        near_miss_latch = False
        red_light_violation_latch = False
        red_light_violation_active_ticks = 0
        offroad_latch = False
        hard_brake_latch = False
        jerk_latch = False
        prev_forward_speed_mps = None
        prev_lon_acc_mps2 = None
        active_stop_sign_id = None
        active_stop_sign_min_speed_kmh = 1e9
        active_stop_sign_seen_ticks = 0

        def _distance_to_driving_lane_center():
            ego_loc = world.player.get_location()
            try:
                wp_exact = world.map.get_waypoint(
                    ego_loc, project_to_road=False, lane_type=carla.LaneType.Driving
                )
            except Exception:
                wp_exact = None
            if wp_exact is not None:
                return 0.0
            try:
                wp_proj = world.map.get_waypoint(
                    ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
            except Exception:
                wp_proj = None
            if wp_proj is None:
                return 1e9
            return float(ego_loc.distance(wp_proj.transform.location))

        def _estimate_front_ttc():
            ego_tf = world.player.get_transform()
            ego_loc = ego_tf.location
            forward = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            ego_v = world.player.get_velocity()
            ego_forward_speed = float(ego_v.x * forward.x + ego_v.y * forward.y + ego_v.z * forward.z)

            best_ttc = float("inf")
            for other in world.world.get_actors().filter("vehicle.*"):
                if other.id == world.player.id:
                    continue
                other_loc = other.get_location()
                dx = other_loc.x - ego_loc.x
                dy = other_loc.y - ego_loc.y
                dz = other_loc.z - ego_loc.z
                forward_dist = dx * forward.x + dy * forward.y + dz * forward.z
                if forward_dist <= 0.5 or forward_dist > 80.0:
                    continue
                lateral = abs(dx * right.x + dy * right.y + dz * right.z)
                if lateral > 2.8:
                    continue
                other_v = other.get_velocity()
                other_forward_speed = float(other_v.x * forward.x + other_v.y * forward.y + other_v.z * forward.z)
                closing_speed = max(0.0, ego_forward_speed - other_forward_speed)
                if closing_speed <= 0.1:
                    continue
                ego_half_len = max(world.player.bounding_box.extent.x, world.player.bounding_box.extent.y)
                oth_half_len = max(other.bounding_box.extent.x, other.bounding_box.extent.y)
                gap = max(0.0, forward_dist - ego_half_len - oth_half_len)
                ttc = gap / closing_speed
                if ttc < best_ttc:
                    best_ttc = ttc
            return best_ttc

        def _nearest_stop_sign_ahead():
            ego_tf = world.player.get_transform()
            ego_loc = ego_tf.location
            forward = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            best_sign = None
            best_dist = 1e9
            for sign in stop_sign_actors:
                if sign is None:
                    continue
                try:
                    s_loc = sign.get_location()
                except Exception:
                    continue
                dx = s_loc.x - ego_loc.x
                dy = s_loc.y - ego_loc.y
                dz = s_loc.z - ego_loc.z
                forward_dist = dx * forward.x + dy * forward.y + dz * forward.z
                if forward_dist < -2.0 or forward_dist > stop_sign_detect_distance_m:
                    continue
                lateral = abs(dx * right.x + dy * right.y + dz * right.z)
                if lateral > stop_sign_lateral_limit_m:
                    continue
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < best_dist:
                    best_dist = dist
                    best_sign = sign
            return best_sign

        def _finalize_active_stop_sign():
            nonlocal active_stop_sign_id, active_stop_sign_min_speed_kmh, active_stop_sign_seen_ticks
            nonlocal stop_sign_violation_events, route_had_stop_sign_violation
            if active_stop_sign_id is None:
                return
            if (
                active_stop_sign_seen_ticks >= stop_sign_min_observe_ticks
                and active_stop_sign_min_speed_kmh > stop_sign_full_stop_speed_kmh
            ):
                stop_sign_violation_events += 1
                route_had_stop_sign_violation = True
            active_stop_sign_id = None
            active_stop_sign_min_speed_kmh = 1e9
            active_stop_sign_seen_ticks = 0

        def _reset_route_metric_state():
            nonlocal route_min_ttc, route_had_near_miss, route_had_red_light_violation
            nonlocal route_had_stop_sign_violation, route_had_offroad, route_had_intervention
            nonlocal route_lane_invasion_base, route_solid_cross_base
            nonlocal near_miss_latch, red_light_violation_latch, red_light_violation_active_ticks, offroad_latch
            nonlocal hard_brake_latch, jerk_latch, prev_forward_speed_mps, prev_lon_acc_mps2
            nonlocal active_stop_sign_id, active_stop_sign_min_speed_kmh, active_stop_sign_seen_ticks
            route_min_ttc = float("inf")
            route_had_near_miss = False
            route_had_red_light_violation = False
            route_had_stop_sign_violation = False
            route_had_offroad = False
            route_had_intervention = False
            route_lane_invasion_base = int(getattr(world.lane_invasion_sensor, "event_count", 0))
            route_solid_cross_base = int(getattr(world.lane_invasion_sensor, "solid_cross_count", 0))
            near_miss_latch = False
            red_light_violation_latch = False
            red_light_violation_active_ticks = 0
            offroad_latch = False
            hard_brake_latch = False
            jerk_latch = False
            prev_forward_speed_mps = None
            prev_lon_acc_mps2 = None
            active_stop_sign_id = None
            active_stop_sign_min_speed_kmh = 1e9
            active_stop_sign_seen_ticks = 0

        def _flush_route_metrics():
            nonlocal red_light_violation_routes, stop_sign_violation_routes
            nonlocal lane_invasion_events, lane_invasion_routes
            nonlocal solid_line_cross_events, solid_line_cross_routes
            nonlocal offroad_routes, near_miss_routes, intervention_routes
            nonlocal min_ttc_global, route_min_ttc_sum, route_min_ttc_valid_count
            _finalize_active_stop_sign()
            lane_events_route = max(
                0,
                int(getattr(world.lane_invasion_sensor, "event_count", 0)) - int(route_lane_invasion_base),
            )
            solid_events_route = max(
                0,
                int(getattr(world.lane_invasion_sensor, "solid_cross_count", 0)) - int(route_solid_cross_base),
            )
            lane_invasion_events += lane_events_route
            solid_line_cross_events += solid_events_route
            if lane_events_route > 0:
                lane_invasion_routes += 1
            if solid_events_route > 0:
                solid_line_cross_routes += 1
            if route_had_red_light_violation:
                red_light_violation_routes += 1
            if route_had_stop_sign_violation:
                stop_sign_violation_routes += 1
            if route_had_offroad:
                offroad_routes += 1
            if route_had_near_miss:
                near_miss_routes += 1
            if route_had_intervention:
                intervention_routes += 1
            if np.isfinite(route_min_ttc):
                min_ttc_global = min(min_ttc_global, route_min_ttc)
                route_min_ttc_sum += route_min_ttc
                route_min_ttc_valid_count += 1

        def start_route_artifact_recording(route_index):
            if world is None or world.camera_manager is None:
                return
            try:
                world.camera_manager.start_route_recording(route_index)
            except Exception as exc:
                logging.warning("Route recording init failed: %s", exc)

        def dump_route_keyframes(reason):
            if world is None or world.camera_manager is None:
                return
            try:
                world.camera_manager.dump_recent_frames(reason, world_tick)
            except Exception as exc:
                logging.warning("Route keyframe dump failed: %s", exc)

        def write_tm_event(event_name, tl_state='unknown', extra=None, route_index_override=None):
            nonlocal tm_event_seq, intervention_events, route_had_intervention
            if not use_tm_control:
                return
            if str(event_name) in {
                "tm_unblock_start",
                "tm_force_lane_change",
                "tm_soft_reset",
                "tm_soft_reset_red_hold",
            }:
                intervention_events += 1
                route_had_intervention = True
            goal_distance_now = None
            try:
                goal_distance_now = float(world.player.get_location().distance(destination))
            except Exception:
                goal_distance_now = None
            rec = {
                "seq": tm_event_seq,
                "event": str(event_name),
                "route_index": int(route_index_override if route_index_override is not None else (finished_routes + 1)),
                "world_tick": int(world_tick),
                "route_ticks": int(route_ticks),
                "progress_percent": round(float(current_route_best_ratio) * 100.0, 2),
                "goal_distance_m": None if goal_distance_now is None else round(goal_distance_now, 2),
                "speed_kmh": round(float(world.player.get_velocity().length() * 3.6), 2),
                "tm_no_progress_ticks": int(tm_no_progress_ticks),
                "stuck_ticks": int(stuck_ticks),
                "red_light_hold_ticks": int(red_light_hold_ticks),
                "tm_post_red_release_countdown_ticks": int(tm_post_red_release_countdown_ticks),
                "traffic_light_state": str(tl_state),
                "tm_replans": int(tm_replans),
                "tm_unblock_used": int(tm_unblock_used),
                "tm_soft_resets": int(tm_soft_resets),
            }
            if extra:
                rec.update(extra)
            try:
                with open(tm_event_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                tm_event_seq += 1
            except Exception as exc:
                logging.warning("TM event logging failed: %s", exc)

        def maybe_step_agent_in_tm():
            """
            Keep BehaviorAgent/BehaviorAgentWM decision loop alive even when TM controls actuation.
            """
            nonlocal tm_agent_loop_countdown, tm_agent_last_action
            global waypoints_number
            if not use_tm_control:
                return None
            interval = max(1, int(args.tm_agent_loop_interval_ticks))
            must_step = (route_ticks <= 2) or (tm_agent_loop_countdown <= 0)
            if not must_step:
                tm_agent_loop_countdown -= 1
                return tm_agent_last_action

            try:
                _ = agent.run_step()
                if hasattr(agent, "set_last_speed"):
                    agent.set_last_speed()
                if hasattr(agent, "set_last_action"):
                    agent.set_last_action()
                if hasattr(agent, "get_last_action"):
                    action_now = agent.get_last_action()
                    if action_now:
                        tm_agent_last_action = action_now
                # In TM mode, route progress bookkeeping uses route-point matching below.
                # Avoid mixing in agent-side waypoint counters, which can drift in hybrid control.
                if (not use_tm_control) and hasattr(agent, "get_waypoint_number"):
                    waypoints_number = agent.get_waypoint_number()
                if args.ablate_wm and hasattr(agent, "extra_prompt_hint"):
                    # Ensure TM hint path receives no WM context under ablation.
                    agent.extra_prompt_hint = ""
            except Exception as exc:
                logging.warning("TM mode agent decision step failed: %s", exc)

            tm_agent_loop_countdown = interval - 1
            return tm_agent_last_action

        def apply_tm_action_hint(action_name):
            """
            TM behavior is coupled to WM risk hints.
            Without WM hint, run TM in uncertainty-safe mode.
            """
            nonlocal tm_wm_hint_missing_ticks
            if not use_tm_control:
                return

            action = (action_name or "").strip().lower()
            base_speed_diff = 0.0
            if action == "speed_up":
                base_speed_diff = float(args.tm_action_hint_speed_up_diff)
            elif action in ("speed_down", "stop"):
                base_speed_diff = float(args.tm_action_hint_slow_down_diff)

            base_left_pct = 0.0
            base_right_pct = 0.0
            lane_change_pct = float(args.tm_action_hint_lane_change_pct)
            if action == "lane_changing_left":
                base_left_pct = lane_change_pct
            elif action == "lane_changing_right":
                base_right_pct = lane_change_pct

            wm_hint = None
            if not args.ablate_wm:
                raw_hint = getattr(agent, "extra_prompt_hint", "")
                if raw_hint:
                    try:
                        wm_hint = json.loads(raw_hint)
                    except Exception:
                        wm_hint = None

            has_wm_hint = isinstance(wm_hint, dict)
            if has_wm_hint:
                tm_wm_hint_missing_ticks = 0
            else:
                tm_wm_hint_missing_ticks += 1

            if not isinstance(wm_hint, dict):
                # No WM context: run conservative uncertainty mode and avoid blind lane-change impulses.
                safe_tm_call('distance_to_leading_vehicle', world.player, float(args.tm_wm_nohint_lead_distance))
                safe_tm_call('vehicle_percentage_speed_difference', world.player, float(args.tm_wm_nohint_speed_diff))
                safe_tm_call('random_left_lanechange_percentage', world.player, 0.0)
                safe_tm_call('random_right_lanechange_percentage', world.player, 0.0)
                safe_tm_call('ignore_lights_percentage', world.player, 0.0)
                safe_tm_call('ignore_signs_percentage', world.player, 0.0)
                safe_tm_call('ignore_vehicles_percentage', world.player, 0.0)
                safe_tm_call('ignore_walkers_percentage', world.player, 0.0)
                safe_tm_call('auto_lane_change', world.player, False)
                if tm_wm_hint_missing_ticks == max(1, int(args.tm_wm_hint_grace_ticks)):
                    write_tm_event(
                        "tm_wm_hint_missing",
                        "unknown",
                        {"wm_hint_missing_ticks": int(tm_wm_hint_missing_ticks)},
                    )
                return

            if tm_unblock_active or tm_loop_escape_lock_ticks > 0:
                return

            front_has = bool(wm_hint.get("wm_front_has", False))
            try:
                front_gap = float(wm_hint.get("wm_front_gap_m", -1.0))
            except Exception:
                front_gap = -1.0
            try:
                front_ttc = float(wm_hint.get("wm_front_ttc_s", -1.0))
            except Exception:
                front_ttc = -1.0
            try:
                lane_offset_norm = float(wm_hint.get("wm_lane_offset_norm", -1.0))
            except Exception:
                lane_offset_norm = -1.0
            try:
                static_ahead_m = float(wm_hint.get("wm_static_ahead_m", -1.0))
            except Exception:
                static_ahead_m = -1.0

            risk_score = 0
            if front_has and front_gap > 0.0:
                if front_gap < 3.0:
                    risk_score += 3
                elif front_gap < 6.0:
                    risk_score += 2
                elif front_gap < 10.0:
                    risk_score += 1
            if front_ttc > 0.0:
                if front_ttc < 1.5:
                    risk_score += 3
                elif front_ttc < 2.5:
                    risk_score += 2
                elif front_ttc < 4.0:
                    risk_score += 1
            if static_ahead_m > 0.0:
                if static_ahead_m < 4.0:
                    risk_score += 2
                elif static_ahead_m < 8.0:
                    risk_score += 1
            if lane_offset_norm > 0.0:
                if lane_offset_norm >= 1.3:
                    risk_score += 2
                elif lane_offset_norm >= 1.1:
                    risk_score += 1

            if risk_score >= 5:
                # High WM risk: enforce conservative TM behavior.
                safe_tm_call('distance_to_leading_vehicle', world.player, 5.0)
                safe_tm_call('vehicle_percentage_speed_difference', world.player, max(30.0, base_speed_diff))
                safe_tm_call('random_left_lanechange_percentage', world.player, 0.0)
                safe_tm_call('random_right_lanechange_percentage', world.player, 0.0)
                safe_tm_call('ignore_lights_percentage', world.player, 0.0)
                safe_tm_call('ignore_signs_percentage', world.player, 0.0)
                safe_tm_call('ignore_vehicles_percentage', world.player, 0.0)
                safe_tm_call('ignore_walkers_percentage', world.player, 0.0)
                safe_tm_call('auto_lane_change', world.player, False)
            elif risk_score >= 3:
                # Medium WM risk: reduce aggression and shrink lane-change impulses.
                safe_tm_call('distance_to_leading_vehicle', world.player, 3.0)
                safe_tm_call('vehicle_percentage_speed_difference', world.player, max(15.0, base_speed_diff))
                safe_tm_call('random_left_lanechange_percentage', world.player, min(base_left_pct, 8.0))
                safe_tm_call('random_right_lanechange_percentage', world.player, min(base_right_pct, 8.0))
                safe_tm_call('ignore_lights_percentage', world.player, 0.0)
                safe_tm_call('ignore_signs_percentage', world.player, 0.0)
                safe_tm_call('ignore_vehicles_percentage', world.player, 0.0)
                safe_tm_call('ignore_walkers_percentage', world.player, 0.0)
                safe_tm_call('auto_lane_change', world.player, True)
            else:
                # Low WM risk: use action-driven baseline TM hint.
                safe_tm_call('distance_to_leading_vehicle', world.player, 1.8)
                safe_tm_call('vehicle_percentage_speed_difference', world.player, base_speed_diff)
                safe_tm_call('random_left_lanechange_percentage', world.player, base_left_pct)
                safe_tm_call('random_right_lanechange_percentage', world.player, base_right_pct)
                safe_tm_call('ignore_lights_percentage', world.player, 0.0)
                safe_tm_call('ignore_signs_percentage', world.player, 0.0)
                safe_tm_call('ignore_vehicles_percentage', world.player, 0.0)
                safe_tm_call('ignore_walkers_percentage', world.player, 0.0)
                safe_tm_call('auto_lane_change', world.player, True)

        if use_tm_control:
            try:
                with open(tm_event_path, 'w', encoding='utf-8') as f:
                    f.write('')
            except Exception as exc:
                logging.warning("TM event log init failed: %s", exc)
            write_tm_event("run_start", "unknown")

        _reset_route_metric_state()
        start_route_artifact_recording(finished_routes + 1)

        clock = pygame.time.Clock()

        # simulation starts
        while True:
            start = time.time()
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            global last_action, waypoints_number, world_tick, run_times, first_collision

            world.tick(clock)
            world_tick = world_tick + 1
            route_ticks += 1
            world.render(display)
            pygame.display.flip()

            ego_speed_kmh = world.player.get_velocity().length() * 3.6
            legal_red_light_hold = False
            tl_state_name = "none"
            try:
                if world.player.is_at_traffic_light():
                    tl = world.player.get_traffic_light()
                    if tl is not None:
                        tl_state_name = str(tl.get_state()).split('.')[-1].lower()
                        legal_red_light_hold = (tl.get_state() == carla.TrafficLightState.Red)
            except Exception:
                legal_red_light_hold = False
                tl_state_name = "unknown"
            is_red_light_now = (tl_state_name == "red")
            if legal_red_light_hold and ego_speed_kmh < args.stuck_speed_kmh:
                red_light_hold_ticks += 1
                if red_light_hold_ticks >= max(1, int(args.tm_post_red_release_min_hold_ticks)):
                    tm_red_wait_escape_armed = True
            else:
                red_light_hold_ticks = 0
            if red_light_hold_ticks > args.max_red_light_hold_ticks:
                legal_red_light_hold = False
            if hasattr(agent, "_last_rule_red_light") and bool(agent._last_rule_red_light):
                legal_red_light_hold = True
            if (not is_red_light_now) and was_red_light_tick and tm_red_wait_escape_armed:
                tm_post_red_release_countdown_ticks = max(
                    tm_post_red_release_countdown_ticks,
                    max(0, int(args.tm_post_red_release_countdown_ticks)),
                )
                tm_red_wait_escape_armed = False
                tm_no_progress_ticks = 0
                if tm_post_red_release_countdown_ticks > 0:
                    write_tm_event(
                        "tm_post_red_release_countdown_start",
                        tl_state_name,
                        {"countdown_ticks": int(tm_post_red_release_countdown_ticks)},
                    )
            elif is_red_light_now and tm_post_red_release_countdown_ticks > 0:
                tm_post_red_release_countdown_ticks = 0

            if ego_speed_kmh < args.stuck_speed_kmh and (not legal_red_light_hold):
                stuck_ticks += 1
            elif legal_red_light_hold:
                # Freeze/decrease stuck counter during legal red-light waiting.
                stuck_ticks = max(0, stuck_ticks - 2)
            else:
                stuck_ticks = 0

            metric_total_ticks += 1

            red_violation_now = bool(is_red_light_now and ego_speed_kmh > red_light_violation_speed_kmh)
            if red_violation_now:
                red_light_violation_active_ticks += 1
            else:
                red_light_violation_active_ticks = 0
                red_light_violation_latch = False
            if (
                red_light_violation_active_ticks >= red_light_violation_min_ticks
                and (not red_light_violation_latch)
            ):
                red_light_violation_events += 1
                route_had_red_light_violation = True
                red_light_violation_latch = True

            stop_sign = _nearest_stop_sign_ahead()
            if stop_sign is not None:
                sid = int(stop_sign.id)
                if active_stop_sign_id != sid:
                    _finalize_active_stop_sign()
                    active_stop_sign_id = sid
                    active_stop_sign_min_speed_kmh = float(ego_speed_kmh)
                    active_stop_sign_seen_ticks = 1
                else:
                    active_stop_sign_seen_ticks += 1
                    active_stop_sign_min_speed_kmh = min(active_stop_sign_min_speed_kmh, float(ego_speed_kmh))
            else:
                _finalize_active_stop_sign()
            was_red_light_tick = is_red_light_now

            offroad_dist = _distance_to_driving_lane_center()
            offroad_now = bool(offroad_dist > offroad_dist_threshold_m)
            if offroad_now:
                offroad_ticks += 1
            if offroad_now and not offroad_latch:
                offroad_events += 1
                route_had_offroad = True
            offroad_latch = offroad_now

            ttc_now = _estimate_front_ttc()
            if np.isfinite(ttc_now):
                route_min_ttc = min(route_min_ttc, float(ttc_now))
            near_miss_now = bool(np.isfinite(ttc_now) and ttc_now < near_miss_ttc_threshold_s)
            if near_miss_now:
                near_miss_ticks += 1
            if near_miss_now and not near_miss_latch:
                near_miss_events += 1
                route_had_near_miss = True
            near_miss_latch = near_miss_now

            ego_tf = world.player.get_transform()
            ego_v = world.player.get_velocity()
            ego_forward = ego_tf.get_forward_vector()
            ego_forward_speed_mps = float(ego_v.x * ego_forward.x + ego_v.y * ego_forward.y + ego_v.z * ego_forward.z)
            if prev_forward_speed_mps is not None:
                lon_acc_mps2 = (ego_forward_speed_mps - prev_forward_speed_mps) / max(1e-3, metric_dt)
                hard_brake_now = bool(lon_acc_mps2 < -hard_brake_acc_threshold_mps2)
                if hard_brake_now and not hard_brake_latch:
                    hard_brake_events += 1
                hard_brake_latch = hard_brake_now

                if prev_lon_acc_mps2 is not None:
                    jerk_mps3 = (lon_acc_mps2 - prev_lon_acc_mps2) / max(1e-3, metric_dt)
                    abs_jerk_mps3 = abs(jerk_mps3)
                    jerk_abs_sum += abs_jerk_mps3
                    jerk_samples += 1
                    jerk_now = bool(abs_jerk_mps3 > jerk_threshold_mps3)
                    if jerk_now and not jerk_latch:
                        jerk_events += 1
                    jerk_latch = jerk_now
                prev_lon_acc_mps2 = lon_acc_mps2
            prev_forward_speed_mps = ego_forward_speed_mps

            # Route progress for the current point-to-point task.
            if len(current_route_points) >= 2:
                ego_loc = world.player.get_location()
                closest_idx = min(range(len(current_route_points)), key=lambda i: ego_loc.distance(current_route_points[i]))
                goal_distance_now = ego_loc.distance(destination)
                if goal_distance_now + 1e-3 < best_goal_distance:
                    best_goal_distance = goal_distance_now
                    tm_no_progress_ticks = 0
                    if tm_post_red_release_countdown_ticks > 0:
                        tm_post_red_release_countdown_ticks = 0
                        write_tm_event(
                            "tm_post_red_release_countdown_end",
                            tl_state_name,
                            {"end_reason": "progress"},
                        )
                elif use_tm_control and tl_state_name != "red":
                    tm_no_progress_ticks += 1
                    if tm_post_red_release_countdown_ticks > 0:
                        tm_post_red_release_countdown_ticks -= 1
                        if tm_post_red_release_countdown_ticks == 0:
                            write_tm_event(
                                "tm_post_red_release_countdown_end",
                                tl_state_name,
                                {"end_reason": "countdown_elapsed"},
                            )

                # Use distance-to-goal reduction as route completion ratio.
                progress_ratio = 1.0 - (best_goal_distance / max(1.0, initial_goal_distance))
                progress_ratio = max(0.0, min(1.0, progress_ratio))
                if progress_ratio > current_route_best_ratio:
                    if use_tm_control:
                        # Keep waypoint count aligned with best route completion instead of
                        # instantaneous nearest-point jumps when the vehicle drifts off-route.
                        waypoints_number = max(int(waypoints_number), int(closest_idx))
                    current_route_best_ratio = progress_ratio
                if world_tick % max(1, args.progress_interval_ticks) == 0:
                    live = {
                        "route_index": finished_routes + 1,
                        "total_routes": total_routes,
                        "current_route_progress_percent": round(current_route_best_ratio * 100.0, 2),
                        "finished_routes": finished_routes,
                        "success_routes": success_routes,
                        "collision_routes": collision_routes,
                        "timeout_routes": timeout_routes,
                    }
                    print(
                        f"[LIVE] route {finished_routes + 1}/{total_routes} "
                        f"progress={current_route_best_ratio * 100.0:.2f}% "
                        f"success={success_routes} collision={collision_routes} timeout={timeout_routes}",
                        flush=True,
                    )
                    with open(route_live_path, 'w', encoding='utf-8') as f:
                        json.dump(live, f, ensure_ascii=False, indent=2)
                    write_tm_event("snapshot", tl_state_name)
                if (
                    use_tm_control
                    and tm_replans < args.tm_max_replans_per_route
                    and tm_no_progress_ticks >= args.tm_replan_stagnation_ticks
                    and tm_post_red_release_countdown_ticks <= 0
                    and route_ticks < max(200, args.max_route_ticks - 200)
                    and tl_state_name != "red"
                    and goal_distance_now > args.tm_near_goal_distance
                    and tm_wm_hint_missing_ticks <= max(0, int(args.tm_wm_hint_grace_ticks))
                ):
                    cur_wp = world.map.get_waypoint(
                        world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    end_wp = world.map.get_waypoint(
                        destination, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if cur_wp is not None and end_wp is not None:
                        route = agent.trace_route(cur_wp, end_wp)
                        path = [wp.transform.location for wp, _ in route]
                        if len(path) < 2:
                            path = [destination]
                        try:
                            traffic_manager.set_path(world.player, path)
                            traffic_manager.auto_lane_change(world.player, True)
                            tm_replans += 1
                            tm_no_progress_ticks = 0
                            print(
                                f"[TM-REPLAN] route={finished_routes + 1} "
                                f"replan={tm_replans} progress={current_route_best_ratio * 100.0:.2f}%",
                                flush=True,
                            )
                            write_tm_event("tm_replan", tl_state_name)
                        except RuntimeError as exc:
                            logging.warning("TM replan set_path failed: %s", exc)

            if use_tm_control:
                goal_distance_now = world.player.get_location().distance(destination)
                if tm_loop_escape_lock_ticks > 0:
                    tm_loop_escape_lock_ticks -= 1
                    if tm_loop_escape_lock_ticks == 0:
                        safe_tm_call('auto_lane_change', world.player, True)
                        write_tm_event("tm_loop_escape_unlock", tl_state_name)
                drift_from_best = max(0.0, goal_distance_now - best_goal_distance)
                if (
                    route_ticks > 200
                    and tm_no_progress_ticks >= args.tm_loop_escape_min_no_progress_ticks
                    and tm_post_red_release_countdown_ticks <= 0
                    and drift_from_best >= args.tm_loop_escape_distance_m
                    and tl_state_name != "red"
                    and (world_tick - tm_last_loop_escape_tick) >= args.tm_loop_escape_cooldown_ticks
                    and tm_wm_hint_missing_ticks <= max(0, int(args.tm_wm_hint_grace_ticks))
                ):
                    cur_wp = world.map.get_waypoint(
                        world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    end_wp = world.map.get_waypoint(
                        destination, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if cur_wp is not None and end_wp is not None:
                        route = agent.trace_route(cur_wp, end_wp)
                        path = [wp.transform.location for wp, _ in route]
                        if len(path) < 2:
                            path = [destination]
                        try:
                            no_progress_before_escape = int(tm_no_progress_ticks)
                            traffic_manager.set_path(world.player, path)
                            safe_tm_call('auto_lane_change', world.player, False)
                            tm_loop_escape_used += 1
                            tm_loop_escape_lock_ticks = max(0, int(args.tm_loop_escape_lock_ticks))
                            tm_last_loop_escape_tick = world_tick
                            tm_no_progress_ticks = max(0, tm_no_progress_ticks // 3)
                            print(
                                f"[TM-LOOP-ESCAPE] route={finished_routes + 1} "
                                f"drift={drift_from_best:.2f}m lock={tm_loop_escape_lock_ticks}",
                                flush=True,
                            )
                            write_tm_event(
                                "tm_loop_escape_replan",
                                tl_state_name,
                                {
                                    "drift_m": round(float(drift_from_best), 2),
                                    "lock_ticks": int(tm_loop_escape_lock_ticks),
                                    "loop_escape_used": int(tm_loop_escape_used),
                                },
                            )
                            if (
                                tm_loop_escape_used >= args.tm_force_lane_change_from_loop_escape
                                and no_progress_before_escape >= max(120, args.tm_force_lane_change_min_no_progress_ticks // 2)
                                and (world_tick - tm_last_force_lane_change_tick) >= args.tm_force_lane_change_cooldown_ticks
                            ):
                                force_right = ((finished_routes + tm_loop_escape_used + tm_unblock_used) % 2 == 0)
                                if safe_tm_call('force_lane_change', world.player, force_right):
                                    tm_last_force_lane_change_tick = world_tick
                                    direction = "right" if force_right else "left"
                                    print(
                                        f"[TM-FORCE-LC] route={finished_routes + 1} "
                                        f"source=loop_escape count={tm_loop_escape_used} direction={direction}",
                                        flush=True,
                                    )
                                    write_tm_event(
                                        "tm_force_lane_change",
                                        tl_state_name,
                                        {
                                            "direction": direction,
                                            "source": "loop_escape",
                                            "loop_escape_used": int(tm_loop_escape_used),
                                            "attempt": int(tm_unblock_used),
                                        },
                                    )
                        except RuntimeError as exc:
                            logging.warning("TM loop-escape replan failed: %s", exc)
                near_goal_stall = goal_distance_now <= args.tm_near_goal_distance
                reserve_for_near_goal = max(0, args.tm_unblock_reserve_near_goal)
                low_progress_unblock_budget = max(0, args.tm_unblock_max_per_route - reserve_for_near_goal)
                unblock_budget_available = (
                    tm_unblock_used < args.tm_unblock_max_per_route
                    and (near_goal_stall or tm_unblock_used < low_progress_unblock_budget)
                )
                if tm_unblock_active:
                    tm_unblock_ticks_left -= 1
                    if tm_wm_hint_missing_ticks > max(0, int(args.tm_wm_hint_grace_ticks)):
                        set_tm_unblock_mode(False)
                        tm_unblock_active = False
                        write_tm_event("tm_unblock_end", tl_state_name, {"end_reason": "wm_hint_missing"})
                    elif tm_unblock_ticks_left <= 0 or tm_no_progress_ticks == 0:
                        end_reason = "duration" if tm_unblock_ticks_left <= 0 else "progress"
                        set_tm_unblock_mode(False)
                        tm_unblock_active = False
                        write_tm_event("tm_unblock_end", tl_state_name, {"end_reason": end_reason})
                elif (
                    unblock_budget_available
                    and route_ticks > 200
                    and tm_no_progress_ticks >= args.tm_unblock_trigger_ticks
                    and tm_post_red_release_countdown_ticks <= 0
                    and ego_speed_kmh <= args.tm_unblock_speed_kmh
                    and tl_state_name != "red"
                    and tm_wm_hint_missing_ticks <= max(0, int(args.tm_wm_hint_grace_ticks))
                ):
                    start_stall = (
                        (not near_goal_stall)
                        and current_route_best_ratio < 0.03
                        and route_ticks <= max(1200, args.tm_unblock_trigger_ticks + 200)
                    )
                    next_unblock_attempt = tm_unblock_used + 1
                    set_tm_unblock_mode(
                        True,
                        high_progress=near_goal_stall,
                        stalled_start=start_stall,
                        unblock_attempt=next_unblock_attempt,
                    )
                    tm_unblock_active = True
                    tm_unblock_ticks_left = args.tm_unblock_duration_ticks
                    tm_unblock_used = next_unblock_attempt
                    if start_stall:
                        phase = "start_stall"
                    else:
                        phase = "near_goal" if near_goal_stall else "mid_route"
                    print(
                        f"[TM-UNBLOCK] route={finished_routes + 1} "
                        f"used={tm_unblock_used} phase={phase} "
                        f"progress={current_route_best_ratio * 100.0:.2f}%",
                        flush=True,
                    )
                    write_tm_event("tm_unblock_start", tl_state_name, {"phase": phase})
                    if (
                        tm_unblock_used >= args.tm_force_lane_change_from_unblock
                        and tm_no_progress_ticks >= args.tm_force_lane_change_min_no_progress_ticks
                        and (world_tick - tm_last_force_lane_change_tick) >= args.tm_force_lane_change_cooldown_ticks
                    ):
                        force_right = ((finished_routes + tm_unblock_used) % 2 == 0)
                        if safe_tm_call('force_lane_change', world.player, force_right):
                            tm_last_force_lane_change_tick = world_tick
                            direction = "right" if force_right else "left"
                            print(
                                f"[TM-FORCE-LC] route={finished_routes + 1} "
                                f"attempt={tm_unblock_used} direction={direction}",
                                flush=True,
                            )
                            write_tm_event(
                                "tm_force_lane_change",
                                tl_state_name,
                                {"direction": direction, "source": "unblock", "attempt": int(tm_unblock_used)},
                            )
                if (
                    args.tm_red_hold_soft_reset_ticks > 0
                    and tm_soft_resets < args.tm_max_soft_resets_per_route
                    and red_light_hold_ticks >= args.tm_red_hold_soft_reset_ticks
                    and route_ticks > 200
                    and route_ticks < max(200, args.max_route_ticks - 200)
                    and tl_state_name == "red"
                    and tm_loop_escape_used >= 1
                    and tm_replans >= args.tm_max_replans_per_route
                    and tm_wm_hint_missing_ticks <= max(0, int(args.tm_wm_hint_grace_ticks))
                ):
                    if try_tm_soft_reset(destination):
                        tm_soft_resets += 1
                        tm_no_progress_ticks = 0
                        tm_replans = 0
                        tm_unblock_active = False
                        tm_unblock_ticks_left = 0
                        red_light_hold_ticks = 0
                        tm_loop_escape_used = 0
                        tm_loop_escape_lock_ticks = 0
                        tm_last_loop_escape_tick = -1000000
                        tm_last_force_lane_change_tick = -1000000
                        tm_post_red_release_countdown_ticks = 0
                        tm_red_wait_escape_armed = False
                        was_red_light_tick = False
                        print(
                            f"[TM-SOFT-RESET-RED] route={finished_routes + 1} "
                            f"hold={args.tm_red_hold_soft_reset_ticks} "
                            f"progress={current_route_best_ratio * 100.0:.2f}%",
                            flush=True,
                        )
                        write_tm_event(
                            "tm_soft_reset_red_hold",
                            tl_state_name,
                            {"trigger_red_hold_ticks": int(args.tm_red_hold_soft_reset_ticks)},
                        )
                if (
                    tm_soft_resets < args.tm_max_soft_resets_per_route
                    and route_ticks > 200
                    and tm_no_progress_ticks >= args.tm_soft_reset_trigger_ticks
                    and tm_post_red_release_countdown_ticks <= 0
                    and current_route_best_ratio <= args.tm_soft_reset_max_progress
                    and route_ticks < max(200, args.max_route_ticks - 200)
                    and tl_state_name != "red"
                    and tm_wm_hint_missing_ticks <= max(0, int(args.tm_wm_hint_grace_ticks))
                ):
                    if try_tm_soft_reset(destination):
                        tm_soft_resets += 1
                        tm_no_progress_ticks = 0
                        tm_replans = 0
                        tm_unblock_active = False
                        tm_unblock_ticks_left = 0
                        tm_loop_escape_used = 0
                        tm_loop_escape_lock_ticks = 0
                        tm_last_loop_escape_tick = -1000000
                        tm_last_force_lane_change_tick = -1000000
                        tm_post_red_release_countdown_ticks = 0
                        tm_red_wait_escape_armed = False
                        was_red_light_tick = False
                        print(
                            f"[TM-SOFT-RESET] route={finished_routes + 1} "
                            f"used={tm_soft_resets} progress={current_route_best_ratio * 100.0:.2f}% "
                            f"goal_distance={goal_distance_now:.2f}m",
                            flush=True,
                        )
                        write_tm_event("tm_soft_reset", tl_state_name)

            if (
                args.allow_progress_success
                and use_tm_control
                and current_route_best_ratio >= args.tm_progress_success_threshold
                and waypoints_number >= args.tm_progress_success_min_waypoints
                and tm_no_progress_ticks >= args.tm_progress_stable_ticks
            ):
                finished_routes += 1
                success_routes += 1
                run_times += 1
                sum_route_completion += current_route_best_ratio
                _flush_route_metrics()
                with open(times_path, 'a') as file:
                    rec = (
                        "route: " + str(finished_routes)
                        + " result: reached_by_tm_progress_stable"
                        + " time ticks: " + str(world_tick)
                        + " waypoints: " + str(waypoints_number)
                        + " route_completion_percent: " + f"{current_route_best_ratio * 100.0:.2f}" + "\n"
                    )
                    file.write(rec)
                print(f"Route {finished_routes}/{total_routes}: reached_by_tm_progress_stable")
                dump_route_keyframes("reached_by_tm_progress_stable")

                if finished_routes >= total_routes:
                    break

                world.collision_sensor.collided = False
                agent.clear_waypoint_number()
                world_tick = 0
                waypoints_number = 0
                last_action = ""
                first_collision = True
                route_ticks = 0
                stuck_ticks = 0
                red_light_hold_ticks = 0
                tm_no_progress_ticks = 0
                tm_replans = 0
                tm_unblock_active = False
                tm_unblock_ticks_left = 0
                tm_unblock_used = 0
                tm_soft_resets = 0
                tm_loop_escape_used = 0
                tm_loop_escape_lock_ticks = 0
                tm_last_loop_escape_tick = -1000000
                tm_last_force_lane_change_tick = -1000000
                tm_post_red_release_countdown_ticks = 0
                tm_red_wait_escape_armed = False
                was_red_light_tick = False
                reset_wm_state_if_supported()
                if not relocate_ego_for_new_route():
                    logging.warning("Route reset: ego relocation failed, keeping current pose.")
                destination = pick_destination()
                set_route_destination(destination)
                set_tm_unblock_mode(False)
                current_route_points = build_route_points(destination)
                initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                current_route_best_ratio = 0.0
                _reset_route_metric_state()
                start_route_artifact_recording(finished_routes + 1)
                continue

            timed_out = route_ticks >= args.max_route_ticks
            stuck_timeout = stuck_ticks >= args.max_stuck_ticks
            if timed_out or stuck_timeout:
                # Optional progress-based fallback (disabled by default):
                # keep for ablation only, not for strict completion metric.
                if args.allow_progress_success and current_route_best_ratio >= 0.995 and waypoints_number >= 5:
                    finished_routes += 1
                    success_routes += 1
                    run_times += 1
                    current_route_best_ratio = 1.0
                    sum_route_completion += current_route_best_ratio
                    _flush_route_metrics()
                    with open(times_path, 'a') as file:
                        rec = (
                            "route: " + str(finished_routes)
                            + " result: reached_by_progress_fallback"
                            + " time ticks: " + str(world_tick)
                            + " waypoints: " + str(waypoints_number)
                            + " route_completion_percent: 100.00\n"
                        )
                        file.write(rec)
                    print(f"Route {finished_routes}/{total_routes}: reached_by_progress_fallback")
                    dump_route_keyframes("reached_by_progress_fallback")

                    if finished_routes >= total_routes:
                        break

                    world.collision_sensor.collided = False
                    agent.clear_waypoint_number()
                    world_tick = 0
                    waypoints_number = 0
                    last_action = ""
                    first_collision = True
                    route_ticks = 0
                    stuck_ticks = 0
                    red_light_hold_ticks = 0
                    tm_no_progress_ticks = 0
                    tm_replans = 0
                    tm_unblock_active = False
                    tm_unblock_ticks_left = 0
                    tm_unblock_used = 0
                    tm_soft_resets = 0
                    tm_loop_escape_used = 0
                    tm_loop_escape_lock_ticks = 0
                    tm_last_loop_escape_tick = -1000000
                    tm_last_force_lane_change_tick = -1000000
                    tm_post_red_release_countdown_ticks = 0
                    tm_red_wait_escape_armed = False
                    was_red_light_tick = False
                    reset_wm_state_if_supported()
                    if not relocate_ego_for_new_route():
                        logging.warning("Route reset: ego relocation failed, keeping current pose.")
                    destination = pick_destination()
                    set_route_destination(destination)
                    set_tm_unblock_mode(False)
                    current_route_points = build_route_points(destination)
                    initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                    current_route_best_ratio = 0.0
                    _reset_route_metric_state()
                    start_route_artifact_recording(finished_routes + 1)
                    continue

                if (
                    args.allow_progress_success
                    and use_tm_control
                    and current_route_best_ratio >= args.tm_progress_success_threshold
                    and waypoints_number >= args.tm_progress_success_min_waypoints
                ):
                    finished_routes += 1
                    success_routes += 1
                    run_times += 1
                    sum_route_completion += current_route_best_ratio
                    _flush_route_metrics()
                    with open(times_path, 'a') as file:
                        rec = (
                            "route: " + str(finished_routes)
                            + " result: reached_by_tm_progress_fallback"
                            + " time ticks: " + str(world_tick)
                            + " waypoints: " + str(waypoints_number)
                            + " route_completion_percent: " + f"{current_route_best_ratio * 100.0:.2f}" + "\n"
                        )
                        file.write(rec)
                    print(f"Route {finished_routes}/{total_routes}: reached_by_tm_progress_fallback")
                    dump_route_keyframes("reached_by_tm_progress_fallback")

                    if finished_routes >= total_routes:
                        break

                    world.collision_sensor.collided = False
                    agent.clear_waypoint_number()
                    world_tick = 0
                    waypoints_number = 0
                    last_action = ""
                    first_collision = True
                    route_ticks = 0
                    stuck_ticks = 0
                    red_light_hold_ticks = 0
                    tm_no_progress_ticks = 0
                    tm_replans = 0
                    tm_unblock_active = False
                    tm_unblock_ticks_left = 0
                    tm_unblock_used = 0
                    tm_soft_resets = 0
                    tm_loop_escape_used = 0
                    tm_loop_escape_lock_ticks = 0
                    tm_last_loop_escape_tick = -1000000
                    tm_last_force_lane_change_tick = -1000000
                    tm_post_red_release_countdown_ticks = 0
                    tm_red_wait_escape_armed = False
                    was_red_light_tick = False
                    reset_wm_state_if_supported()
                    if not relocate_ego_for_new_route():
                        logging.warning("Route reset: ego relocation failed, keeping current pose.")
                    destination = pick_destination()
                    set_route_destination(destination)
                    set_tm_unblock_mode(False)
                    current_route_points = build_route_points(destination)
                    initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                    current_route_best_ratio = 0.0
                    _reset_route_metric_state()
                    start_route_artifact_recording(finished_routes + 1)
                    continue

                goal_distance = world.player.get_location().distance(destination)
                is_high_progress_timeout = (
                    current_route_best_ratio >= args.tm_progress_success_threshold
                )
                if is_high_progress_timeout:
                    high_progress_timeout_routes += 1
                    high_progress_timeout_goal_distance_sum += goal_distance
                is_near_goal_timeout = goal_distance <= args.tm_near_goal_distance
                if is_near_goal_timeout:
                    near_goal_timeout_routes += 1
                    near_goal_timeout_goal_distance_sum += goal_distance

                finished_routes += 1
                timeout_routes += 1
                run_times += 1
                sum_route_completion += current_route_best_ratio
                if stuck_timeout:
                    stuck_timeout_routes += 1
                else:
                    route_timeout_only_routes += 1
                _flush_route_metrics()
                reason = "stuck_timeout" if stuck_timeout else "route_timeout"
                if is_near_goal_timeout:
                    reason = reason + "_near_goal"
                elif is_high_progress_timeout:
                    reason = reason + "_high_progress"
                with open(times_path, 'a') as file:
                    rec = (
                        "route: " + str(finished_routes)
                        + " result: " + reason
                        + " time ticks: " + str(world_tick)
                        + " waypoints: " + str(waypoints_number)
                        + " goal_distance_m: " + f"{goal_distance:.2f}"
                        + " route_completion_percent: " + f"{current_route_best_ratio * 100.0:.2f}" + "\n"
                    )
                    file.write(rec)
                print(f"Route {finished_routes}/{total_routes}: {reason}")
                write_tm_event(
                    reason,
                    tl_state_name,
                    {"goal_distance_m": round(goal_distance, 2)},
                    route_index_override=finished_routes,
                )
                dump_route_keyframes(reason)

                if finished_routes >= total_routes:
                    break

                world.collision_sensor.collided = False
                agent.clear_waypoint_number()
                world_tick = 0
                waypoints_number = 0
                last_action = ""
                first_collision = True
                route_ticks = 0
                stuck_ticks = 0
                red_light_hold_ticks = 0
                tm_no_progress_ticks = 0
                tm_replans = 0
                tm_unblock_active = False
                tm_unblock_ticks_left = 0
                tm_unblock_used = 0
                tm_soft_resets = 0
                tm_loop_escape_used = 0
                tm_loop_escape_lock_ticks = 0
                tm_last_loop_escape_tick = -1000000
                tm_last_force_lane_change_tick = -1000000
                tm_post_red_release_countdown_ticks = 0
                tm_red_wait_escape_armed = False
                was_red_light_tick = False
                reset_wm_state_if_supported()
                if not relocate_ego_for_new_route():
                    logging.warning("Route reset: ego relocation failed, keeping current pose.")
                destination = pick_destination()
                set_route_destination(destination)
                set_tm_unblock_mode(False)
                current_route_points = build_route_points(destination)
                initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                current_route_best_ratio = 0.0
                _reset_route_metric_state()
                start_route_artifact_recording(finished_routes + 1)
                continue

            if world.collision_sensor.collided:
                if route_ticks <= args.collision_grace_ticks:
                    world.collision_sensor.collided = False
                    world.collision_sensor.history.clear()
                    continue
                finished_routes += 1
                collision_routes += 1
                run_times += 1
                sum_route_completion += current_route_best_ratio
                _flush_route_metrics()
                with open(times_path, 'a') as file:
                    rec = (
                        "route: " + str(finished_routes)
                        + " result: collision"
                        + " time ticks: " + str(world_tick)
                        + " waypoints: " + str(waypoints_number)
                        + " route_completion_percent: " + f"{current_route_best_ratio * 100.0:.2f}" + "\n"
                    )
                    file.write(rec)
                print(f"Route {finished_routes}/{total_routes}: collision")
                write_tm_event("collision", tl_state_name, route_index_override=finished_routes)
                dump_route_keyframes("collision")

                if finished_routes >= total_routes:
                    break

                world.collision_sensor.collided = False
                agent.clear_waypoint_number()
                world_tick = 0
                waypoints_number = 0
                last_action = ""
                first_collision = True
                route_ticks = 0
                stuck_ticks = 0
                red_light_hold_ticks = 0
                tm_no_progress_ticks = 0
                tm_replans = 0
                tm_unblock_active = False
                tm_unblock_ticks_left = 0
                tm_unblock_used = 0
                tm_soft_resets = 0
                tm_loop_escape_used = 0
                tm_loop_escape_lock_ticks = 0
                tm_last_loop_escape_tick = -1000000
                tm_last_force_lane_change_tick = -1000000
                tm_post_red_release_countdown_ticks = 0
                tm_red_wait_escape_armed = False
                was_red_light_tick = False
                reset_wm_state_if_supported()
                if not relocate_ego_for_new_route():
                    logging.warning("Route reset: ego relocation failed, keeping current pose.")
                destination = pick_destination()
                set_route_destination(destination)
                set_tm_unblock_mode(False)
                current_route_points = build_route_points(destination)
                initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                current_route_best_ratio = 0.0
                _reset_route_metric_state()
                start_route_artifact_recording(finished_routes + 1)
                continue

            route_reached = False
            if use_tm_control:
                route_reached = world.player.get_location().distance(destination) <= args.tm_arrival_distance
            else:
                route_reached = agent.done()

            if route_reached:
                finished_routes += 1
                success_routes += 1
                run_times += 1
                current_route_best_ratio = 1.0
                sum_route_completion += current_route_best_ratio
                _flush_route_metrics()
                world.hud.notification("Target reached", seconds=2.0)
                print(f"Route {finished_routes}/{total_routes}: reached")
                with open(times_path, 'a') as file:
                    rec = (
                        "route: " + str(finished_routes)
                        + " result: reached"
                        + " time ticks: " + str(world_tick)
                        + " waypoints: " + str(waypoints_number)
                        + " route_completion_percent: 100.00\n"
                    )
                    file.write(rec)
                write_tm_event("reached", tl_state_name, route_index_override=finished_routes)
                dump_route_keyframes("reached")

                if finished_routes >= total_routes:
                    break

                agent.clear_waypoint_number()
                world_tick = 0
                waypoints_number = 0
                last_action = ""
                first_collision = True
                world.collision_sensor.collided = False
                route_ticks = 0
                stuck_ticks = 0
                red_light_hold_ticks = 0
                tm_no_progress_ticks = 0
                tm_replans = 0
                tm_unblock_active = False
                tm_unblock_ticks_left = 0
                tm_unblock_used = 0
                tm_soft_resets = 0
                tm_loop_escape_used = 0
                tm_loop_escape_lock_ticks = 0
                tm_last_loop_escape_tick = -1000000
                tm_last_force_lane_change_tick = -1000000
                tm_post_red_release_countdown_ticks = 0
                tm_red_wait_escape_armed = False
                was_red_light_tick = False
                reset_wm_state_if_supported()
                if not relocate_ego_for_new_route():
                    logging.warning("Route reset: ego relocation failed, keeping current pose.")
                destination = pick_destination()
                set_route_destination(destination)
                set_tm_unblock_mode(False)
                current_route_points = build_route_points(destination)
                initial_goal_distance, best_goal_distance = init_route_progress_state(destination)
                current_route_best_ratio = 0.0
                _reset_route_metric_state()
                start_route_artifact_recording(finished_routes + 1)
                continue
            if use_tm_control:
                tm_action = maybe_step_agent_in_tm()
                apply_tm_action_hint(tm_action)
                if tm_action:
                    last_action = f"tm_autopilot|{tm_action}"
                else:
                    last_action = "tm_autopilot"
            else:
                control = agent.run_step()
                agent.set_last_speed()
                agent.set_last_action()
                last_action = agent.get_last_action()
                waypoints_number = agent.get_waypoint_number()
                if control is not None:
                    control.manual_gear_shift = False
                    world.player.apply_control(control)
            end = time.time()
            # print(start-end)
            # print("this is one tick!!!")
            # time.sleep(14)

        completion_rate = (success_routes / float(total_routes)) if total_routes > 0 else 0.0
        avg_route_completion = (sum_route_completion / float(total_routes)) if total_routes > 0 else 0.0
        high_progress_timeout_rate = (high_progress_timeout_routes / float(total_routes)) if total_routes > 0 else 0.0
        stuck_timeout_rate = (stuck_timeout_routes / float(total_routes)) if total_routes > 0 else 0.0
        route_timeout_only_rate = (route_timeout_only_routes / float(total_routes)) if total_routes > 0 else 0.0
        avg_goal_distance_high_progress_timeout = (
            high_progress_timeout_goal_distance_sum / float(high_progress_timeout_routes)
            if high_progress_timeout_routes > 0
            else None
        )
        near_goal_timeout_rate = (near_goal_timeout_routes / float(total_routes)) if total_routes > 0 else 0.0
        avg_goal_distance_near_goal_timeout = (
            near_goal_timeout_goal_distance_sum / float(near_goal_timeout_routes)
            if near_goal_timeout_routes > 0
            else None
        )
        red_light_violation_route_rate = (red_light_violation_routes / float(total_routes)) if total_routes > 0 else 0.0
        stop_sign_violation_route_rate = (stop_sign_violation_routes / float(total_routes)) if total_routes > 0 else 0.0
        lane_invasion_route_rate = (lane_invasion_routes / float(total_routes)) if total_routes > 0 else 0.0
        solid_line_cross_route_rate = (solid_line_cross_routes / float(total_routes)) if total_routes > 0 else 0.0
        offroad_route_rate = (offroad_routes / float(total_routes)) if total_routes > 0 else 0.0
        near_miss_route_rate = (near_miss_routes / float(total_routes)) if total_routes > 0 else 0.0
        intervention_route_rate = (intervention_routes / float(total_routes)) if total_routes > 0 else 0.0
        offroad_tick_rate = (offroad_ticks / float(metric_total_ticks)) if metric_total_ticks > 0 else 0.0
        near_miss_tick_rate = (near_miss_ticks / float(metric_total_ticks)) if metric_total_ticks > 0 else 0.0
        hard_brake_event_rate_per_1k_ticks = (
            hard_brake_events * 1000.0 / float(metric_total_ticks)
        ) if metric_total_ticks > 0 else 0.0
        jerk_event_rate_per_1k_ticks = (
            jerk_events * 1000.0 / float(metric_total_ticks)
        ) if metric_total_ticks > 0 else 0.0
        mean_abs_jerk_mps3 = (jerk_abs_sum / float(jerk_samples)) if jerk_samples > 0 else None
        global_min_ttc_s = None if not np.isfinite(min_ttc_global) else float(min_ttc_global)
        avg_route_min_ttc_s = (
            route_min_ttc_sum / float(route_min_ttc_valid_count)
        ) if route_min_ttc_valid_count > 0 else None
        summary = {
            "total_routes": total_routes,
            "finished_routes": finished_routes,
            "success_routes": success_routes,
            "collision_routes": collision_routes,
            "timeout_routes": timeout_routes,
            "route_timeout_only_routes": route_timeout_only_routes,
            "route_timeout_only_rate": route_timeout_only_rate,
            "route_timeout_only_rate_percent": route_timeout_only_rate * 100.0,
            "stuck_timeout_routes": stuck_timeout_routes,
            "stuck_timeout_rate": stuck_timeout_rate,
            "stuck_timeout_rate_percent": stuck_timeout_rate * 100.0,
            "high_progress_timeout_routes": high_progress_timeout_routes,
            "high_progress_timeout_rate": high_progress_timeout_rate,
            "high_progress_timeout_rate_percent": high_progress_timeout_rate * 100.0,
            "avg_goal_distance_high_progress_timeout_m": avg_goal_distance_high_progress_timeout,
            "near_goal_timeout_routes": near_goal_timeout_routes,
            "near_goal_timeout_rate": near_goal_timeout_rate,
            "near_goal_timeout_rate_percent": near_goal_timeout_rate * 100.0,
            "avg_goal_distance_near_goal_timeout_m": avg_goal_distance_near_goal_timeout,
            "completion_rate": completion_rate,
            "completion_rate_percent": completion_rate * 100.0,
            "avg_route_completion": avg_route_completion,
            "avg_route_completion_percent": avg_route_completion * 100.0,
            "traffic_rule/red_light_violation_events": red_light_violation_events,
            "traffic_rule/red_light_violation_route_rate": red_light_violation_route_rate,
            "traffic_rule/red_light_violation_route_rate_percent": red_light_violation_route_rate * 100.0,
            "traffic_rule/stop_sign_violation_events": stop_sign_violation_events,
            "traffic_rule/stop_sign_violation_route_rate": stop_sign_violation_route_rate,
            "traffic_rule/stop_sign_violation_route_rate_percent": stop_sign_violation_route_rate * 100.0,
            "traffic_rule/lane_invasion_events": lane_invasion_events,
            "traffic_rule/lane_invasion_route_rate": lane_invasion_route_rate,
            "traffic_rule/lane_invasion_route_rate_percent": lane_invasion_route_rate * 100.0,
            "traffic_rule/solid_line_cross_events": solid_line_cross_events,
            "traffic_rule/solid_line_cross_route_rate": solid_line_cross_route_rate,
            "traffic_rule/solid_line_cross_route_rate_percent": solid_line_cross_route_rate * 100.0,
            "traffic_rule/offroad_events": offroad_events,
            "traffic_rule/offroad_ticks": offroad_ticks,
            "traffic_rule/offroad_tick_rate": offroad_tick_rate,
            "traffic_rule/offroad_tick_rate_percent": offroad_tick_rate * 100.0,
            "traffic_rule/offroad_route_rate": offroad_route_rate,
            "traffic_rule/offroad_route_rate_percent": offroad_route_rate * 100.0,
            "safety/min_ttc_global_s": global_min_ttc_s,
            "safety/min_ttc_route_avg_s": avg_route_min_ttc_s,
            "safety/near_miss_ttc_threshold_s": near_miss_ttc_threshold_s,
            "safety/near_miss_events": near_miss_events,
            "safety/near_miss_ticks": near_miss_ticks,
            "safety/near_miss_tick_rate": near_miss_tick_rate,
            "safety/near_miss_tick_rate_percent": near_miss_tick_rate * 100.0,
            "safety/near_miss_route_rate": near_miss_route_rate,
            "safety/near_miss_route_rate_percent": near_miss_route_rate * 100.0,
            "style/hard_brake_acc_threshold_mps2": hard_brake_acc_threshold_mps2,
            "style/hard_brake_events": hard_brake_events,
            "style/hard_brake_event_rate_per_1k_ticks": hard_brake_event_rate_per_1k_ticks,
            "style/jerk_threshold_mps3": jerk_threshold_mps3,
            "style/jerk_events": jerk_events,
            "style/jerk_event_rate_per_1k_ticks": jerk_event_rate_per_1k_ticks,
            "style/jerk_mean_abs_mps3": mean_abs_jerk_mps3,
            "safety_guard/intervention_events": intervention_events,
            "safety_guard/intervention_route_rate": intervention_route_rate,
            "safety_guard/intervention_route_rate_percent": intervention_route_rate * 100.0,
            "metric_total_ticks": metric_total_ticks,
        }
        print("Route evaluation summary:", summary)
        with open(os.path.join(run_path, 'route_eval_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(status_path, 'w', encoding='utf-8') as f:
            f.write('status: finished\n')
            f.write('end_time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n')
            f.write('completion_rate_percent: ' + f"{completion_rate * 100.0:.2f}" + '\n')
            f.write('avg_route_completion_percent: ' + f"{avg_route_completion * 100.0:.2f}" + '\n')
    except KeyboardInterrupt:
        with open(status_path, 'w', encoding='utf-8') as f:
            f.write('status: interrupted\n')
            f.write('end_time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n')
        raise
    except Exception:
        with open(status_path, 'w', encoding='utf-8') as f:
            f.write('status: failed\n')
            f.write('end_time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n')
        raise
    finally:

        if world is not None:
            try:
                if world.player is not None:
                    world.player.set_autopilot(False, args.tm_port)
            except Exception:
                pass
            try:
                settings = world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.world.apply_settings(settings)
            except Exception as exc:
                logging.warning("Cleanup apply_settings failed: %s", exc)

            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except Exception as exc:
                    logging.warning("Cleanup traffic_manager reset failed: %s", exc)

            try:
                world.destroy()
            except Exception as exc:
                logging.warning("Cleanup world.destroy failed: %s", exc)

        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status_text = f.read()
            if 'status: running' in status_text:
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write('status: finished\n')
                    f.write('end_time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n')

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--carla-timeout',
        type=float,
        default=180.0,
        help='CARLA client timeout in seconds (default: 180.0)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1600x900)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--log-root',
        default=None,
        help='Directory for baseline run logs')
    argparser.add_argument(
        '--openai-api-key',
        default=None,
        help='LLM API key. Prefer env var OPENAI_API_KEY to avoid exposing it in process args.')
    argparser.add_argument(
        '--openai-api-base',
        default=None,
        help='LLM API base URL. Defaults to OPENAI_API_BASE env or DeepSeek URL.')
    argparser.add_argument(
        '--num-routes',
        type=int,
        default=100,
        help='Number of point-to-point routes to evaluate (default: 100)')
    argparser.add_argument(
        '--progress-interval-ticks',
        type=int,
        default=200,
        help='Print/save live progress every N ticks (default: 200)')
    argparser.add_argument(
        '--max-route-ticks',
        type=int,
        default=4000,
        help='Mark a route as timeout if route ticks exceed this value (default: 4000)')
    argparser.add_argument(
        '--stuck-speed-kmh',
        type=float,
        default=0.3,
        help='Speed threshold for stuck detection in km/h (default: 0.3)')
    argparser.add_argument(
        '--max-stuck-ticks',
        type=int,
        default=600,
        help='Mark a route as stuck timeout after this many consecutive low-speed ticks (default: 600)')
    argparser.add_argument(
        '--max-red-light-hold-ticks',
        type=int,
        default=420,
        help='Do not exempt red-light waiting from stuck counting beyond this many ticks (default: 420)')
    argparser.add_argument(
        '--red-light-violation-speed-kmh',
        type=float,
        default=12.0,
        help='Count red-light violation only when speed is above this threshold in km/h (default: 12.0)')
    argparser.add_argument(
        '--red-light-violation-min-ticks',
        type=int,
        default=10,
        help='Require at least this many consecutive ticks above red-light violation speed to count one event (default: 10)')
    argparser.add_argument(
        '--collision-grace-ticks',
        type=int,
        default=5,
        help='Ignore collision callbacks within first N ticks after route reset (default: 5)')
    argparser.add_argument(
        '--ablate-wm',
        action='store_true',
        help='Ablate WM contribution while keeping LLM-in-loop active (default: off)')
    argparser.add_argument(
        '--tm-arrival-distance',
        type=float,
        default=8.0,
        help='Arrival distance in meters for tm control mode (default: 8.0)')
    argparser.add_argument(
        '--tm-near-goal-distance',
        type=float,
        default=25.0,
        help='Distance-to-goal threshold in meters for near-goal stall diagnostics and unblock reservation (default: 25.0)')
    argparser.add_argument(
        '--tm-replan-stagnation-ticks',
        type=int,
        default=500,
        help='In tm mode, replan path if progress does not improve for this many ticks (default: 500)')
    argparser.add_argument(
        '--tm-max-replans-per-route',
        type=int,
        default=0,
        help='Maximum path replans per route in tm mode (default: 0, disabled)')
    argparser.add_argument(
        '--tm-red-hold-soft-reset-ticks',
        type=int,
        default=0,
        help='In tm mode, trigger one soft-reset after this many consecutive red-light hold ticks if replans are exhausted (default: 0, disabled)')
    argparser.add_argument(
        '--tm-post-red-release-countdown-ticks',
        type=int,
        default=120,
        help='After a long red-light wait ends, defer TM escape actions for this many ticks (default: 120)')
    argparser.add_argument(
        '--tm-post-red-release-min-hold-ticks',
        type=int,
        default=80,
        help='Only arm post-red escape defer when red-light hold reaches this many low-speed ticks (default: 80)')
    argparser.add_argument(
        '--tm-progress-success-threshold',
        type=float,
        default=0.90,
        help='In tm mode, progress threshold for pseudo-success fallback when --allow-progress-success is enabled (default: 0.90)')
    argparser.add_argument(
        '--tm-progress-success-min-waypoints',
        type=int,
        default=30,
        help='Minimum progressed waypoints for pseudo-success fallback when --allow-progress-success is enabled (default: 30)')
    argparser.add_argument(
        '--tm-progress-stable-ticks',
        type=int,
        default=600,
        help='In tm mode, stable ticks for pseudo-success fallback when --allow-progress-success is enabled (default: 600)')
    argparser.add_argument(
        '--allow-progress-success',
        action='store_true',
        help='Allow progress-based pseudo-success in tm mode (default: disabled for strict metric)')
    argparser.add_argument(
        '--tm-unblock-trigger-ticks',
        type=int,
        default=700,
        help='Enable temporary tm unblock mode if no progress for this many ticks (default: 700)')
    argparser.add_argument(
        '--tm-unblock-duration-ticks',
        type=int,
        default=220,
        help='Duration of temporary tm unblock mode in ticks (default: 220)')
    argparser.add_argument(
        '--tm-unblock-max-per-route',
        type=int,
        default=2,
        help='Maximum unblock activations per route (default: 2)')
    argparser.add_argument(
        '--tm-unblock-reserve-near-goal',
        '--tm-unblock-reserve-high-progress',
        dest='tm_unblock_reserve_near_goal',
        type=int,
        default=1,
        help='Reserve this many unblock activations for near-goal stalls (default: 1)')
    argparser.add_argument(
        '--tm-unblock-speed-kmh',
        type=float,
        default=2.0,
        help='Only trigger tm unblock when ego speed is below this value (default: 2.0)')
    argparser.add_argument(
        '--tm-force-lane-change-from-unblock',
        type=int,
        default=2,
        help='Enable one-shot TM force_lane_change from this unblock attempt index (default: 2)')
    argparser.add_argument(
        '--tm-force-lane-change-min-no-progress-ticks',
        type=int,
        default=500,
        help='Require at least this many no-progress ticks before force_lane_change (default: 500)')
    argparser.add_argument(
        '--tm-force-lane-change-cooldown-ticks',
        type=int,
        default=500,
        help='Minimum interval between TM force_lane_change actions (default: 500)')
    argparser.add_argument(
        '--tm-force-lane-change-from-loop-escape',
        type=int,
        default=2,
        help='Enable one-shot TM force_lane_change after this many loop-escape replans in a route (default: 2)')
    argparser.add_argument(
        '--tm-loop-escape-distance-m',
        type=float,
        default=45.0,
        help='If goal distance drifts above best by this many meters, trigger corrective TM replan (default: 45.0)')
    argparser.add_argument(
        '--tm-loop-escape-min-no-progress-ticks',
        type=int,
        default=300,
        help='Require this many no-progress ticks before loop-escape corrective replan (default: 300)')
    argparser.add_argument(
        '--tm-loop-escape-lock-ticks',
        type=int,
        default=220,
        help='Ticks to temporarily disable auto lane change after loop-escape replan (default: 220)')
    argparser.add_argument(
        '--tm-loop-escape-cooldown-ticks',
        type=int,
        default=700,
        help='Minimum interval between loop-escape corrective replans (default: 700)')
    argparser.add_argument(
        '--tm-soft-reset-trigger-ticks',
        type=int,
        default=900,
        help='Trigger TM soft reset if no progress for this many ticks (default: 900)')
    argparser.add_argument(
        '--tm-max-soft-resets-per-route',
        type=int,
        default=1,
        help='Maximum TM soft reset attempts per route (default: 1)')
    argparser.add_argument(
        '--tm-soft-reset-brake-ticks',
        type=int,
        default=8,
        help='Brake settle ticks before TM soft reset path re-apply (default: 8)')
    argparser.add_argument(
        '--tm-soft-reset-max-progress',
        type=float,
        default=1.0,
        help='Only allow TM soft reset when best progress is at or below this ratio (default: 1.0)')
    argparser.add_argument(
        '--tm-agent-loop-interval-ticks',
        type=int,
        default=5,
        help='In tm mode, call agent.run_step every N ticks to keep LLM/WM decision loop active (default: 5)')
    argparser.add_argument(
        '--tm-action-hint-speed-up-diff',
        type=float,
        default=-12.0,
        help='TM speed diff when LLM Action is speed_up (default: -12.0)')
    argparser.add_argument(
        '--tm-action-hint-slow-down-diff',
        type=float,
        default=20.0,
        help='TM speed diff when LLM Action is speed_down/stop (default: 20.0)')
    argparser.add_argument(
        '--tm-action-hint-lane-change-pct',
        type=float,
        default=35.0,
        help='TM random lane-change percentage when LLM Action asks lane change (default: 35.0)')
    argparser.add_argument(
        '--tm-wm-hint-grace-ticks',
        type=int,
        default=20,
        help='In tm mode, WM hint may be missing for at most this many ticks before TM disables replan/unblock escapes (default: 20)')
    argparser.add_argument(
        '--tm-wm-nohint-speed-diff',
        type=float,
        default=55.0,
        help='In tm mode, speed diff used when WM hint is missing (default: 55.0)')
    argparser.add_argument(
        '--tm-wm-nohint-lead-distance',
        type=float,
        default=6.0,
        help='In tm mode, lead distance used when WM hint is missing (default: 6.0)')
    argparser.add_argument(
        '--tm-port',
        type=int,
        default=8000,
        help='Traffic manager port, fallback to +10/+20 if bind error occurs (default: 8000)')

    args = argparser.parse_args()
    args.loop = True
    if args.log_root is None:
        args.log_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'experiments',
            'wm',
        )
    args.log_root = os.path.abspath(args.log_root)
    os.makedirs(args.log_root, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    global run_path, collision_path, times_path, video_path, status_path, run_args_path, route_live_path, tm_event_path
    run_path = os.path.join(args.log_root, f'automatic_control_{timestamp}')
    os.makedirs(run_path, exist_ok=True)
    collision_path = os.path.join(run_path, 'collision.txt')
    times_path = os.path.join(run_path, 'times.txt')
    video_path = os.path.join(run_path, 'driving.mp4')
    status_path = os.path.join(run_path, 'status.txt')
    run_args_path = os.path.join(run_path, 'run_args.txt')
    route_live_path = os.path.join(run_path, 'route_eval_live.json')
    tm_event_path = os.path.join(run_path, 'tm_event_log.jsonl')
    args.log_root = run_path

    redacted_argv = []
    hide_next = False
    for token in sys.argv:
        if hide_next:
            redacted_argv.append('***REDACTED***')
            hide_next = False
            continue
        if token == '--openai-api-key':
            redacted_argv.append(token)
            hide_next = True
            continue
        if token.startswith('--openai-api-key='):
            redacted_argv.append('--openai-api-key=***REDACTED***')
            continue
        redacted_argv.append(token)

    with open(run_args_path, 'w', encoding='utf-8') as f:
        f.write('cmd: ' + ' '.join(redacted_argv) + '\n')
        f.write('host: ' + str(args.host) + '\n')
        f.write('carla_timeout: ' + str(args.carla_timeout) + '\n')
        f.write('port: ' + str(args.port) + '\n')
        f.write('agent: BehaviorAgentWM (forced)\n')
        f.write('behavior: cautious (forced)\n')
        f.write('sync: ' + str(args.sync) + '\n')
        f.write('num_routes: ' + str(args.num_routes) + '\n')
        f.write('progress_interval_ticks: ' + str(args.progress_interval_ticks) + '\n')
        f.write('max_route_ticks: ' + str(args.max_route_ticks) + '\n')
        f.write('stuck_speed_kmh: ' + str(args.stuck_speed_kmh) + '\n')
        f.write('max_stuck_ticks: ' + str(args.max_stuck_ticks) + '\n')
        f.write('max_red_light_hold_ticks: ' + str(args.max_red_light_hold_ticks) + '\n')
        f.write('red_light_violation_speed_kmh: ' + str(args.red_light_violation_speed_kmh) + '\n')
        f.write('red_light_violation_min_ticks: ' + str(args.red_light_violation_min_ticks) + '\n')
        f.write('collision_grace_ticks: ' + str(args.collision_grace_ticks) + '\n')
        f.write('control_mode: tm (forced)\n')
        f.write('ablate_wm: ' + str(args.ablate_wm) + '\n')
        f.write('openai_api_key_provided: ' + str(bool(args.openai_api_key)) + '\n')
        f.write('openai_api_base: ' + str(args.openai_api_base) + '\n')
        f.write('tm_arrival_distance: ' + str(args.tm_arrival_distance) + '\n')
        f.write('tm_near_goal_distance: ' + str(args.tm_near_goal_distance) + '\n')
        f.write('tm_replan_stagnation_ticks: ' + str(args.tm_replan_stagnation_ticks) + '\n')
        f.write('tm_max_replans_per_route: ' + str(args.tm_max_replans_per_route) + '\n')
        f.write('tm_red_hold_soft_reset_ticks: ' + str(args.tm_red_hold_soft_reset_ticks) + '\n')
        f.write('tm_post_red_release_countdown_ticks: ' + str(args.tm_post_red_release_countdown_ticks) + '\n')
        f.write('tm_post_red_release_min_hold_ticks: ' + str(args.tm_post_red_release_min_hold_ticks) + '\n')
        f.write('tm_progress_success_threshold: ' + str(args.tm_progress_success_threshold) + '\n')
        f.write('tm_progress_success_min_waypoints: ' + str(args.tm_progress_success_min_waypoints) + '\n')
        f.write('tm_progress_stable_ticks: ' + str(args.tm_progress_stable_ticks) + '\n')
        f.write('allow_progress_success: ' + str(args.allow_progress_success) + '\n')
        f.write('tm_unblock_trigger_ticks: ' + str(args.tm_unblock_trigger_ticks) + '\n')
        f.write('tm_unblock_duration_ticks: ' + str(args.tm_unblock_duration_ticks) + '\n')
        f.write('tm_unblock_max_per_route: ' + str(args.tm_unblock_max_per_route) + '\n')
        f.write('tm_unblock_reserve_near_goal: ' + str(args.tm_unblock_reserve_near_goal) + '\n')
        f.write('tm_unblock_speed_kmh: ' + str(args.tm_unblock_speed_kmh) + '\n')
        f.write('tm_force_lane_change_from_unblock: ' + str(args.tm_force_lane_change_from_unblock) + '\n')
        f.write('tm_force_lane_change_min_no_progress_ticks: ' + str(args.tm_force_lane_change_min_no_progress_ticks) + '\n')
        f.write('tm_force_lane_change_cooldown_ticks: ' + str(args.tm_force_lane_change_cooldown_ticks) + '\n')
        f.write('tm_force_lane_change_from_loop_escape: ' + str(args.tm_force_lane_change_from_loop_escape) + '\n')
        f.write('tm_loop_escape_distance_m: ' + str(args.tm_loop_escape_distance_m) + '\n')
        f.write('tm_loop_escape_min_no_progress_ticks: ' + str(args.tm_loop_escape_min_no_progress_ticks) + '\n')
        f.write('tm_loop_escape_lock_ticks: ' + str(args.tm_loop_escape_lock_ticks) + '\n')
        f.write('tm_loop_escape_cooldown_ticks: ' + str(args.tm_loop_escape_cooldown_ticks) + '\n')
        f.write('tm_soft_reset_trigger_ticks: ' + str(args.tm_soft_reset_trigger_ticks) + '\n')
        f.write('tm_max_soft_resets_per_route: ' + str(args.tm_max_soft_resets_per_route) + '\n')
        f.write('tm_soft_reset_brake_ticks: ' + str(args.tm_soft_reset_brake_ticks) + '\n')
        f.write('tm_soft_reset_max_progress: ' + str(args.tm_soft_reset_max_progress) + '\n')
        f.write('tm_agent_loop_interval_ticks: ' + str(args.tm_agent_loop_interval_ticks) + '\n')
        f.write('tm_action_hint_speed_up_diff: ' + str(args.tm_action_hint_speed_up_diff) + '\n')
        f.write('tm_action_hint_slow_down_diff: ' + str(args.tm_action_hint_slow_down_diff) + '\n')
        f.write('tm_action_hint_lane_change_pct: ' + str(args.tm_action_hint_lane_change_pct) + '\n')
        f.write('tm_wm_hint_grace_ticks: ' + str(args.tm_wm_hint_grace_ticks) + '\n')
        f.write('tm_wm_nohint_speed_diff: ' + str(args.tm_wm_nohint_speed_diff) + '\n')
        f.write('tm_wm_nohint_lead_distance: ' + str(args.tm_wm_nohint_lead_distance) + '\n')
        f.write('tm_port: ' + str(args.tm_port) + '\n')
        f.write('tm_event_log: ' + str(tm_event_path) + '\n')

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    logging.info('wm run dir: %s', run_path)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        global world_tick
        print("\nthis is the world tick:")
        print(world_tick)
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
