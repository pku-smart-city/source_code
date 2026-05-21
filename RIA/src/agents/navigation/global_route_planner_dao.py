# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Compatibility DAO for legacy CARLA GlobalRoutePlanner callers.

Some external evaluators/scenario_runner variants still construct:

    dao = GlobalRoutePlannerDAO(world_map, resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

This project primarily uses the newer planner signature
GlobalRoutePlanner(world_map, resolution), so this DAO intentionally provides
the minimal adapter surface expected by older call sites.
"""


class GlobalRoutePlannerDAO(object):
    """Legacy adapter around carla.Map."""

    def __init__(self, world_map, sampling_resolution=2.0):
        self._wmap = world_map
        self._sampling_resolution = float(sampling_resolution)

    def get_topology(self):
        return self._wmap.get_topology()

    def get_waypoint(self, location):
        return self._wmap.get_waypoint(location)

    def get_resolution(self):
        return self._sampling_resolution

