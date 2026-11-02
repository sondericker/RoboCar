"""Spin and roam: turn a full circle sweeping, drive to the open place a full
turn would show the most new floor from, and spin again."""

import math
import unittest
from pathlib import Path

from explore_sim import CORRIDOR, ROOM_A, ROOM_B, FloorPlan
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.exploration import SPIN_AWAY_CM, SPIN_CLEARANCE_CM, CoverageGrid
from robocar.mapping import MappingController
from robocar.store import MappingStore
from robocar.range_sensor import RangeSensorController


class Camera:
    def capture_snapshot(self):
        return b"jpeg"


class SimulatedSpin:
    """The controller's own spin-and-roam loop, with the hardware edges
    simulated: a sweep is a lidar sweep of the floor plan, a turn sets the
    heading, and a drive follows the map's route, stopping at walls."""

    def __init__(self, start, drive_works=True):
        self.floor = FloorPlan("lidar")
        self.drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        self.pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
        self.sonar = RangeSensorController(None, physical=False)
        self.mapping = MappingController(Camera(), self.drive, self.pan_tilt, self.sonar, MappingStore(Path(":memory:")))
        self.pose = start
        self.sweeps = self.bumps = 0
        self.drive_works = drive_works
        self.mapping._sweep_here = self.sweep_here
        self.mapping._turn_to_heading = self.turn_to
        self.mapping._drive_to_station = self.drive_to

    def close(self):
        self.mapping.close()
        self.drive.close()
        self.pan_tilt.close()

    def place(self, pose):
        self.pose = pose
        with self.mapping._lock:
            self.mapping._pose = pose
            self.mapping._last_odometry = None

    def sweep_here(self):
        scan = self.floor.scan(self.pose)
        with self.mapping._map_lock:
            self.mapping._grid.integrate(scan, self.pose, 2.0)
        self.place(self.pose)
        self.sweeps += 1
        return scan

    def turn_to(self, heading):
        self.place((self.pose[0], self.pose[1], heading))
        return True

    def drive_to(self, goal, started_at):
        if not self.drive_works:
            return "blocked"
        for _ in range(12):
            if math.hypot(goal[0] - self.pose[0], goal[1] - self.pose[1]) <= 25:
                return "arrived"
            with self.mapping._map_lock:
                leg = self.mapping._grid.route(self.pose, goal)
            if leg is None:
                return "blocked"
            bearing, length = leg
            self.turn_to(self.pose[2] + bearing)
            moved, bumped = self.floor.drive(self.pose, min(length, 60.0))
            self.bumps += bumped
            self.place(moved)
            if bumped:
                return "blocked"
        return "blocked"

    def run(self):
        self.mapping._enabled = True
        self.mapping._run_spin_explore()


class SpinAndRoamTests(unittest.TestCase):
    def test_it_maps_both_rooms_and_the_corridor_and_knows_when_it_is_done(self):
        for start in ((-100.0, -60.0, 45.0), (-600.0, 0.0, 0.0)):
            sim = SimulatedSpin(start)
            try:
                sim.run()
                self.assertEqual(sim.mapping.status()["state"], "explored", start)
                self.assertEqual(sim.bumps, 0, start)
                for box in (ROOM_A, CORRIDOR, ROOM_B):
                    self.assertGreater(sim.floor.room_coverage(sim.mapping._grid, box), 0.9, (start, box))
                # A full circle at each station: far fewer sweeps than stations x 4 would be wasteful.
                stations = sim.mapping._stations
                self.assertLessEqual(sim.sweeps, 4 * len(stations) + 12)
                for index, here in enumerate(stations):
                    for other in stations[:index]:
                        self.assertGreaterEqual(math.hypot(here[0] - other[0], here[1] - other[1]), SPIN_AWAY_CM - 25.0)
            finally:
                sim.close()

    def test_a_station_it_cannot_reach_is_given_up_without_spinning_again_where_it_stands(self):
        sim = SimulatedSpin((-100.0, -60.0, 45.0), drive_works=False)
        try:
            sim.run()
            self.assertEqual(len(sim.mapping._stations), 1)
            self.assertEqual(sim.mapping.status()["state"], "explored")
        finally:
            sim.close()

    def test_stations_are_open_reachable_and_new(self):
        floor = FloorPlan("lidar")
        grid = CoverageGrid()
        start = (0.0, -40.0, 0.0)
        for heading in (0.0, 90.0, 180.0, 270.0):
            grid.integrate(floor.scan((0.0, -40.0, heading)), (0.0, -40.0, heading), 2.0)
        station = grid.spin_station(start, avoid=[start[:2]])
        self.assertIsNotNone(station)
        self.assertGreaterEqual(floor.clearance(station.target), SPIN_CLEARANCE_CM - 6.0)
        self.assertGreaterEqual(math.hypot(station.target[0] - start[0], station.target[1] - start[1]), SPIN_AWAY_CM)
        self.assertTrue(math.isfinite(station.path_cm))
        # A map with nothing unknown left in reach has nowhere worth a spin.
        self.assertIsNone(CoverageGrid().spin_station(start))

    def test_it_will_not_turn_on_the_spot_beside_a_mapped_wall(self):
        floor = FloorPlan("lidar")
        grid = CoverageGrid()
        for x in (0.0, 60.0):
            for heading in (0.0, 90.0, 180.0, 270.0):
                grid.integrate(floor.scan((x, 0.0, heading)), (x, 0.0, heading), 2.0)
        self.assertTrue(grid.clear_all_round((0.0, 0.0, 0.0), 20.0))
        # Room A's right wall is at x = 150: 12 cm from a car standing at 138.
        self.assertFalse(grid.clear_all_round((138.0, 0.0, 0.0), 20.0))


if __name__ == "__main__":
    unittest.main()
