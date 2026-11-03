"""Spin and roam: turn a full circle sweeping, drive to the open place a full
turn would show the most new floor from, and spin again."""

import math

import numpy as np
import unittest
from pathlib import Path

from explore_sim import CORRIDOR, ROOM_A, ROOM_B, FloorPlan
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.exploration import SPIN_AWAY_CM, SPIN_CLEARANCE_CM, CoverageGrid
from robocar.localization import relative, wrap_degrees
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
        self.mapping._frontier_explore = self.frontier_step
        self.frontier_steps = 0

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

    def frontier_step(self, limits_reached, max_moves=None, until=None):
        """A frontier step: a sweep, then toward the nearest unexplored edge
        along the map's route, as far as the map shows open."""
        if limits_reached():
            return "limit"
        self.sweep_here()
        with self.mapping._map_lock:
            frontiers, _, _ = self.mapping._grid.frontiers(self.pose)
        if not frontiers:
            return "explored"
        target = min(frontiers, key=lambda frontier: frontier.path_cm).target
        with self.mapping._map_lock:
            leg = self.mapping._grid.route(self.pose, target)
        if leg is None:
            return "blocked"
        bearing, length = leg
        self.turn_to(self.pose[2] + bearing)
        with self.mapping._map_lock:
            open_run = self.mapping._grid.clear_run((*self.pose[:2], self.pose[2]), length)
        moved, bumped = self.floor.drive(self.pose, max(10.0, min(length, open_run - 45.0)))
        self.bumps += bumped
        self.place(moved)
        self.frontier_steps += 1
        return "limit"

    def run(self, strategy="spin"):
        self.mapping._enabled = True
        if strategy == "room":
            self.mapping._run_room_mapping()
        else:
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


class CircleFitTests(unittest.TestCase):
    """A station's full circle, fitted to the map from before it, puts a
    station the wheels and gyro placed a little wrong back where it is."""

    def setUp(self):
        self.floor = FloorPlan("lidar")
        self.drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        self.pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
        self.sonar = RangeSensorController(None, physical=False)
        self.mapping = MappingController(Camera(), self.drive, self.pan_tilt, self.sonar, MappingStore(Path(":memory:")))
        self.mapping._run_id = "circle-run"
        self.previous = None

    def tearDown(self):
        self.mapping.close()
        self.drive.close()
        self.pan_tilt.close()

    def stop(self, true_pose, believed, sigma):
        """A stop: swept from where the car really is, recorded where it thinks it is."""
        mapping = self.mapping
        scan = self.floor.scan(true_pose)
        node = mapping._store.save(b"jpeg", scan, "circle-run", "explore", believed, None)
        edge = None if self.previous is None else (self.previous[0], relative(self.previous[1], believed), sigma)
        mapping._places.add_stop(node, "circle-run", believed, scan, (2.0, 1.0), None, edge)
        with mapping._map_lock:
            mapping._grid.integrate(scan, believed, 2.0)
        with mapping._lock:
            mapping._last_node = node
        self.previous = (node, believed)
        return node, scan

    def test_a_drifted_station_is_put_back(self):
        # Two stations already mapped round, placed exactly.
        for spot in ((-100.0, -60.0), (60.0, 20.0)):
            for heading in (0.0, 90.0, 180.0, 270.0):
                self.stop((*spot, heading), (*spot, heading), (1.0, 1.0, 0.5))
        base = self.mapping._station_base()
        # Drove to a station; the estimate drifted 12 cm and 4 degrees on the way.
        true_spot = (-40.0, 40.0)
        drift = (12.0, -6.0, 4.0)
        sweeps = []
        for index, heading in enumerate((0.0, 90.0, 180.0, 270.0)):
            true_pose = (*true_spot, heading)
            believed = (true_spot[0] + drift[0], true_spot[1] + drift[1], wrap_degrees(heading + drift[2]))
            # The drive there is uncertain; the quarter turns on the spot are not.
            sigma = (10.0, 10.0, 4.0) if index == 0 else (1.0, 1.0, 0.5)
            sweeps.append(self.stop(true_pose, believed, sigma))
        with self.mapping._lock:
            self.mapping._pose = believed
            self.mapping._pose_covariance = np.diag([10.0**2, 10.0**2, np.radians(5.0) ** 2])
        fit = self.mapping._fit_circle(base, sweeps)
        self.assertTrue(fit["used"], fit)
        settled = self.mapping._pose
        self.assertLess(math.hypot(settled[0] - true_spot[0], settled[1] - true_spot[1]), 4.0, settled)
        self.assertLess(abs(wrap_degrees(settled[2] - 270.0)), 1.5, settled)
        # The whole station moved with it, and the link is kept.
        first = self.mapping._places.graph.nodes[sweeps[0][0]]
        self.assertLess(abs(wrap_degrees(first[2] - 0.0)), 1.5, first)
        self.assertTrue(any(link["kind"] == "circle" for link in self.mapping._store.links()))

    def test_nothing_to_fit_to_on_the_first_station(self):
        base = self.mapping._station_base()
        sweeps = [self.stop((0.0, 0.0, heading), (0.0, 0.0, heading), (1.0, 1.0, 0.5)) for heading in (0.0, 90.0, 180.0)]
        self.assertIsNone(self.mapping._fit_circle(base, sweeps))


class KnownRoomTests(unittest.TestCase):
    """The room as drawn on the dashboard: the open floor seen so far closed
    into a shape, growing as more is seen."""

    def test_the_room_grows_into_one_closed_shape(self):
        floor = FloorPlan("lidar")
        grid = CoverageGrid()
        self.assertEqual(grid.known_room(), [])
        for heading in (0.0, 90.0, 180.0, 270.0):
            grid.integrate(floor.scan((0.0, 0.0, heading)), (0.0, 0.0, heading), 2.0)
        first = grid.known_room()
        self.assertEqual(len(first), 1)
        for spot in ((-250.0, 80.0), (-520.0, 100.0)):
            for heading in (0.0, 90.0, 180.0, 270.0):
                grid.integrate(floor.scan((*spot, heading)), (*spot, heading), 2.0)
        whole = grid.known_room()
        self.assertEqual(len(whole), 1)

        def area(points):
            return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))) / 2

        # Both rooms and the corridor: about 7.5 + 2 + 8 square metres.
        self.assertGreater(area(whole[0]["outline"]), area(first[0]["outline"]) * 1.8)
        self.assertGreater(area(whole[0]["outline"]) / 10000.0, 15.0)
        # Inside the walls.
        xs = [x for x, _ in whole[0]["outline"]]
        self.assertGreater(min(xs), -660.0)
        self.assertLess(max(xs), 160.0)


class RoomMappingTests(unittest.TestCase):
    """Room mapping: weighing spinning across the room, spinning nearby and
    frontier steps, with a preference for crossing the room."""

    def test_it_maps_the_house_mostly_by_crossing_rooms_to_spin(self):
        for start in ((-600.0, 0.0, 0.0), (0.0, 100.0, 180.0)):
            sim = SimulatedSpin(start)
            try:
                sim.run("room")
                self.assertEqual(sim.mapping.status()["state"], "explored", start)
                self.assertEqual(sim.bumps, 0, start)
                for box in (ROOM_A, CORRIDOR, ROOM_B):
                    self.assertGreater(sim.floor.room_coverage(sim.mapping._grid, box), 0.9, (start, box))
                choices = [choice["chose"] for choice in sim.mapping._room_choices]
                self.assertIn("far_spin", choices, start)
                # Spinning does the work; frontier steps only pick up scraps.
                self.assertLessEqual(sim.frontier_steps, 5, start)
                self.assertLessEqual(sim.sweeps, 4 * len(sim.mapping._stations) + 12, start)
            finally:
                sim.close()


class RoomPlanTests(unittest.TestCase):
    def test_it_stops_when_the_best_left_is_scraps(self):
        from robocar.room_plan import MappingOption, MethodRecords, choose

        records = MethodRecords()
        good = MappingOption("spin", 900.0, 45.0, 20.0)
        scraps = MappingOption("frontier", 40.0, 20.0, 1.0)
        self.assertIs(choose([good], records, best_rate=30.0), good)
        self.assertIsNone(choose([scraps], records, best_rate=60.0))
        self.assertIs(choose([scraps], records, best_rate=10.0), scraps)
        self.assertIsNone(choose([], records))

    def test_a_method_that_underdelivers_loses_its_standing_and_it_is_kept(self):
        import json

        from robocar.room_plan import MethodRecords

        records = MethodRecords()
        for _ in range(8):
            records.note("frontier", 200.0, 10.0)
        self.assertLess(records.track["frontier"], 0.2)
        records.note_timing("spin_seconds", 60.0)
        again = MethodRecords.from_json(json.loads(json.dumps(records.to_json())))
        self.assertAlmostEqual(again.track["frontier"], records.track["frontier"], places=2)
        self.assertAlmostEqual(again.spin_seconds, records.spin_seconds, places=1)


if __name__ == "__main__":
    unittest.main()
