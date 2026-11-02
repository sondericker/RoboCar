"""The TF-Luna lidar: the driver, lidar sweeps on the map, and what the hunt
does with precise ranges."""

import copy
import math
import random
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from explore_sim import CORRIDOR, ROOM_A, ROOM_B, FloorPlan, explore
from hunt_sim import SimCat, SimHuntBody, build_map
from sonar_room import SonarRoom
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.exploration import CoverageGrid, ExplorationLearner
from robocar.hunt import MIN_CAT_DISTANCE_CM, HuntEngine, HuntMemory, _cat_motion, move_mouse
from robocar.mapping import (
    LIDAR_SCAN_ANGLES,
    MappingController,
    MappingStore,
    PointCloud,
    direction_clearances,
    forward_clearance,
    lidar_points,
    plan_explore_step,
)
from robocar.range_sensor import RangeSensorController, TFLunaRangeSensor, lidar_distance
from robocar.semantics import sightings_from
from robocar.surfaces import LIDAR, SONAR, SurfaceMap, extract_echoes, scan_model
from robocar.watch import MAX_GAIN, MOVING_CAP_US, MotionExposure


class FakeBus:
    def __init__(self, frame):
        self.frame = frame
        self.closed = False

    def read_i2c_block_data(self, address, register, length):
        assert (address, register, length) == (0x10, 0x00, 6)
        return list(self.frame)

    def close(self):
        self.closed = True


def tfluna_frame(distance, strength, temperature_c=42.0):
    centi = int(temperature_c * 100)
    return [distance & 0xFF, distance >> 8, strength & 0xFF, strength >> 8, centi & 0xFF, centi >> 8]


class DriverTests(unittest.TestCase):
    def test_weak_saturated_and_impossible_frames_are_not_trusted(self):
        self.assertEqual(lidar_distance(83, 4800), 83.0)
        self.assertIsNone(lidar_distance(83, 99))  # too weak
        self.assertIsNone(lidar_distance(83, 0xFFFF))  # saturated
        self.assertIsNone(lidar_distance(0, 4800))
        self.assertIsNone(lidar_distance(900, 4800))

    def test_reads_distance_strength_and_temperature_in_one_transaction(self):
        bus = FakeBus(tfluna_frame(318, 1816, 48.0))
        with patch.dict(sys.modules, {"smbus2": types.SimpleNamespace(SMBus=lambda number: bus)}):
            sensor = TFLunaRangeSensor()
        controller = RangeSensorController(sensor, True, sample_interval=10, name="Lidar")
        try:
            self.assertEqual(controller.sample_now(), 318.0)
            status = controller.status()
            self.assertEqual(status["signal"], 1816)
            self.assertEqual(status["temperature_c"], 48.0)
            bus.frame = tfluna_frame(318, 20)
            self.assertIsNone(controller.sample_now())
            self.assertEqual(controller.status()["error"], "No echo received")
        finally:
            controller.close()
        self.assertTrue(bus.closed)

    def test_a_missing_lidar_fails_at_start_up(self):
        class Silent(FakeBus):
            def read_i2c_block_data(self, *args):
                raise OSError(121, "Remote I/O error")

        bus = Silent(())
        with patch.dict(sys.modules, {"smbus2": types.SimpleNamespace(SMBus=lambda number: bus)}):
            with self.assertRaises(OSError):
                TFLunaRangeSensor()
        self.assertTrue(bus.closed)


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.room = SonarRoom()

    def test_a_lidar_sweep_is_recognized_as_one(self):
        self.assertIs(scan_model(self.room.lidar_scan((0.0, 0.0, 0.0))), LIDAR)
        self.assertIs(scan_model(self.room.scan((0.0, 0.0, 0.0))), SONAR)

    def test_walls_from_lidar_sweeps_land_where_the_walls_are(self):
        surface_map = SurfaceMap()
        rng = np.random.default_rng(5)
        for pose in ((0.0, 0.0, 0.0), (0.0, 0.0, 90.0), (0.0, 0.0, -90.0), (20.0, -20.0, 180.0), (-30.0, 30.0, 45.0)):
            surface_map.integrate(self.room.lidar_scan(pose, rng=rng), pose, 2.0, 1.0)
        walls = surface_map.walls()
        self.assertGreaterEqual(len(walls), 3)
        # Every wall sits on one of the room's walls (or the post), within a few cm.
        for wall in walls:
            distances = []
            for start, end in self.room.segments:
                edge = end - start
                t = np.clip(((wall.centroid - start) @ edge) / (edge @ edge), 0, 1)
                distances.append(float(np.linalg.norm(wall.centroid - (start + t * edge))))
            self.assertLess(min(distances), 3.0)
        # The sonar's beam width estimate is left alone by lidar sweeps.
        self.assertEqual(surface_map.beam_width_deg, 25.0)

    def test_a_reading_straddling_an_edge_is_dropped_but_a_chair_leg_is_kept(self):
        scan = [{"angle_deg": angle, "distance_cm": 200.0, "sensor": "lidar"} for angle in range(-10, 11)]
        scan[5]["distance_cm"] = 150.0  # between a near edge...
        for item in scan[:5]:
            item["distance_cm"] = 80.0  # ...on the left and the far wall on the right
        scan[15]["distance_cm"] = 60.0  # a lone thin leg in front of the wall
        echoes = [echo for run in extract_echoes(scan) for echo in run]
        ranges = [round(echo.range_cm) for echo in echoes]
        self.assertNotIn(150, ranges)
        self.assertIn(60, ranges)
        self.assertEqual(next(echo for echo in echoes if round(echo.range_cm) == 60).kind, "point")


class ExplorationTests(unittest.TestCase):
    def test_lidar_sweeps_explore_both_rooms_quickly_without_bumping(self):
        learner = ExplorationLearner()
        floor, grid, history, bumps = explore(
            30, plan=lambda *args: plan_explore_step(*args, learner=learner), start=(-100.0, -60.0, 45.0), sensor="lidar"
        )
        self.assertEqual(bumps, 0)
        for box in (ROOM_A, CORRIDOR, ROOM_B):
            self.assertGreater(floor.room_coverage(grid, box), 0.85, box)

    def check_matches(self, floor, grid, area, seed, cases=40):
        rng = random.Random(seed)
        errors = []
        done = 0
        while done < cases:
            truth = (rng.uniform(area[0], area[2]), rng.uniform(area[1], area[3]), rng.uniform(0, 360))
            if floor.clearance(truth[:2]) < 25:
                continue
            done += 1
            guess = (truth[0] + rng.uniform(-20, 20), truth[1] + rng.uniform(-20, 20), truth[2] + rng.uniform(-15, 15))
            match = grid.match_scan(floor.scan(truth), guess, 24.0, 20.0)
            if match is not None and match.good:
                errors.append(
                    (math.hypot(match.pose[0] - truth[0], match.pose[1] - truth[1]), abs((match.pose[2] - truth[2] + 180) % 360 - 180))
                )
        return errors

    def test_lidar_sweeps_fit_the_map_within_a_few_centimetres(self):
        # The kernel was set on seed 7 of this house; seeds 11 and 23 took no part.
        floor = FloorPlan("lidar")
        grid = CoverageGrid()
        for x in range(-120, 121, 60):
            for y in range(-70, 121, 60):
                for heading in (0, 90, 180, 270):
                    if floor.clearance((x, y)) >= 25:
                        grid.integrate(floor.scan((x, y, heading)), (x, y, heading), 2.0)
        for seed in (11, 23):
            errors = self.check_matches(floor, grid, (-120, -70, 120, 120), seed)
            self.assertGreaterEqual(len(errors), 25, f"seed {seed}: too few trusted")
            self.assertLess(max(error[0] for error in errors), 8.0, errors)
            self.assertLess(max(error[1] for error in errors), 3.5, errors)


def corridor_scan(points):
    """A lidar sweep, -60 to 60 degrees, reading 300 cm except at `points` (angle: range)."""
    return [
        {"angle_deg": angle, "distance_cm": points.get(angle, 300.0), "sensor": "lidar"} for angle in range(60, -61, -1)
    ]


class ClearanceTests(unittest.TestCase):
    def test_something_in_the_car_wide_corridor_limits_the_way_ahead(self):
        # A chair leg 60 cm ahead and 10 cm right of centre: in the way.
        leg = round(math.degrees(math.atan2(10, 60)))
        self.assertAlmostEqual(forward_clearance(corridor_scan({-leg: 61.0})), 60.0, delta=1.5)
        # The same leg 35 cm to the side is not.
        aside = round(math.degrees(math.atan2(35, 60)))
        self.assertGreater(forward_clearance(corridor_scan({-aside: 69.0})), 250.0)

    def test_the_sonar_riding_along_catches_glass_the_lidar_sees_through(self):
        scan = corridor_scan({})
        for item in scan:
            if abs(item["angle_deg"]) <= 5:
                item["sonar_cm"] = 40.0
        self.assertEqual(forward_clearance(scan), 40.0)
        self.assertEqual(direction_clearances(scan)[0.0], 40.0)

    def test_no_reading_ahead_at_all_is_not_clear(self):
        scan = corridor_scan({})
        for item in scan:
            if abs(item["angle_deg"]) <= 3:
                item["distance_cm"] = None
        self.assertIsNone(forward_clearance(scan))


class FakeLidar:
    """Reads 100 cm plus the head's pan: each direction its own range."""

    settle_seconds = 0.01

    def __init__(self, pan_tilt_driver):
        self.pan_tilt_driver = pan_tilt_driver

    def get_distance(self):
        return 100.0 + self.pan_tilt_driver.pan

    def details(self):
        return {"signal": 3000, "temperature_c": 40.0}

    def close(self):
        pass


class FixedSonar:
    def __init__(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance

    def close(self):
        pass


class MappingTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("SCAN_FIRST_SETTLE_SECONDS", 0.0),
            ("LIDAR_STEP_SETTLE_SECONDS", 0.0),
            ("LIDAR_SAMPLE_INTERVAL_SECONDS", 0.0),
        ):
            patcher = patch(f"robocar.mapping.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        self.pan_driver = SimulatedPanTilt()
        self.pan_tilt = PanTiltController(self.pan_driver, physical=False)
        self.sonar = RangeSensorController(FixedSonar(150.0), physical=True, sample_interval=0.5)
        self.lidar = RangeSensorController(FakeLidar(self.pan_driver), physical=True, sample_interval=0.5, name="Lidar")

        class Camera:
            def capture_snapshot(self):
                return b"jpeg"

        self.mapping = MappingController(
            Camera(), self.drive, self.pan_tilt, self.sonar, MappingStore(Path(":memory:")), lidar=self.lidar
        )
        for closer in (self.lidar.close, self.sonar.close, self.pan_tilt.close, self.drive.close, self.mapping.close):
            self.addCleanup(closer)

    def test_a_sweep_reads_the_lidar_every_degree_with_the_sonar_alongside(self):
        scan, _ = self.mapping._sweep(100, 90, ())
        self.assertEqual(len(scan), len(LIDAR_SCAN_ANGLES))
        self.assertEqual(len(LIDAR_SCAN_ANGLES), 121)
        for item in scan:
            self.assertEqual(item["sensor"], "lidar")
            # The reading taken with the head at the pan that faces this angle.
            self.assertEqual(item["distance_cm"], 100.0 + 90 - item["angle_deg"])
            self.assertEqual(item["signal"], 3000)
            self.assertEqual(item["sonar_cm"], 150.0)
        self.assertEqual(self.pan_driver.pan, 90)

    def test_safety_takes_the_nearer_sensor_and_measuring_takes_the_lidar(self):
        self.pan_tilt.set_angles(90, 100)
        self.assertEqual(self.mapping._range_now(precise=True), 190.0)
        self.assertEqual(self.mapping._range_now(), 150.0)  # the sonar sees something nearer
        self.assertTrue(self.mapping._ranging_enabled())

    def test_the_map_keeps_the_lidar_points(self):
        scan = corridor_scan({0: 100.0, 10: 120.0})
        points = lidar_points(scan, (0.0, 0.0, 0.0))
        self.assertEqual(len(points), len(scan))
        cloud = PointCloud()
        cloud.add(points)
        cloud.add(points)  # the same points again take no more room
        self.assertEqual(len(cloud), len(points))
        ahead = min(points, key=lambda point: abs(point[0]))
        self.assertAlmostEqual(ahead[1], 107.0, delta=0.1)  # 100 cm from the sensor, 7 cm ahead of centre


class SemanticsTests(unittest.TestCase):
    def test_an_object_takes_the_lidar_range_across_its_middle_not_the_wall_behind(self):
        # A cat-sized box in the middle of a 640-wide picture, the wall 300 cm
        # behind it, one stray short reading at its edge.
        scan = corridor_scan({angle: 120.0 + abs(angle) for angle in range(-3, 4)})
        scan[60 - 5]["distance_cm"] = 40.0  # 5 degrees left: beside the cat, not on it
        detections = [{"label": "cat", "confidence": 0.9, "box": [290, 200, 60, 50]}]
        sighting = sightings_from(detections, 640, 0.0, scan)[0]
        self.assertAlmostEqual(sighting.range_cm, 121.0, delta=2.0)


class HuntTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.floor = FloorPlan()
        cls.grid = build_map(cls.floor)

    def test_with_the_lidar_the_cat_is_measured_closely_and_never_touched(self):
        errors = []
        for seed in (40, 43, 46):
            cat = SimCat(self.floor, (60.0, 80.0), random.Random(seed))
            body = SimHuntBody(self.floor, copy.deepcopy(self.grid), (0.0, -40.0, 0.0), cat, random.Random(seed + 1), lidar=True)
            engine = HuntEngine(body, HuntMemory(), rng=random.Random(seed + 2), session_seconds=150)
            report = body.report

            def watch(status, body=body, cat=cat, engine=engine):
                report(status)
                track = engine.track
                if status["phase"] == "playing" and track is not None and track.distance is not None and body.now() - track.seen_at < 0.5:
                    # From the sensor on the head to the near side of the cat.
                    origin = np.array(body.truth[:2]) + 7.0 * np.array(
                        [-math.sin(math.radians(body.truth[2])), math.cos(math.radians(body.truth[2]))]
                    )
                    truth = float(np.linalg.norm(np.array([cat.x, cat.y]) - origin)) - 10.0
                    errors.append(abs(track.distance - truth))

            body.report = watch
            engine.run()
            self.assertEqual(body.touches, 0, f"seed {seed}")
            self.assertEqual(body.bumps, 0, f"seed {seed}")
            self.assertGreater(body.precise_ranges, 0, f"seed {seed}")
        self.assertTrue(errors)
        self.assertLess(float(np.median(errors)), 4.0)

    def test_playing_mouse_the_car_scoots_away_from_a_charging_cat(self):
        cat = SimCat(self.floor, (0.0, 80.0), random.Random(1), react=0.0, wander=0.0)
        body = SimHuntBody(self.floor, copy.deepcopy(self.grid), (0.0, -10.0, 0.0), cat, random.Random(2), lidar=True)
        engine = HuntEngine(body, HuntMemory(), rng=random.Random(3), session_seconds=60)
        engine._look(90)
        self.assertIsNotNone(engine.track)
        engine._range_to_cat()
        start = body.truth
        spend = body._spend
        charge = {"until": body.t + 1.2}

        def spend_charging(seconds):
            # For a moment the cat runs at the car, 60 cm/s, stopping short of it.
            if body.t < charge["until"]:
                gap = np.array(body.truth[:2]) - np.array([cat.x, cat.y])
                length = float(np.linalg.norm(gap))
                if length > 55.0:
                    step = gap / length * min(60.0 * seconds, length - 55.0)
                    cat.x, cat.y = cat.x + float(step[0]), cat.y + float(step[1])
            spend(seconds)

        body._spend = spend_charging
        description = move_mouse(engine)
        self.assertIn("fled", description)
        self.assertNotIn("fled 0", description)
        moved = math.hypot(body.truth[0] - start[0], body.truth[1] - start[1])
        self.assertGreater(moved, 5.0)
        self.assertEqual(body.touches, 0)
        self.assertGreaterEqual(body.closest_cat, MIN_CAT_DISTANCE_CM)

    def test_cat_motion_measures_closing_speed(self):
        class Engine:
            pass

        engine = Engine()
        clock = [0.0]
        engine.body = types.SimpleNamespace(now=lambda: clock[0])
        engine.track = types.SimpleNamespace(distance=100.0)
        engine._cat_world = lambda: np.array([0.0, engine.track.distance])
        history = deque()
        for step in range(8):
            clock[0] = step * 0.1
            engine.track.distance = 100.0 - 5.0 * step  # 50 cm/s closer
            moved, closing = _cat_motion(engine, history)
        self.assertAlmostEqual(closing, 50.0, delta=1.0)
        self.assertAlmostEqual(moved, 25.0, delta=1.0)


class ExposureTests(unittest.TestCase):
    def test_in_dim_light_moving_frames_do_not_shorten_the_exposure_into_darkness(self):
        exposure = MotionExposure()
        exposure.gain = MAX_GAIN
        # Dark frames, moving ones softer than still ones: chasing sharpness
        # here once made everything black.
        for _ in range(40):
            exposure.note_sharpness(0.5, moving=False, brightness=40.0)
            exposure.note_sharpness(0.2, moving=True, brightness=40.0)
        self.assertGreaterEqual(exposure.moving_cap_us, MOVING_CAP_US)

    def test_in_good_light_soft_moving_frames_still_shorten_it(self):
        exposure = MotionExposure()
        exposure.gain = 4.0
        for _ in range(40):
            exposure.note_sharpness(0.5, moving=False, brightness=110.0)
            exposure.note_sharpness(0.2, moving=True, brightness=110.0)
        self.assertLess(exposure.moving_cap_us, MOVING_CAP_US)


if __name__ == "__main__":
    unittest.main()
