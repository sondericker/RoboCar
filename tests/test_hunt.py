import copy
import math
import random
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from explore_sim import FloorPlan
from hunt_sim import SimCat, SimHuntBody, build_map
from robocar.config import Config
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.hunt import (
    DEFAULT_SIZE_SCALE,
    MIN_CAT_DISTANCE_CM,
    MOVES,
    HuntEngine,
    HuntMemory,
    cat_view_from,
    load_memory,
    save_memory,
)
from robocar.mapping import MappingController, MappingStore
from robocar.range_sensor import RangeSensorController
from robocar.web import create_app

SIDEWAYS_MOVES = {"sidestep", "shimmy", "orbit", "zigzag"}


class HouseHunt(unittest.TestCase):
    """The two-room house, mapped once for all the hunts below."""

    @classmethod
    def setUpClass(cls):
        cls.floor = FloorPlan()
        cls.grid = build_map(cls.floor)

    def hunt(self, car, cat_at, seed, seconds, cat=None, memory=None, person=None):
        cat = cat or SimCat(self.floor, cat_at, random.Random(seed))
        body = SimHuntBody(self.floor, copy.deepcopy(self.grid), car, cat, random.Random(seed + 1))
        body.person = person
        engine = HuntEngine(body, memory or HuntMemory(), rng=random.Random(seed + 2), session_seconds=seconds)
        true_distances = []
        report = body.report

        def watch(status):
            report(status)
            if status["phase"] == "playing":
                true_distances.append(math.hypot(cat.x - body.truth[0], cat.y - body.truth[1]))

        body.report = watch
        reason = engine.run()
        return engine, body, cat, reason, true_distances


class SeekingTests(HouseHunt):
    def test_it_finds_the_cat_in_the_other_room_without_bumping_anything(self):
        engine, body, cat, reason, _ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 300)
        found = next((s["elapsed_seconds"] for s in body.statuses if s["phase"] in ("approaching", "playing")), None)
        self.assertIsNotNone(found, "never found the cat")
        self.assertLess(found, 200.0)
        self.assertEqual(body.bumps, 0)
        self.assertEqual(body.touches, 0)
        self.assertEqual(reason, "time_limit")

    def test_a_remembered_haunt_is_searched_first(self):
        memory = HuntMemory()
        memory.cat_spots = [[-520.0, 150.0, 6.0]]
        _, plain, *_ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 250)
        _, remembered, *_ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 250, memory=memory)

        def found(body):
            return next((s["elapsed_seconds"] for s in body.statuses if s["phase"] in ("approaching", "playing")), 1e9)

        self.assertLessEqual(found(remembered), found(plain))


class PlayingTests(HouseHunt):
    def test_play_is_varied_safe_and_rests_now_and_then(self):
        engine, body, cat, reason, distances = self.hunt((0.0, 50.0, 0.0), (80.0, 100.0), 41, 600)
        kinds = engine.stats["moves_by_kind"]
        self.assertGreaterEqual(len(kinds), 10, kinds)
        self.assertEqual(body.bumps, 0)
        self.assertEqual(body.touches, 0)
        self.assertLess(statistics.median(distances), 130.0)
        # Rests: backed off and waited the better part of a minute, then went back to playing.
        self.assertGreaterEqual(engine.stats["rests"], 1)
        rests, began = [], None
        for status in body.statuses:
            if status["phase"] == "resting" and began is None:
                began = status["elapsed_seconds"]
            elif status["phase"] == "playing" and began is not None:
                rests.append(status["elapsed_seconds"] - began)
                began = None
        self.assertTrue(rests, "no rest finished")
        self.assertGreaterEqual(max(rests), 40.0)

    def test_no_move_runs_three_times_in_a_row(self):
        engine, body, *_ = self.hunt((0.0, 50.0, 0.0), (80.0, 100.0), 43, 300)
        sequence = []
        original = engine._choose
        # Re-run the choice policy alone over a long stretch with this engine's state.
        for _ in range(200):
            sequence.append(original())
            engine.recent.append(sequence[-1])
        triples = sum(1 for a, b, c in zip(sequence, sequence[1:], sequence[2:]) if a == b == c)
        self.assertEqual(triples, 0)
        self.assertGreaterEqual(len(set(sequence)), 10)

    def test_it_learns_what_the_cat_likes_and_remembers_it(self):
        memory = HuntMemory()
        for session in range(2):
            cat = SimCat(self.floor, (80.0, 100.0), random.Random(session), likes={"strafe"}, react=0.8, wander=0.0)
            self.hunt((0.0, 0.0, 0.0), None, 50 + session, 600, cat=cat, memory=memory)
        top = [item["move"] for item in memory.favourites(2)]
        self.assertTrue(set(top) <= SIDEWAYS_MOVES, memory.move_scores)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hunt.json"
            save_memory(path, memory)
            self.assertEqual(load_memory(path).favourites(2), memory.favourites(2))

    def test_a_cat_that_wanders_off_is_found_again(self):
        class Wanderer(SimCat):
            moved = False

            def idle(self, seconds):
                # Halfway through, it trots off to the other room.
                if not self.moved and body_clock[0] > 150:
                    self.x, self.y, self.moved = -560.0, 0.0, True
                    return
                super().idle(seconds)

        body_clock = [0.0]
        cat = Wanderer(self.floor, (80.0, 100.0), random.Random(7), wander=0.0)
        floor, grid = self.floor, copy.deepcopy(self.grid)
        body = SimHuntBody(floor, grid, (0.0, 50.0, 0.0), cat, random.Random(8))
        spend = body._spend

        def clocked(seconds):
            spend(seconds)
            body_clock[0] = body.t

        body._spend = clocked
        engine = HuntEngine(body, HuntMemory(), rng=random.Random(9), session_seconds=900)
        engine.run()
        self.assertTrue(cat.moved)
        self.assertGreaterEqual(engine.stats["finds"], 2)
        phases = [s["phase"] for s in body.statuses]
        self.assertIn("searching", phases)
        self.assertEqual(body.bumps, 0)

    def test_it_never_drives_at_a_person_close_ahead(self):
        # Several runs: play moves turn the car about, and any forward part
        # of a move must first look where it is going.
        for seed, person in ((61, (10.0, 80.0)), (62, (60.0, 40.0)), (63, (-20.0, 90.0))):
            engine, body, *_ = self.hunt((0.0, 0.0, 0.0), (80.0, 100.0), seed, 240, person=person)
            self.assertEqual(body.moves_toward_person, 0, f"seed {seed}")
            self.assertEqual(body.touches, 0, f"seed {seed}")

    def test_looking_around_after_a_stop_uses_the_sweep_pictures(self):
        engine, body, *_ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 120)
        self.assertGreater(body.sweep_looks_used, 0)


class PartsTests(unittest.TestCase):
    def test_the_cat_and_people_are_picked_out_of_a_picture(self):
        detections = [
            {"label": "cat", "confidence": 0.6, "box": [100, 200, 80, 60]},
            {"label": "cat", "confidence": 0.9, "box": [280, 200, 80, 60]},
            {"label": "person", "confidence": 0.9, "box": [280, 0, 120, 400]},
        ]
        look = cat_view_from(detections, 90, 640, 480)
        self.assertEqual(look.cat.confidence, 0.9)
        self.assertAlmostEqual(look.cat.bearing_deg, 0.0, delta=0.5)
        self.assertTrue(look.person_ahead)
        panned = cat_view_from(detections[1:2], 60, 640, 480)
        self.assertAlmostEqual(panned.cat.bearing_deg, 30.0, delta=0.5)
        self.assertIsNone(cat_view_from([{"label": "dog", "confidence": 0.9, "box": [0, 0, 5, 5]}], 90, 640, 480).cat)

    def test_distance_learning_shrugs_off_stray_readings(self):
        memory = HuntMemory()
        true_scale = 1.2 * DEFAULT_SIZE_SCALE
        for index in range(12):
            distance = 60.0 + 5 * index
            sonar = distance if index % 4 else distance * 0.55  # every fourth catches a wall
            memory.calibrate(true_scale / distance, sonar)
        self.assertAlmostEqual(memory.size_scale, true_scale, delta=0.05 * true_scale)

    def test_every_move_is_registered_with_its_rules(self):
        from robocar.hunt import MOVE_SPECS

        self.assertEqual(set(MOVES), set(MOVE_SPECS))
        self.assertGreaterEqual(len(MOVES), 14)
        self.assertEqual(MIN_CAT_DISTANCE_CM, 30.0)


class ControllerTests(unittest.TestCase):
    def controller(self, detector=object()):
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)

        class Sensor:
            def get_distance(self):
                return 120.0

            def close(self):
                pass

        sensor = RangeSensorController(Sensor(), physical=True, sample_interval=5)
        mapping = MappingController(object(), drive, pan_tilt, sensor, MappingStore(Path(":memory:")), detector=detector)
        for resource in (sensor, pan_tilt, drive, mapping):
            self.addCleanup(resource.close)
        return mapping

    def test_the_hunt_needs_a_map_and_eyes(self):
        mapping = self.controller()
        with self.assertRaisesRegex(RuntimeError, "Map the room first"):
            mapping.start_hunt(True)
        with self.assertRaises(ValueError):
            mapping.start_hunt(False)
        with self.assertRaisesRegex(RuntimeError, "object detection"):
            self.controller(detector=None).start_hunt(True)

    def test_moves_become_mecanum_commands(self):
        from robocar.mapping import _HuntBody

        mapping = self.controller()
        body = _HuntBody(mapping)
        calls = []
        with patch.object(mapping, "_move_burst", side_effect=lambda *a, **k: calls.append((a, k)) or True):
            body.move(20.0, 0.0)
            body.move(0.0, -10.0)
            body.move(10.0, 10.0, speed=0.5)
            body.arc(0.7, 0.1, 1.0)
        (throttle, _), _ = calls[0]
        self.assertGreater(throttle, 0)
        (_, _), kwargs = calls[1]
        self.assertLess(kwargs["strafe"], 0)  # left
        (throttle, seconds), kwargs = calls[2]
        self.assertAlmostEqual(throttle, kwargs["strafe"])
        self.assertLessEqual(throttle, 0.5)
        self.assertEqual(calls[3][1], {"steering": 0.1, "strafe": 0.7})


class HuntPictureTests(unittest.TestCase):
    """Pictures wait for the car to be still; the preview never interrupts."""

    def setUp(self):
        import cv2
        import numpy as np

        self.events = []
        events = self.events
        image = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))[1].tobytes()

        class Camera:
            def capture_snapshot(self):
                events.append(("picture", __import__("time").monotonic()))
                return image

        class Detector:
            available = True

            def detect(self, frame):
                return [{"label": "cat", "confidence": 0.9, "box": [280, 200, 80, 60]}]

        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        sensor = RangeSensorController(None, physical=False)
        self.mapping = MappingController(Camera(), drive, pan_tilt, sensor, MappingStore(Path(":memory:")), detector=Detector())
        for resource in (sensor, pan_tilt, drive, self.mapping):
            self.addCleanup(resource.close)

    def test_a_picture_after_a_move_waits_for_the_car_to_settle(self):
        import time

        from robocar.mapping import HUNT_STILL_SECONDS, _HuntBody

        body = _HuntBody(self.mapping)
        with patch.object(self.mapping, "_move_burst", return_value=True):
            body.move(10.0, 0.0)
        moved = time.monotonic()
        look = body.look(body._pan)
        self.assertIsNotNone(look.cat)
        first_picture = min(t for kind, t in self.events if kind == "picture")
        self.assertGreaterEqual(first_picture - moved, HUNT_STILL_SECONDS - 0.02)
        self.assertIsNotNone(self.mapping.hunt_frame())

    def test_a_hunt_stop_faces_ahead_and_its_sweep_pictures_are_checked_for_the_cat(self):
        from robocar.hunt import pan_bearing
        from robocar.mapping import SweepStrip, _HuntBody

        body = _HuntBody(self.mapping)
        body._point_head(150)
        mapping = self.mapping
        picture = mapping._camera.capture_snapshot()
        pans_at_stop = []

        def stop(mode):
            pans_at_stop.append(mapping._pan_tilt.status()["pan"])
            mapping._sweep_strip = SweepStrip(
                {pan: (picture,) for pan in range(30, 151, 10)}, 90, 100, mapping._drive.odometry()
            )

        with patch.object(mapping, "_capture_observation", side_effect=stop):
            body.observe()
        self.assertEqual(pans_at_stop, [90])
        looks = body.sweep_looks((30, 60, 90, 120, 150))
        self.assertEqual(sorted(looks), [30, 60, 90, 120, 150])
        for pan, look in looks.items():
            self.assertIsNotNone(look.cat)
            self.assertAlmostEqual(look.cat.bearing_deg, pan_bearing(pan), delta=5.0)
        self.assertEqual(body.sweep_looks((90,)), {})  # used once
        # Not after the car has moved since.
        with patch.object(mapping, "_capture_observation", side_effect=stop):
            body.observe()
        with patch.object(mapping._drive, "odometry", return_value=(0.0, 30.0, 0.0)):
            self.assertEqual(body.sweep_looks((90,)), {})

    def test_the_preview_only_shoots_when_the_car_is_idle_and_still(self):
        from robocar.mapping import HUNT_STILL_SECONDS, _HuntBody

        body = _HuntBody(self.mapping)
        with body._acting():
            body.preview()
        self.assertEqual(self.events, [])  # busy: no picture
        body.preview()
        self.assertEqual(self.events, [])  # just moved: not still yet
        body._still_since -= HUNT_STILL_SECONDS + 0.1
        body.preview()
        self.assertTrue(self.events)
        self.assertIsNotNone(self.mapping.hunt_frame())


class SharpPictureTests(unittest.TestCase):
    def test_a_blurred_picture_is_retaken_and_the_settle_delay_learned(self):
        import cv2
        import numpy as np

        from robocar.mapping import _HuntBody

        rng = np.random.default_rng(3)
        scene = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        sharp = cv2.imencode(".jpg", scene)[1].tobytes()
        smeared = cv2.imencode(".jpg", cv2.blur(scene, (25, 1)))[1].tobytes()  # sideways motion blur
        shots = []

        class Camera:
            def __init__(self):
                self.queue = []

            def capture_snapshot(self):
                picture = self.queue.pop(0) if self.queue else sharp
                shots.append(picture)
                return picture

        class Detector:
            available = True

            def detect(self, frame):
                return []

        camera = Camera()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(camera, drive, pan_tilt, sensor, MappingStore(Path(":memory:")), detector=Detector())
        for resource in (sensor, pan_tilt, drive, mapping):
            self.addCleanup(resource.close)
        body = _HuntBody(mapping)
        with patch.object(mapping, "_wait_until_steady", return_value=True):
            body._look_now()  # learns how sharp a still picture here looks
            delay = body.still_seconds
            camera.queue = [smeared]
            shots.clear()
            body._look_now()
            self.assertEqual(shots, [smeared, sharp])  # retaken, the sharp one kept
            self.assertEqual(body.retakes, 1)
            self.assertGreater(body.still_seconds, delay)  # motion blur: wait longer from now on
            longer = body.still_seconds
            for _ in range(5):
                body._look_now()
            self.assertLess(body.still_seconds, longer)  # sharp first shots: back down again
        self.assertEqual(mapping.hunt_status()["camera"], {})  # only reported while hunting


class WebHuntTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Config(mapping_database=Path(":memory:")))
        self.client = self.app.test_client()

    def tearDown(self):
        for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
            self.app.extensions[name].close()

    def test_the_hunt_api_and_dashboard(self):
        status = self.client.get("/api/hunt").get_json()
        self.assertFalse(status["running"])
        self.assertIn("favourites", status["learned"])
        self.assertEqual(self.client.post("/api/hunt", json={"enabled": "yes"}).status_code, 400)
        refused = self.client.post("/api/hunt", json={"enabled": True, "attended": True})
        self.assertEqual(refused.status_code, 409)  # simulation mode: no hardware
        self.assertEqual(self.client.get("/api/hunt/frame.jpg").status_code, 404)
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="hunt-toggle"', page)
        self.assertIn('id="hunt-feed"', page)

    def test_the_hunt_video_streams_its_latest_picture(self):
        mapping = self.app.extensions["robocar_mapping"]
        mapping._hunt_frame = b"\xff\xd8fake-jpeg\xff\xd9"
        mapping._hunt_frame_at = 1.0
        response = self.client.get("/hunt.mjpg")
        first = next(response.response)
        self.assertIn(b"fake-jpeg", first)
        response.close()
        self.assertEqual(self.client.get("/api/hunt/frame.jpg").data, mapping._hunt_frame)


if __name__ == "__main__":
    unittest.main()
