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
from robocar.mapping import MappingController
from robocar.store import MappingStore
from robocar.range_sensor import RangeSensorController
from robocar.web import create_app

SIDEWAYS_MOVES = {"sidestep", "shimmy", "orbit", "zigzag", "mouse"}  # mouse darts sideways
HUNT_STILL_SECONDS_FOR_TESTS = 0.4  # a little longer than the car takes to count as still


class HouseHunt(unittest.TestCase):
    """The two-room house, mapped once for all the hunts below."""

    @classmethod
    def setUpClass(cls):
        cls.floor = FloorPlan()
        cls.grid = build_map(cls.floor)

    def hunt(self, car, cat_at, seed, seconds, cat=None, memory=None, person=None, camera_fixed=True):
        cat = cat or SimCat(self.floor, cat_at, random.Random(seed))
        body = SimHuntBody(self.floor, copy.deepcopy(self.grid), car, cat, random.Random(seed + 1), camera_fixed=camera_fixed)
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

    def test_a_camera_on_the_head_still_works(self):
        engine, body, cat, reason, _ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 300, camera_fixed=False)
        found = next((s["elapsed_seconds"] for s in body.statuses if s["phase"] in ("approaching", "playing")), None)
        self.assertIsNotNone(found, "never found the cat")
        self.assertEqual(body.bumps, 0)
        self.assertEqual(body.touches, 0)

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

    def test_the_moment_the_cat_is_seen_the_car_goes_after_it(self):
        for seed, cat_at in ((31, (-520.0, 150.0)), (41, (90.0, 110.0)), (51, (-300.0, 80.0))):
            engine, body, *_ = self.hunt((0.0, -40.0, 0.0), cat_at, seed, 240)
            self.assertIsNotNone(body.first_visible_at, f"seed {seed}: the cat never came into view")
            chasing = next((st["elapsed_seconds"] for st in body.statuses if st["phase"] == "approaching"), None)
            self.assertIsNotNone(chasing, f"seed {seed}: never went after the cat")
            # No finishing the turn or sweep first: after it within moments.
            self.assertLess(chasing - body.first_visible_at, 1.0, f"seed {seed}")

    def test_it_drives_while_looking_instead_of_stopping_to_sweep(self):
        engine, body, *_ = self.hunt((0.0, -40.0, 0.0), (-520.0, 150.0), 31, 240)
        self.assertEqual(body.observations, 0)  # it knew where it was: no sweeps at all
        self.assertGreater(body.commanded_seconds, 10.0)
        self.assertEqual(body.bumps, 0)


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
        from robocar.bodies import CarHuntBody

        mapping = self.controller()
        body = CarHuntBody(mapping)
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


class HuntBodyTests(unittest.TestCase):
    """The real car's hunt body: eyes on the feed, interruptible actions."""

    def setUp(self):
        import cv2
        import numpy as np

        self.frames = []
        frames = self.frames
        self.cat_visible = True
        test = self
        picture = np.full((480, 640, 3), 90, np.uint8)

        class Camera:
            def capture_frame(self):
                import time

                frames.append(time.monotonic())
                time.sleep(0.01)
                return picture.copy()

            def capture_snapshot(self):
                return cv2.imencode(".jpg", picture)[1].tobytes()

            def set_exposure(self, settings):
                test.exposure = settings

        class Detector:
            available = True

            def detect(self, frame):
                return [{"label": "cat", "confidence": 0.9, "box": [280, 200, 80, 60]}] if test.cat_visible else []

        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)

        class Sensor:
            def get_distance(self):
                return 120.0

            def close(self):
                pass

        self.sensor = RangeSensorController(Sensor(), physical=True, sample_interval=0.05)
        self.mapping = MappingController(Camera(), drive, pan_tilt, self.sensor, MappingStore(Path(":memory:")), detector=Detector())
        for resource in (self.sensor, pan_tilt, drive, self.mapping):
            self.addCleanup(resource.close)

    def body(self):
        from robocar.bodies import CarHuntBody

        body = CarHuntBody(self.mapping)
        body.start()
        self.addCleanup(body.close)
        return body

    def test_a_look_is_the_next_frame_after_the_head_has_turned(self):
        import time

        body = self.body()
        body.look(150)
        pointed = time.monotonic()
        look = body.look(150)
        self.assertIsNotNone(look.cat)
        self.assertTrue(any(at >= pointed - 0.02 for at in self.frames))
        self.assertIsNotNone(self.mapping.hunt_frame())  # the dashboard's live picture
        camera = self.mapping.hunt_status()["camera"]
        self.assertEqual(camera, {})  # filled in by reports while hunting
        body.report({"phase": "seeking", "action": "test"})
        self.assertGreater(self.mapping.hunt_status()["camera"]["fps"], 0)
        self.assertIn("exposure_ms", self.mapping.hunt_status()["camera"])

    def test_a_cat_cuts_a_move_or_turn_short_without_stopping_the_hunt(self):
        import time

        body = self.body()
        self.cat_visible = False
        body.interrupt_on_cat(True)
        with patch.object(self.mapping._drive, "status", wraps=self.mapping._drive.status):
            started = time.monotonic()
            threading_timer = __import__("threading").Timer(0.3, lambda: setattr(self, "cat_visible", True))
            threading_timer.start()
            self.assertTrue(self.mapping._move_burst(0.4, 5.0))  # ends early, and is not a failure
            self.assertLess(time.monotonic() - started, 2.0)
        self.assertIsNotNone(body.spotted())
        self.assertFalse(body.stopped())
        self.assertTrue(self.mapping._turn_by(90.0))  # still interrupted: returns at once
        body.clear_spotted()
        body.interrupt_on_cat(False)
        self.assertFalse(self.mapping._interrupt.is_set())

    def test_a_sweep_is_cut_short_by_a_cat(self):
        body = self.body()
        body.interrupt_on_cat(True)
        self.mapping._interrupt.set()
        scan, _ = self.mapping._sweep(100, 90, ())
        self.assertIsNone(scan)

    def test_steady_commands_and_the_sonar_ahead(self):
        body = self.body()
        body.look(150)
        self.assertEqual(body.range_ahead(), 120.0)
        self.assertEqual(body._pan, 90)  # turned to face ahead
        body.command(0.3, 0.0, 0.05)
        self.assertAlmostEqual(self.mapping._drive.status()["throttle"], 0.3)
        self.assertTrue(body._moving())
        body.halt()
        self.assertEqual(self.mapping._drive.status()["throttle"], 0)

    def test_a_fixed_camera_looks_ahead_whatever_the_head_does(self):
        self.mapping.camera_on_head = False
        body = self.body()
        self.assertFalse(body.camera_pans())
        look = body.look(150)
        self.assertIsNotNone(look.cat)
        sighting = body.sighting()
        self.assertEqual(sighting.pan, 90)
        self.assertAlmostEqual(sighting.look.cat.bearing_deg, 0.0, delta=3.0)  # the cat is ahead, not 60 degrees right

    def test_with_two_cameras_the_hunt_watches_through_the_forward_one(self):
        import numpy as np

        head_frames = []

        class HeadCamera:
            def capture_frame(self):
                head_frames.append(1)
                return np.zeros((480, 640, 3), np.uint8)

            def capture_snapshot(self):
                return None

        forward = self.mapping._camera
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        mapping = MappingController(
            HeadCamera(), drive, pan_tilt, self.sensor, MappingStore(Path(":memory:")),
            detector=self.mapping._detector, camera_on_head=True, hunt_camera=forward,
        )
        for resource in (pan_tilt, drive, mapping):
            self.addCleanup(resource.close)
        from robocar.bodies import CarHuntBody

        body = CarHuntBody(mapping)
        body.start()
        self.addCleanup(body.close)
        self.assertFalse(body.camera_pans())  # the forward camera does not turn with the head
        self.assertTrue(mapping.camera_on_head)  # mapping still pans the head camera
        look = body.look(150)
        self.assertIsNotNone(look.cat)
        self.assertEqual(head_frames, [])  # the head camera was not used for watching

    def test_still_frames_are_offered_for_placing_the_car(self):
        import time

        self.mapping.camera_on_head = False
        body = self.body()
        self.assertIsNotNone(body.watcher.wait_for(time.monotonic(), 2.0))
        time.sleep(HUNT_STILL_SECONDS_FOR_TESTS)
        with patch.object(self.mapping, "place_fix_from_picture", return_value={"used": True}) as fix:
            deadline = time.monotonic() + 2.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = body.try_place_fix()
                time.sleep(0.05)
        self.assertEqual(result, {"used": True})
        jpeg, bearing, odometry = fix.call_args.args
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(bearing, 0.0)  # the fixed camera looks ahead
        self.assertEqual(len(odometry), 3)

    def test_a_hunt_sweep_shows_a_panorama_from_the_feed(self):
        import time

        body = self.body()

        def sweep(mode):
            for pan in range(30, 151, 10):
                self.mapping._pan_tilt.set_angles(pan, 100)
                time.sleep(0.05)

        with patch.object(self.mapping, "_capture_observation", side_effect=sweep):
            body.observe()
        panels = {panel["name"] for panel in self.mapping.vision_panels()}
        self.assertIn("panorama", panels)


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


class SpinLookTests(unittest.TestCase):
    """Looking round on the spot: the gyro says whether the car is really
    turning, so a turn too weak to move it, or a wheel spinning in place,
    must not leave it trying for ever."""

    def engine(self, turns_from):
        floor = FloorPlan()
        cat = SimCat(floor, (-500.0, 150.0), random.Random(1))
        body = SimHuntBody(floor, build_map(floor), (0.0, -40.0, 0.0), cat, random.Random(2))
        drive_for = body._drive_for

        def weak(dt):
            # Below this power the car does not turn at all.
            throttle, strafe, steering = body.command_now
            if abs(steering) < turns_from:
                body.commanded_seconds += dt
                return
            drive_for(dt)

        body._drive_for = weak
        return HuntEngine(body, HuntMemory(), rng=random.Random(3), session_seconds=60), body

    def test_more_power_when_the_car_does_not_turn(self):
        engine, body = self.engine(turns_from=0.5)
        start = body.truth[2]
        self.assertFalse(engine._spin_look())
        self.assertGreaterEqual(abs(body.truth[2] - start), 300.0)
        self.assertLess(body.now(), 15.0)

    def test_it_gives_up_when_it_cannot_turn_at_all(self):
        engine, body = self.engine(turns_from=5.0)
        self.assertFalse(engine._spin_look())
        self.assertLess(body.now(), 8.0)
        self.assertEqual(body.command_now, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
