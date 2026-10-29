import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.exploration import CoverageGrid
from robocar.mapping import MappingController, MappingStore, plan_explore_step
from robocar.range_sensor import RangeSensorController
from robocar.semantics import (
    CAMERA_HFOV_DEG,
    SemanticMap,
    Sighting,
    hazards_from,
    mark_dynamic,
    pixel_bearing,
    sightings_from,
    static_scan,
)
from robocar.surfaces import direction, sonar_origin
from sonar_room import SonarRoom


def sweep(distance_at):
    return [
        {"angle_deg": angle, "distance_cm": distance_at(angle), "samples": 3, "error": None}
        for angle in range(57, -61, -3)
    ]


def box_at(bearing_deg: float, width_px: int = 80, frame_width: int = 640) -> list[int]:
    """A detection box whose centre lies `bearing_deg` from the camera axis."""
    focal = (frame_width / 2) / math.tan(math.radians(CAMERA_HFOV_DEG / 2))
    centre = frame_width / 2 - focal * math.tan(math.radians(bearing_deg))
    return [round(centre - width_px / 2), 100, width_px, 200]


class GeometryTests(unittest.TestCase):
    def test_picture_columns_map_to_bearings(self):
        self.assertAlmostEqual(pixel_bearing(320, 640), 0.0)
        self.assertAlmostEqual(pixel_bearing(0, 640), CAMERA_HFOV_DEG / 2, places=6)
        self.assertAlmostEqual(pixel_bearing(640, 640), -CAMERA_HFOV_DEG / 2, places=6)

    def test_a_detection_gets_its_bearing_and_the_sonar_range_toward_it(self):
        scan = sweep(lambda angle: 85.0 if 5 <= angle <= 20 else 240.0)
        detections = [
            {"label": "chair", "confidence": 0.8, "box": box_at(12.0)},
            {"label": "cup", "confidence": 0.9, "box": box_at(-10.0)},  # too small to matter
        ]
        sightings = sightings_from(detections, 640, 0.0, scan)
        self.assertEqual([s.label for s in sightings], ["chair"])
        self.assertAlmostEqual(sightings[0].bearing_deg, 12.0, delta=0.5)
        self.assertEqual(sightings[0].range_cm, 85.0)
        # The head panned 30 degrees left: everything shifts with it.
        panned = sightings_from(detections, 640, 30.0, sweep(lambda angle: None))
        self.assertAlmostEqual(panned[0].bearing_deg, 42.0, delta=0.5)
        self.assertIsNone(panned[0].range_cm)


class DynamicTests(unittest.TestCase):
    def test_readings_that_hit_a_person_are_kept_out_of_the_map(self):
        # A person 80 cm away just left of centre, the wall 200 cm away.
        scan = sweep(lambda angle: 80.0 if -3 <= angle <= 12 else 200.0)
        person = sightings_from([{"label": "person", "confidence": 0.9, "box": box_at(5.0)}], 640, 0.0, scan)
        marked = mark_dynamic(scan, person)
        flagged = {item["angle_deg"] for item in marked if item.get("dynamic")}
        self.assertTrue({0, 3, 6, 9, 12} <= flagged)
        self.assertFalse(any(item.get("dynamic") for item in marked if item["distance_cm"] == 200.0))
        cleaned = static_scan(marked)
        self.assertTrue(all(item["distance_cm"] is None for item in cleaned if item["angle_deg"] in flagged))
        self.assertEqual(hazards_from(person)[0][1], 80.0)


class LandmarkTests(unittest.TestCase):
    def sighting(self, label, pose, point, ranged=True, confidence=0.8):
        origin = sonar_origin(pose)
        offset = np.array(point) - origin
        bearing = math.degrees(math.atan2(-offset[0], offset[1])) - pose[2]
        return Sighting(label, confidence, bearing, 5.0, float(np.linalg.norm(offset)) if ranged else None, (0, 0, 1, 1))

    def test_sightings_of_the_same_chair_fuse_and_different_chairs_stay_apart(self):
        semantic = SemanticMap()
        chair, other_chair = (50.0, 150.0), (-60.0, 150.0)
        for pose in ((0.0, 0.0, 0.0), (30.0, 20.0, 10.0), (-20.0, 40.0, -5.0)):
            noisy = (chair[0] + 3.0, chair[1] - 2.0)
            semantic.add([self.sighting("chair", pose, noisy), self.sighting("chair", pose, other_chair)], pose)
        chairs = sorted(semantic.confirmed(), key=lambda landmark: landmark.x)
        self.assertEqual(len(chairs), 2)
        self.assertAlmostEqual(chairs[1].x, 53.0, delta=2.0)
        self.assertEqual(chairs[1].sightings, 3)

    def test_two_bearings_from_different_places_triangulate_a_landmark(self):
        semantic = SemanticMap()
        fridge = (40.0, 200.0)
        semantic.add([self.sighting("refrigerator", (0.0, 0.0, 0.0), fridge, ranged=False)], (0.0, 0.0, 0.0))
        self.assertEqual(semantic.landmarks, [])  # one bearing alone says too little
        semantic.add([self.sighting("refrigerator", (120.0, 0.0, 20.0), fridge, ranged=False)], (120.0, 0.0, 20.0))
        self.assertEqual(len(semantic.landmarks), 1)
        landmark = semantic.landmarks[0]
        self.assertAlmostEqual(landmark.x, fridge[0], delta=3.0)
        self.assertAlmostEqual(landmark.y, fridge[1], delta=3.0)

    def test_people_never_become_landmarks(self):
        semantic = SemanticMap()
        semantic.add([self.sighting("person", (0.0, 0.0, 0.0), (0.0, 100.0), confidence=0.95)], (0.0, 0.0, 0.0))
        self.assertEqual(semantic.landmarks, [])

    def test_rooms_are_named_from_what_is_in_them_and_split_by_walls(self):
        semantic = SemanticMap()
        pose = (0.0, 0.0, 0.0)
        for _ in range(2):
            semantic.add(
                [
                    self.sighting("bed", pose, (0.0, 150.0)),
                    self.sighting("chair", pose, (80.0, 150.0)),
                    self.sighting("refrigerator", pose, (200.0, 160.0)),
                    self.sighting("oven", pose, (260.0, 160.0)),
                ],
                pose,
            )
        names = {room["name"] for room in semantic.rooms()}
        self.assertIn("kitchen", {room["name"] for room in semantic.rooms(lambda a, b: abs(a[0] - b[0]) > 100)})
        # Without a wall the chair links the bed and the kitchen into one room.
        self.assertEqual(len(semantic.rooms()), 1)
        walled = semantic.rooms(lambda a, b: (a[0] < 150) != (b[0] < 150))
        self.assertEqual(sorted(room["name"] for room in walled), ["bedroom", "kitchen"])
        self.assertTrue(names)


class PlannerHazardTests(unittest.TestCase):
    def corridor(self):
        walls = (
            ((-40.0, -50.0), (-40.0, 290.0)),
            ((40.0, -50.0), (40.0, 290.0)),
            ((-40.0, -50.0), (40.0, -50.0)),
            ((-40.0, 290.0), (40.0, 290.0)),
        )
        room = SonarRoom(walls=walls, posts=())
        grid = CoverageGrid()
        pose = (0.0, 0.0, 0.0)
        for heading in (0.0, -90.0, 180.0, 90.0):
            looking = (0.0, 0.0, heading)
            grid.integrate(room.scan(looking), looking, 25.0)
        return room, grid, pose

    def test_a_person_close_ahead_stops_the_car_driving_on(self):
        room, grid, pose = self.corridor()
        scan = room.scan(pose)
        self.assertEqual(plan_explore_step(scan, pose, grid, 0, 0).action, "forward")
        step = plan_explore_step(scan, pose, grid, 0, 0, hazards=[(4.0, 90.0)])
        self.assertNotEqual(step.action, "forward")
        self.assertIn("person or animal", step.reason)
        # Far away, or off to the side, they do not hold the car up.
        self.assertEqual(plan_explore_step(scan, pose, grid, 0, 0, hazards=[(4.0, 300.0)]).action, "forward")
        self.assertEqual(plan_explore_step(scan, pose, grid, 0, 0, hazards=[(70.0, 60.0)]).action, "forward")


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return [dict(detection) for detection in self.detections]


class PictureCamera:
    def capture_snapshot(self) -> bytes:
        image = np.full((480, 640, 3), 90, dtype=np.uint8)
        return cv2.imencode(".jpg", image)[1].tobytes()


class MappingIntegrationTests(unittest.TestCase):
    def controller(self, detections, directory):
        room = SonarRoom()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_driver = SimulatedPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=True)

        class Sensor:
            def get_distance(self):
                return room.distance(drive.odometry(), 90 - pan_driver.pan)

            def close(self):
                pass

        range_sensor = RangeSensorController(Sensor(), physical=True, sample_interval=5)
        mapping = MappingController(
            PictureCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(directory) / "map.sqlite3"),
            detector=FakeDetector(detections),
        )
        self.addCleanup(range_sensor.close)
        self.addCleanup(pan_tilt.close)
        self.addCleanup(drive.close)
        self.addCleanup(mapping.close)
        mapping._run_id = "semantic"
        return mapping

    def test_a_chair_seen_at_a_stop_lands_on_the_map_where_the_sonar_puts_it(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = self.controller([{"label": "chair", "confidence": 0.9, "box": box_at(0.0, 60)}], directory)
            mapping._capture_observation(mode="explore")
            landmarks = mapping.surface_map()["landmarks"]
            self.assertEqual([landmark["label"] for landmark in landmarks], ["chair"])
            # Straight ahead, at the front wall's distance (120 cm from the car's centre).
            self.assertAlmostEqual(landmarks[0]["x"], 0.0, delta=8.0)
            self.assertAlmostEqual(landmarks[0]["y"], 120.0, delta=8.0)
            stored = mapping.observations()[-1]["estimate"]["detections"]
            self.assertEqual(stored[0]["label"], "chair")
            # Rebuilt from the database, it is still there.
            mapping._rebuild_map(mapping._last_node, "semantic")
            self.assertEqual(len(mapping.surface_map()["landmarks"]), 1)
            self.assertIn("objects", {panel["name"] for panel in mapping.vision_panels()})

    def test_a_person_in_view_is_a_hazard_and_out_of_the_walls(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = self.controller([{"label": "person", "confidence": 0.9, "box": box_at(0.0, 120)}], directory)
            mapping._capture_observation(mode="explore")
            scan = mapping.observations()[-1]["scan"]
            self.assertTrue(any(item.get("dynamic") for item in scan if abs(item["angle_deg"]) <= 6))
            self.assertTrue(mapping.status()["hazards"])
            self.assertEqual(mapping.surface_map()["landmarks"], [])
            # The straight-ahead readings were the person's, so they made no wall there.
            for surface in mapping.surface_map()["surfaces"]:
                middle = np.add(surface["start"], surface["end"]) / 2
                self.assertFalse(abs(middle[0]) < 10 and abs(middle[1] - 120) < 15, surface)

    def test_a_failing_detector_never_stops_mapping(self):
        class Broken:
            def detect(self, frame):
                raise RuntimeError("HAT fell off")

        with tempfile.TemporaryDirectory() as directory:
            mapping = self.controller([], directory)
            mapping._detector = Broken()
            scan = mapping._capture_observation(mode="explore")
            self.assertTrue(scan)
            self.assertEqual(mapping.observations()[-1]["estimate"]["detections"], [])


if __name__ == "__main__":
    unittest.main()
