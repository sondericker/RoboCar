import math
import unittest

import numpy as np

from robocar.surfaces import SurfaceMap, extract_echoes, to_world
from sonar_room import SonarRoom

SURVEY_POSES = [
    (0.0, 0.0, 0.0),
    (0.0, 30.0, 0.0),
    (0.0, 30.0, 35.0),
    (-20.0, 50.0, 35.0),
    (-20.0, 50.0, -40.0),
    (10.0, 20.0, -40.0),
    (10.0, 20.0, 80.0),
    (10.0, 20.0, -90.0),
    (0.0, 0.0, 180.0),
]


def wall_scan(distance_cm: float, normal_angle_deg: float = 0.0) -> list[dict[str, object]]:
    return SonarRoom(
        walls=(
            (
                tuple(distance_cm * np.array([-math.sin(math.radians(normal_angle_deg)), math.cos(math.radians(normal_angle_deg))])
                + 400 * np.array([math.cos(math.radians(normal_angle_deg)), math.sin(math.radians(normal_angle_deg))])),
                tuple(distance_cm * np.array([-math.sin(math.radians(normal_angle_deg)), math.cos(math.radians(normal_angle_deg))])
                - 400 * np.array([math.cos(math.radians(normal_angle_deg)), math.sin(math.radians(normal_angle_deg))])),
            ),
        ),
        posts=(),
    ).scan((0.0, 0.0, 0.0))


class EchoExtractionTests(unittest.TestCase):
    def test_constant_depth_region_becomes_one_plane_echo_facing_the_sensor(self):
        echoes = [echo for run in extract_echoes(wall_scan(100.0)) for echo in run]
        planes = [echo for echo in echoes if echo.kind == "plane"]
        self.assertEqual(len(planes), 1)
        self.assertAlmostEqual(planes[0].bearing_deg, 0.0, delta=0.1)

    def test_oblique_echoes_are_shifted_onto_the_wall_instead_of_bowing(self):
        scan = wall_scan(100.0)
        echoes = [echo for run in extract_echoes(scan) for echo in run]
        corrected = [abs(echo.y_cm - 100.0) for echo in echoes]
        raw = [
            abs(7.0 + float(item["distance_cm"]) * math.cos(math.radians(float(item["angle_deg"]))) - 100.0)
            for item in scan
            if item["distance_cm"] is not None
        ]
        self.assertLess(max(corrected), 1.0)
        self.assertGreater(max(raw), 5.0)

    def test_range_jumps_split_the_sweep_into_separate_surfaces(self):
        scan = [
            {"angle_deg": angle, "distance_cm": 50.0 if angle < 0 else 150.0, "samples": 3, "error": None}
            for angle in range(-30, 31, 6)
        ]
        self.assertEqual(len(extract_echoes(scan)), 2)

    def test_missing_echoes_and_out_of_range_readings_are_ignored(self):
        scan = [
            {"angle_deg": 0, "distance_cm": None, "samples": 0, "error": "No echo received"},
            {"angle_deg": 6, "distance_cm": 900.0, "samples": 3, "error": None},
            {"angle_deg": 12, "distance_cm": 1.0, "samples": 3, "error": None},
        ]
        self.assertEqual(extract_echoes(scan), [])


class SurfaceMapTests(unittest.TestCase):
    def build_room_map(self) -> SurfaceMap:
        room = SonarRoom()
        surface_map = SurfaceMap()
        for pose in SURVEY_POSES:
            surface_map.integrate(room.scan(pose), pose)
        return surface_map

    def test_sweeps_from_several_poses_converge_on_the_room_walls(self):
        surface_map = self.build_room_map()
        walls = surface_map.walls(confirmed_only=True)
        expected = {
            "front": (np.array([0.0, -1.0]), 120.0),
            "left": (np.array([1.0, 0.0]), -110.0),
            "right": (np.array([-1.0, 0.0]), 130.0),
        }
        for name, (normal, position) in expected.items():
            matching = [
                wall
                for wall in walls
                if float(wall.normal @ normal) > math.cos(math.radians(3)) and wall.length_cm > 100
            ]
            self.assertEqual(len(matching), 1, name)
            wall = matching[0]
            coordinate = wall.centroid[1] if name == "front" else wall.centroid[0]
            self.assertAlmostEqual(coordinate, position, delta=1.5, msg=name)
            self.assertLess(wall.rms_cm, 1.5, name)
            self.assertGreaterEqual(wall.observations, 4, name)

    def test_walls_meeting_at_a_hidden_corner_join_into_one_contour(self):
        contours = self.build_room_map().contours()
        corner_contours = [contour for contour in contours if len(contour["surface_ids"]) >= 2]
        corners = [np.array(point) for contour in corner_contours for point in contour["points"]]
        self.assertTrue(any(np.linalg.norm(point - np.array([-110.0, 120.0])) < 2.0 for point in corners))
        for contour in contours:
            self.assertFalse(contour["closed"] and len(contour["surface_ids"]) < 3)

    def test_point_reflector_is_kept_as_a_feature_not_a_wall(self):
        surface_map = SurfaceMap()
        scan = [
            {"angle_deg": angle, "distance_cm": 60.0 if abs(angle) <= 6 else None, "samples": 3, "error": None}
            for angle in range(-30, 31, 6)
        ]
        surface_map.integrate(scan, (0.0, 0.0, 0.0))
        surface_map.integrate(scan, (0.0, 0.0, 0.0))
        features = surface_map.features(confirmed_only=True)
        self.assertEqual(len(features), 1)
        self.assertEqual(surface_map.walls(), [])
        self.assertAlmostEqual(features[0].centroid[1], 67.0, delta=0.5)

    def test_surfaces_that_beams_pass_through_are_retired(self):
        box = SonarRoom(walls=(((-30.0, 80.0), (30.0, 80.0)), ((-200.0, 200.0), (200.0, 200.0))), posts=())
        empty = SonarRoom(walls=(((-200.0, 200.0), (200.0, 200.0)),), posts=())
        surface_map = SurfaceMap()
        pose = (0.0, 0.0, 0.0)
        surface_map.integrate(box.scan(pose), pose)
        self.assertTrue(any(abs(wall.centroid[1] - 80.0) < 2 for wall in surface_map.walls()))
        for _ in range(3):
            surface_map.integrate(empty.scan(pose), pose)
        self.assertFalse(any(abs(wall.centroid[1] - 80.0) < 10 for wall in surface_map.walls()))

    def test_pose_uncertainty_lowers_the_weight_of_new_echoes(self):
        room = SonarRoom()
        surface_map = SurfaceMap()
        surface_map.integrate(room.scan((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        front = max(surface_map.walls(), key=lambda wall: wall.length_cm)
        precise_weight = float(front.weights.mean())
        surface_map.integrate(room.scan((0.0, 10.0, 0.0)), (0.0, 13.0, 0.0), position_sigma_cm=8.0)
        front = max(surface_map.walls(), key=lambda wall: wall.length_cm)
        self.assertLess(float(front.weights[-1]), precise_weight / 2)
        # The misplaced sweep shifts the refined wall by under half its 3 cm error.
        self.assertAlmostEqual(front.centroid[1], 120.0, delta=1.5)

    def test_to_dict_reports_world_geometry(self):
        data = self.build_room_map().to_dict()
        self.assertEqual(data["scan_count"], len(SURVEY_POSES))
        self.assertTrue(data["surfaces"])
        self.assertTrue(data["contours"])
        self.assertTrue(all(len(echo) == 3 for echo in data["echoes"]))

    def test_world_transform_turns_counterclockwise(self):
        left = to_world((10.0, 0.0, 90.0), np.array([0.0, 20.0]))[0]
        np.testing.assert_allclose(left, [-10.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
