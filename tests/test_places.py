import math
import tempfile
import unittest

from robocar.semantics import CAMERA_HFOV_DEG
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from camera_scene import RoomScene
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt
from robocar.localization import RegistrationResult, compose, relative
from robocar.mapping import MappingController
from robocar.store import MappingStore
from robocar.places import (
    GraphEdge,
    PlaceRecognizer,
    PoseGraph,
    information_from_sigma,
    locate_from_places,
    make_keyframe,
    verify_place,
)
from robocar.range_sensor import RangeSensorController
from robocar.surfaces import SurfaceMap
from sonar_room import ROOM_WALLS, SonarRoom


def position_error(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


class PoseGraphTests(unittest.TestCase):
    def build_square(self) -> tuple[PoseGraph, list]:
        truth = [(0.0, 0.0, 0.0)]
        moves = []
        for _ in range(4):
            moves += [(0.0, 50.0, 0.0)] * 4 + [(0.0, 0.0, 90.0)]
        for move in moves:
            truth.append(compose(truth[-1], move))
        graph = PoseGraph()
        estimate = (0.0, 0.0, 0.0)
        graph.add_node(0, estimate, "run")
        for index, move in enumerate(moves):
            # Odometry that reads 5% long and over-turns at every corner.
            noisy = (move[0], move[1] * 1.05, move[2] + (2.0 if move[2] else 0.3))
            estimate = compose(estimate, noisy)
            graph.add_node(index + 1, estimate, "run")
            sigma = (2.0, 2.0 + 0.1 * abs(noisy[1]), 1.0 + 0.1 * abs(noisy[2]))
            graph.add_edge(GraphEdge(index, index + 1, noisy, information_from_sigma(sigma), "odometry"))
        return graph, truth

    def test_loop_closure_pulls_a_drifting_path_back_together(self):
        graph, truth = self.build_square()
        end = len(truth) - 1
        self.assertGreater(position_error(graph.nodes[end], truth[end]), 25.0)
        graph.add_edge(GraphEdge(0, end, relative(truth[0], truth[end]), information_from_sigma((3, 3, 1.5)), "loop"))
        graph.optimize(graph.component(0))
        self.assertLess(position_error(graph.nodes[end], truth[end]), 1.0)
        self.assertLess(max(position_error(graph.nodes[node], truth[node]) for node in graph.nodes), 16.0)

    def test_a_false_place_match_is_rejected(self):
        graph, truth = self.build_square()
        end = len(truth) - 1
        graph.add_edge(GraphEdge(0, end, relative(truth[0], truth[end]), information_from_sigma((3, 3, 1.5)), "loop"))
        false_match = GraphEdge(3, 10, (0.0, 0.0, 0.0), information_from_sigma((3, 3, 1.5)), "loop")
        graph.add_edge(false_match)
        graph.optimize(graph.component(0))
        rejected = graph.reject_inconsistent_loops(graph.component(0))
        self.assertEqual(rejected, [false_match])
        self.assertLess(position_error(graph.nodes[end], truth[end]), 1.0)

    def test_separate_sessions_are_separate_components(self):
        graph = PoseGraph()
        for node, run in ((1, "a"), (2, "a"), (3, "b"), (4, "b")):
            graph.add_node(node, (0.0, 0.0, 0.0), run)
        graph.add_edge(GraphEdge(1, 2, (0.0, 10.0, 0.0), information_from_sigma((1, 1, 1)), "odometry"))
        graph.add_edge(GraphEdge(3, 4, (0.0, 10.0, 0.0), information_from_sigma((1, 1, 1)), "odometry"))
        self.assertEqual(sorted(map(sorted, graph.components())), [[1, 2], [3, 4]])


class PlaceRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.scene = RoomScene(ROOM_WALLS)

    def keyframe(self, node: int, pose, scene=None, pan_offset: float = 0.0):
        view = (scene or self.scene).view(pose, pan_offset)
        return make_keyframe(node, "run", view, pan_offset)

    def test_verification_measures_a_turn_between_views(self):
        match = verify_place(self.keyframe(1, (0.0, 0.0, 0.0)), self.keyframe(2, (0.0, 0.0, 20.0)))
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match.yaw_deg, 20.0, delta=2.0)

    def test_verification_accounts_for_where_the_camera_was_panned(self):
        # Same camera direction, but the car turned 20 degrees and the head
        # panned back: the car's turn is what is reported.
        match = verify_place(self.keyframe(1, (0.0, 0.0, 0.0)), self.keyframe(2, (0.0, 0.0, 20.0), pan_offset=-20.0))
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match.yaw_deg, 20.0, delta=2.0)

    def test_unrelated_views_do_not_verify(self):
        remembered = self.keyframe(1, (0.0, 0.0, 0.0))
        self.assertIsNone(verify_place(remembered, self.keyframe(2, (-40.0, -30.0, 150.0))))
        other_room = RoomScene(ROOM_WALLS, seed=99)
        self.assertIsNone(verify_place(remembered, self.keyframe(3, (0.0, 0.0, 0.0), scene=other_room)))

    def test_descriptor_voting_finds_the_matching_place(self):
        recognizer = PlaceRecognizer()
        poses = {1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 90.0), 3: (0.0, 0.0, 180.0), 4: (0.0, 0.0, -90.0), 5: (40.0, 40.0, 45.0)}
        for node, pose in poses.items():
            recognizer.add(self.keyframe(node, pose))
        query = self.keyframe(9, (10.0, -10.0, 85.0))
        candidates = recognizer.candidates(query, lambda keyframe: True)
        self.assertEqual(candidates[0][0].observation_id, 2)


class CameraFixTests(unittest.TestCase):
    """Placing the car from its camera picture alone, against remembered places."""

    def test_a_live_picture_places_the_car_from_remembered_places(self):
        # The forward HQ camera's view.
        scene = RoomScene(ROOM_WALLS, hfov_deg=CAMERA_HFOV_DEG)
        places = {1: (0.0, 0.0, 0.0), 2: (40.0, 30.0, 20.0), 3: (-40.0, 20.0, -25.0), 4: (0.0, 60.0, 10.0), 5: (60.0, -20.0, 35.0)}
        recognizer = PlaceRecognizer()
        for node, pose in places.items():
            recognizer.add(make_keyframe(node, "run", scene.view(pose, 0.0), 0.0, CAMERA_HFOV_DEG))
        truth = (30.0, 0.0, 5.0)
        query = make_keyframe(-1, "run", scene.view(truth, 0.0), 0.0, CAMERA_HFOV_DEG)
        fix = locate_from_places(query, recognizer, places, (45.0, -10.0, 5.0))
        self.assertIsNotNone(fix)
        self.assertGreaterEqual(len(fix.places), 2)
        self.assertLess(math.hypot(fix.position[0] - truth[0], fix.position[1] - truth[1]), 12.0)
        self.assertLess(abs(fix.heading_deg - truth[2]), 3.0)

    def synthetic(self, truth, places, wrong=()):
        """Matches as a perfect camera would report them (or `wrong` ones)."""
        from robocar.places import PlaceMatch

        def verify(remembered, query):
            place = places[remembered.observation_id]
            yaw = truth[2] - place[2] + (25.0 if remembered.observation_id in wrong else 0.0)
            bearing = math.degrees(math.atan2(-(truth[0] - place[0]), truth[1] - place[1])) - place[2]
            return PlaceMatch(remembered.observation_id, yaw, bearing, 100, 120)

        class Recognizer:
            def candidates(self, query, allowed, limit=8):
                return [(keyframe, 50) for keyframe in keyframes if allowed(keyframe)][:limit]

        keyframes = [SimpleNamespace(observation_id=node) for node in places]
        return Recognizer(), verify

    def test_one_confidently_wrong_match_is_outvoted(self):
        truth = (20.0, 30.0, 40.0)
        places = {1: (0.0, 0.0, 10.0), 2: (80.0, 20.0, 90.0), 3: (10.0, 90.0, -60.0)}
        recognizer, verify = self.synthetic(truth, places, wrong={3})
        fix = locate_from_places(SimpleNamespace(), recognizer, places, truth, verify=verify)
        self.assertEqual(sorted(fix.places), [1, 2])
        self.assertAlmostEqual(fix.heading_deg, 40.0, delta=0.1)
        self.assertAlmostEqual(fix.position[0], 20.0, delta=0.5)
        self.assertAlmostEqual(fix.position[1], 30.0, delta=0.5)

    def test_no_fix_without_two_agreeing_places_seeing_it_from_different_sides(self):
        truth = (20.0, 30.0, 40.0)
        one = {1: (0.0, 0.0, 10.0)}
        recognizer, verify = self.synthetic(truth, one)
        self.assertIsNone(locate_from_places(SimpleNamespace(), recognizer, one, truth, verify=verify))
        # Two places in line with the car: their directions do not cross.
        in_line = {1: (0.0, 0.0, 10.0), 2: (10.0, 15.0, 10.0)}
        recognizer, verify = self.synthetic(truth, in_line)
        self.assertIsNone(locate_from_places(SimpleNamespace(), recognizer, in_line, truth, verify=verify))
        # Disagreeing headings.
        apart = {1: (0.0, 0.0, 10.0), 2: (80.0, 20.0, 90.0)}
        recognizer, verify = self.synthetic(truth, apart, wrong={2})
        self.assertIsNone(locate_from_places(SimpleNamespace(), recognizer, apart, truth, verify=verify))

    def test_places_far_from_where_the_car_thinks_it_is_are_not_considered(self):
        truth = (20.0, 30.0, 40.0)
        places = {1: (0.0, 0.0, 10.0), 2: (80.0, 20.0, 90.0)}
        recognizer, verify = self.synthetic(truth, places)
        self.assertIsNone(locate_from_places(SimpleNamespace(), recognizer, places, (900.0, 900.0, 0.0), verify=verify))


class SimulatedHouse:
    """Camera, sonar and odometry for a car whose true pose the test controls."""

    def __init__(self, database: Path, scene_seed: int = 11) -> None:
        self.room = SonarRoom()
        self.scene = RoomScene(ROOM_WALLS, seed=scene_seed)
        self.truth = (0.0, 0.0, 0.0)
        self.odometry = (0.0, 0.0, 0.0)
        self.odometry_scale = (1.0, 1.0)
        self.pan = SimulatedPanTilt()
        house = self

        class Camera:
            def capture_snapshot(self) -> bytes:
                return house.scene.view(house.truth, 90 - house.pan.pan)

        class Sensor:
            def get_distance(self) -> float | None:
                return house.room.distance(house.truth, 90 - house.pan.pan)

            def close(self) -> None:
                pass

        class Odometry(DriveController):
            def odometry(self):
                return house.odometry

        self.drive = Odometry(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        self.pan_tilt = PanTiltController(self.pan, physical=True)
        self.range_sensor = RangeSensorController(Sensor(), physical=True, sample_interval=5)
        self.database = database
        self.mapping = MappingController(Camera(), self.drive, self.pan_tilt, self.range_sensor, MappingStore(database))

    def move_to(self, pose) -> None:
        """Move the car; odometry integrates each move with its scale error."""
        moved = relative(self.truth, pose)
        self.odometry = compose(
            self.odometry,
            (moved[0] * self.odometry_scale[0], moved[1] * self.odometry_scale[0], moved[2] * self.odometry_scale[1]),
        )
        self.truth = pose

    def begin_session(self, run_id: str, start) -> None:
        self.truth = start
        self.odometry = (0.0, 0.0, 0.0)
        mapping = self.mapping
        with mapping._lock:
            mapping._run_id = run_id
            mapping._reset_localization()
        with mapping._map_lock:
            mapping._surface_map = SurfaceMap()
            mapping._path, mapping._history, mapping._links = [], [], []
            mapping._map_run_id = run_id

    def stop_at(self, pose) -> tuple[tuple[float, float, float], dict | None]:
        self.move_to(pose)
        self.mapping._capture_observation(mode="explore")
        status = self.mapping.status()
        return tuple(status["pose"]), status["localization"]["place"]

    def close(self) -> None:
        self.mapping.close()
        self.drive.close()
        self.pan_tilt.close()
        self.range_sensor.close()


class RememberedPlacesTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("SCAN_FIRST_SETTLE_SECONDS", 0.0),
            ("SCAN_STEP_SETTLE_SECONDS", 0.0),
            ("SCAN_SAMPLE_INTERVAL_SECONDS", 0.0),
            ("SCAN_SAMPLES_PER_ANGLE", 1),
            ("CAMERA_MOTION_FRAME_INTERVAL_SECONDS", 0.0),
            ("CAMERA_SETTLE_SECONDS", 0.0),
        ):
            patcher = patch(f"robocar.mapping.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "mapping.sqlite3"

    def map_first_session(self, house: SimulatedHouse) -> None:
        house.begin_session("first", (0.0, 0.0, 0.0))
        for pose in ((0.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 20.0, 35.0), (0.0, 20.0, -35.0), (20.0, 0.0, -35.0)):
            house.stop_at(pose)

    def test_a_new_session_recognizes_where_it_is(self):
        house = SimulatedHouse(self.database)
        try:
            self.map_first_session(house)
            start = (35.0, -15.0, 25.0)
            house.begin_session("second", start)
            estimate, place = house.stop_at(start)
            self.assertIsNotNone(place)
            self.assertEqual(place["kind"], "relocalized")
            self.assertEqual(place["place_run_id"], "first")
            self.assertLess(position_error(estimate, start), 6.0)
            self.assertLess(abs(estimate[2] - start[2]), 3.0)
            estimate, _ = house.stop_at((35.0, -15.0, 0.0))
            self.assertLess(position_error(estimate, (35.0, -15.0)), 4.0)
            surface_map = house.mapping.surface_map()
            self.assertEqual(len(surface_map["history"]), 1)
            self.assertGreaterEqual(len(surface_map["links"]), 1)
            self.assertEqual(house.mapping.status()["places"]["sessions"], 2)
        finally:
            house.close()

        # Everything is rebuilt the same way after a restart.
        reopened = SimulatedHouse(self.database)
        try:
            status = reopened.mapping.status()
            self.assertEqual(status["places"]["places"], 7)
            self.assertGreaterEqual(status["places"]["links"], 1)
            self.assertEqual(status["places"]["sessions"], 2)
            self.assertLess(position_error(status["pose"], (35.0, -15.0)), 4.0)
            self.assertEqual(len(reopened.mapping.surface_map()["history"]), 1)
        finally:
            reopened.close()

    def test_an_unfamiliar_room_is_not_mistaken_for_a_remembered_one(self):
        house = SimulatedHouse(self.database)
        try:
            self.map_first_session(house)
        finally:
            house.close()
        elsewhere = SimulatedHouse(self.database, scene_seed=99)
        try:
            elsewhere.begin_session("elsewhere", (35.0, -15.0, 25.0))
            _, place = elsewhere.stop_at((35.0, -15.0, 25.0))
            self.assertIsNone(place)
            self.assertEqual(elsewhere.mapping.status()["places"]["links"], 0)
        finally:
            elsewhere.close()

    def test_returning_to_the_start_closes_the_loop_and_removes_drift(self):
        house = SimulatedHouse(self.database)
        # No per-stop sonar fit (as along featureless walls), and odometry
        # that reads 12% long on distance and 10% long on turns.
        disabled = patch(
            "robocar.mapping.register_scan",
            lambda echoes, pose, covariance, surface_map: RegistrationResult(
                tuple(pose), covariance, False, 0, 0.0, (0.0, 0.0, 0.0), "disabled"
            ),
        )
        disabled.start()
        self.addCleanup(disabled.stop)
        # Nor camera motion between stops: only odometry carries the pose.
        blind = patch("robocar.mapping.estimate_view_motion", return_value=[])
        blind.start()
        self.addCleanup(blind.stop)
        house.odometry_scale = (1.12, 1.10)
        start = (0.0, -40.0, 0.0)
        house.begin_session("loop", start)
        path = [
            (0, -40, 0), (0, -10, 0), (0, 20, 0), (0, 20, 90), (-30, 20, 90), (-60, 20, 90), (-60, 20, 180),
            (-60, -10, 180), (-60, -40, 180), (-60, -40, -90), (-30, -40, -90), (0, -40, -90),
        ]
        try:
            for pose in path:
                estimate, place = house.stop_at(pose)
                self.assertIsNone(place)
            drifted = position_error(estimate, relative(start, path[-1]))
            self.assertGreater(drifted, 8.0)
            estimate, place = house.stop_at((0.0, -40.0, 0.0))
            self.assertIsNotNone(place)
            self.assertEqual(place["kind"], "loop")
            self.assertLess(position_error(estimate, (0.0, 0.0)), 5.0)
            self.assertLess(abs(estimate[2]), 2.0)
        finally:
            house.close()


if __name__ == "__main__":
    unittest.main()
