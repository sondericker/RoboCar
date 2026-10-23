import math
import unittest

import numpy as np

from robocar.localization import (
    VisualMotion,
    compose,
    predict_motion,
    propagate,
    register_scan,
    relative,
)
from robocar.surfaces import SurfaceMap, extract_echoes
from sonar_room import SonarRoom


class PoseMathTests(unittest.TestCase):
    def test_compose_and_relative_are_inverses(self):
        start = (12.0, -4.0, 30.0)
        delta = (3.0, 25.0, -12.0)
        end = compose(start, delta)
        np.testing.assert_allclose(relative(start, end), delta, atol=1e-9)

    def test_forward_motion_with_a_left_heading_moves_left(self):
        x, y, heading = compose((0.0, 0.0, 90.0), (0.0, 10.0, 0.0))
        self.assertAlmostEqual(x, -10.0)
        self.assertAlmostEqual(y, 0.0)

    def test_propagation_grows_uncertainty(self):
        prediction = predict_motion((0.0, 30.0, 0.0), None)
        _, covariance = propagate((0.0, 0.0, 0.0), np.zeros((3, 3)), prediction)
        self.assertGreater(covariance[1, 1], covariance[0, 0])
        self.assertGreater(covariance[2, 2], 0)


class MotionPredictionTests(unittest.TestCase):
    def test_consistent_camera_yaw_refines_odometry(self):
        prediction = predict_motion((0.0, 0.0, 12.0), VisualMotion(10.0, None, 3, 0.4))
        self.assertTrue(prediction.visual_used)
        self.assertAlmostEqual(prediction.delta[2], 10.0, delta=0.6)
        self.assertLess(math.degrees(math.sqrt(prediction.covariance[2, 2])), 2.0)

    def test_agreeing_camera_views_overrule_slipping_wheels(self):
        prediction = predict_motion((0.0, 0.0, 12.0), VisualMotion(35.0, None, 3, 1.0))
        self.assertTrue(prediction.visual_used)
        self.assertAlmostEqual(prediction.delta[2], 35.0)

    def test_a_single_disagreeing_view_is_rejected(self):
        prediction = predict_motion((0.0, 0.0, 12.0), VisualMotion(40.0, None, 1, 0.0))
        self.assertFalse(prediction.visual_used)
        self.assertTrue(prediction.visual_rejected)
        self.assertAlmostEqual(prediction.delta[2], 12.0)

    def test_camera_travel_direction_corrects_sideways_slip(self):
        prediction = predict_motion((0.0, 30.0, 0.0), VisualMotion(0.0, 10.0, 3, 0.0))
        self.assertTrue(prediction.bearing_used)
        self.assertLess(prediction.delta[0], -0.5)
        self.assertAlmostEqual(math.hypot(prediction.delta[0], prediction.delta[1]), 30.0, delta=0.01)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.room = SonarRoom()
        self.surface_map = SurfaceMap()
        for pose in [(0, 0, 0), (0, 30, 0), (0, 30, 35), (-20, 50, 35), (-20, 50, -40), (10, 20, -40)]:
            self.surface_map.integrate(self.room.scan(pose), pose)

    def register(self, truth, prior):
        echoes = [echo for run in extract_echoes(self.room.scan(truth)) for echo in run]
        covariance = np.diag([10.0**2, 10.0**2, math.radians(6) ** 2])
        return register_scan(echoes, prior, covariance, self.surface_map)

    def test_registration_recovers_position_and_heading_error(self):
        truth = (0.0, 40.0, 10.0)
        result = self.register(truth, (8.0, 34.0, 14.0))
        self.assertTrue(result.accepted, result.reason)
        self.assertLess(math.hypot(result.pose[0] - truth[0], result.pose[1] - truth[1]), 3.0)
        self.assertLess(abs(result.pose[2] - truth[2]), 1.5)

    def test_registration_reports_unconstrained_directions(self):
        # Facing only the front wall, forward position is pinned but sideways is not.
        result = self.register((0.0, 60.0, 0.0), (0.0, 48.0, 0.0))
        self.assertTrue(result.accepted, result.reason)
        self.assertAlmostEqual(result.pose[1], 60.0, delta=1.0)
        self.assertGreater(result.covariance[0, 0], 4 * result.covariance[1, 1])

    def test_registration_rejects_sonar_that_contradicts_a_confident_prediction(self):
        echoes = [echo for run in extract_echoes(self.room.scan((0.0, 40.0, 0.0))) for echo in run]
        tight = np.diag([1.0, 1.0, math.radians(0.5) ** 2])
        result = register_scan(echoes, (0.0, 32.0, 0.0), tight, self.surface_map)
        self.assertFalse(result.accepted)
        self.assertIn("disagrees", result.reason)
        self.assertEqual(result.pose, (0.0, 32.0, 0.0))

    def test_registration_waits_for_mapped_surfaces(self):
        echoes = [echo for run in extract_echoes(self.room.scan((0.0, 0.0, 0.0))) for echo in run]
        result = register_scan(echoes, (0.0, 0.0, 0.0), np.eye(3), SurfaceMap())
        self.assertFalse(result.accepted)
        self.assertIn("no mapped surfaces", result.reason)


if __name__ == "__main__":
    unittest.main()
