import threading
import time
import unittest

import numpy as np

from robocar.watch import (
    MAX_GAIN,
    MOVING_CAP_LIMITS_US,
    STILL_CAP_US,
    MotionExposure,
    Watcher,
)

CAT = [{"label": "cat", "confidence": 0.9, "box": [280, 200, 80, 60]}]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def frame(level=110, size=(48, 64)):
    return np.full((*size, 3), level, np.uint8)


class WatcherTests(unittest.TestCase):
    def watcher(self, detections, **kwargs):
        clock = kwargs.pop("clock", Clock())
        self.clock = clock
        self.published = []
        return Watcher(
            lambda: frame(),
            lambda image: detections(),
            lambda: 120,
            lambda: (1.0, 2.0, 30.0),
            lambda: False,
            publish=lambda image, found: self.published.append(found),
            clock=clock,
            **kwargs,
        )

    def test_each_frame_is_published_with_where_it_was_taken(self):
        watcher = self.watcher(lambda: CAT)
        sighting = watcher.step()
        self.assertEqual(sighting.pan, 120)
        self.assertEqual(sighting.odometry, (1.0, 2.0, 30.0))
        self.assertIsNotNone(sighting.look.cat)
        self.assertIs(watcher.latest(), sighting)
        self.assertEqual(len(self.published), 1)

    def test_a_cat_interrupts_only_while_interrupts_are_on(self):
        seen = [False]
        watcher = self.watcher(lambda: CAT if seen[0] else [])
        watcher.step()
        self.assertFalse(watcher.interrupt.is_set())
        seen[0] = True
        watcher.step()
        self.assertFalse(watcher.interrupt.is_set())  # off: noted, no interrupt
        self.assertIsNotNone(watcher.spotted())
        watcher.set_interrupts(True)
        self.assertIsNone(watcher.spotted())  # a stale sighting is forgotten
        watcher.step()
        self.assertTrue(watcher.interrupt.is_set())
        self.assertIsNotNone(watcher.spotted())
        watcher.clear_spotted()
        self.assertFalse(watcher.interrupt.is_set())
        self.assertIsNone(watcher.spotted())

    def test_waiting_for_the_next_frame(self):
        watcher = Watcher(lambda: frame(), lambda image: [], lambda: 90, lambda: (0.0, 0.0, 0.0), lambda: False)
        watcher.start()
        self.addCleanup(watcher.stop)
        now = time.monotonic()
        sighting = watcher.wait_for(now, 2.0)
        self.assertIsNotNone(sighting)
        self.assertGreaterEqual(sighting.at, now)
        self.assertIsNone(watcher.wait_for(time.monotonic() + 60, 0.1))  # never comes: gives up

    def test_the_watcher_survives_a_failing_camera(self):
        calls = [0]

        def grab():
            calls[0] += 1
            if calls[0] < 3:
                raise RuntimeError("camera hiccup")
            return frame()

        watcher = Watcher(grab, lambda image: [], lambda: 90, lambda: (0.0, 0.0, 0.0), lambda: False)
        watcher.start()
        self.addCleanup(watcher.stop)
        self.assertIsNotNone(watcher.wait_for(0.0, 3.0))
        self.assertIsNone(watcher.error)

    def test_pictures_for_a_panorama_are_kept_one_per_pan_step(self):
        pans = iter([30, 31, 40, 50, 52, 60])
        watcher = Watcher(lambda: frame(), lambda image: [], lambda: next(pans), lambda: (0.0, 0.0, 0.0), lambda: False)
        watcher.collect(True)
        for _ in range(6):
            watcher.step()
        kept = watcher.collect(False)
        self.assertEqual(sorted(kept), [30, 40, 50, 60])
        self.assertEqual(watcher.collect(False), {})

    def test_the_interrupt_can_be_shared(self):
        shared = threading.Event()
        watcher = self.watcher(lambda: CAT, interrupt=shared)
        watcher.set_interrupts(True)
        watcher.step()
        self.assertTrue(shared.is_set())


class MotionExposureTests(unittest.TestCase):
    def test_a_dim_scene_while_moving_gets_gain_not_long_exposures(self):
        exposure = MotionExposure()
        for _ in range(30):
            exposure.update(20.0, moving=True)
        self.assertLessEqual(exposure.exposure_us, exposure.moving_cap_us + 1)
        self.assertAlmostEqual(exposure.gain, MAX_GAIN, delta=0.01)

    def test_standing_still_allows_longer_exposures(self):
        exposure = MotionExposure()
        for _ in range(30):
            exposure.update(20.0, moving=False)
        self.assertGreater(exposure.exposure_us, exposure.moving_cap_us)
        self.assertLessEqual(exposure.exposure_us, STILL_CAP_US)

    def test_a_bright_scene_drops_gain_first_then_exposure(self):
        exposure = MotionExposure()
        for _ in range(40):
            exposure.update(250.0, moving=True)
        self.assertAlmostEqual(exposure.gain, 1.0, delta=0.01)
        self.assertLess(exposure.exposure_us, exposure.moving_cap_us)

    def test_small_changes_are_not_worth_sending(self):
        exposure = MotionExposure()
        exposure.update(20.0, moving=True)
        settled = MotionExposure(exposure_us=exposure.exposure_us, gain=exposure.gain)
        self.assertIsNone(settled.update(108.0, moving=True))

    def test_blur_while_moving_shortens_the_moving_cap_and_sharpness_lets_it_grow(self):
        exposure = MotionExposure()
        start = exposure.moving_cap_us
        for _ in range(10):
            exposure.note_sharpness(100.0, moving=False)
            exposure.note_sharpness(40.0, moving=True)
        self.assertLess(exposure.moving_cap_us, start)
        shortened = exposure.moving_cap_us
        for _ in range(10):
            exposure.note_sharpness(100.0, moving=False)
            exposure.note_sharpness(97.0, moving=True)
        self.assertGreater(exposure.moving_cap_us, shortened)
        for _ in range(200):
            exposure.note_sharpness(100.0, moving=False)
            exposure.note_sharpness(10.0, moving=True)
        self.assertGreaterEqual(exposure.moving_cap_us, MOVING_CAP_LIMITS_US[0])


if __name__ == "__main__":
    unittest.main()
