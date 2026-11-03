"""The real car, as the hunt and the motion calibration see it.

Both engines talk to the car through narrow interfaces (hunt.HuntBody and
calibration's rig) so the same code runs the car and the simulations the
tests use. These adapters implement them on top of the mapping controller,
which owns the drive, the pan/tilt head, the range sensors, the cameras and
the map.
"""

import contextlib
import logging
import math
import statistics
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from robocar.exploration import CoverageGrid
from robocar.drive import PAN_LIMITS, pan_bearing
from robocar.hunt import Look
from robocar.hunt import Sighting as FeedSighting
from robocar.localization import Pose, compose, relative
from robocar.mapping import (
    AUTO_ALIGN_FRAMES,
    AUTO_FORWARD_COMMAND,
    AUTO_FORWARD_SECONDS,
    AUTO_STEP_CM,
    AUTO_TURN_PIECE_DEGREES,
    STRIP_CENTRE_PAN,
    SweepStrip,
)
from robocar.ranging import LIDAR_SAMPLE_INTERVAL_SECONDS, SCAN_SAMPLE_INTERVAL_SECONDS
from robocar.watch import Watcher

if TYPE_CHECKING:
    from robocar.mapping import MappingController

logger = logging.getLogger(__name__)

# The hunt waits this long after pointing the head before a picture or
# ping, plus a little per degree turned (the servo takes about 0.2 s for 60).
HUNT_HEAD_SETTLE_SECONDS = 0.08
HUNT_HEAD_SECONDS_PER_DEGREE = 0.0025
# The car counts as still (the hunt's camera may then use longer exposures)
# this long after its last move or head turn.
HUNT_STILL_SECONDS = 0.3
# A look waits at most this long for the next frame of the camera feed.
HUNT_LOOK_TIMEOUT_SECONDS = 1.0
# While hunting, the car tries to place itself from a still frame of its
# camera (matched to remembered places) this often.
HUNT_PLACE_FIX_SECONDS = 1.5
# ...using frames no older than this.
HUNT_PLACE_FIX_FRAME_SECONDS = 1.0


class CarHuntBody:
    """The car as the cat hunt sees it (robocar.hunt.HuntBody): its eyes are
    a Watcher running detection on the camera feed the whole time."""

    def __init__(self, controller: "MappingController") -> None:
        self._controller = controller
        status = controller._pan_tilt.status()
        self._tilt = int(status["tilt"])
        self._pan = int(status["pan"])
        self.status: dict[str, object] = {}
        # When the car or head last moved (the watcher allows longer
        # exposures once both have been still a moment).
        self._lock = threading.Lock()
        self._still_since = time.monotonic()
        self._busy = 0
        camera = controller._hunt_camera
        grab = getattr(camera, "capture_frame", None)
        if grab is None:

            def grab() -> np.ndarray | None:
                jpeg = camera.capture_snapshot()
                return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if jpeg else None

        detect = getattr(controller._detector, "detect", None)
        self.watcher = Watcher(
            grab,
            detect if detect is not None else (lambda frame: []),
            # Where the camera looks: with the head, where the servo really
            # points (sweeps move it directly); fixed, always ahead.
            (lambda: int(controller._pan_tilt.status()["pan"])) if controller.hunt_camera_on_head else (lambda: STRIP_CENTRE_PAN),
            controller._drive.odometry,
            self._moving,
            publish=lambda frame, detections: controller._publish_hunt_frame(frame, detections, self.status),
            set_exposure=getattr(camera, "set_exposure", None),
            interrupt=controller._interrupt,
        )

    def start(self) -> None:
        self.watcher.start()
        self._placing = threading.Event()
        self._placer = threading.Thread(target=self._place_loop, name="hunt-placer", daemon=True)
        self._placer.start()

    def close(self) -> None:
        placing = getattr(self, "_placing", None)
        if placing is not None:
            placing.set()
            self._placer.join(timeout=5)
        self.watcher.stop()

    def _place_loop(self) -> None:
        """Now and then, while the car stands still, place it on the map from
        what its camera sees, where two or more remembered places agree."""
        last_tried = -math.inf
        while not self._placing.wait(HUNT_PLACE_FIX_SECONDS):
            try:
                self.try_place_fix(last_tried)
                still = self.watcher.still_frame()
                if still is not None:
                    last_tried = max(last_tried, still[0].at)
            except Exception:  # placing is a bonus: never let it stop the hunt
                logger.warning("Placing the car from its camera failed", exc_info=True)

    def try_place_fix(self, after: float = -math.inf) -> dict[str, object] | None:
        still = self.watcher.still_frame()
        if still is None:
            return None
        sighting, frame = still
        if sighting.at <= after or time.monotonic() - sighting.at > HUNT_PLACE_FIX_FRAME_SECONDS:
            return None
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            return None
        camera_bearing = pan_bearing(sighting.pan) if self._controller.hunt_camera_on_head else 0.0
        return self._controller.place_fix_from_picture(jpeg.tobytes(), camera_bearing, sighting.odometry)

    def now(self) -> float:
        return time.monotonic()

    @contextmanager
    def _acting(self):
        with self._lock:
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1
                self._still_since = time.monotonic()

    def _moving(self) -> bool:
        drive = self._controller._drive.status()
        if any(abs(float(drive[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")):
            return True
        with self._lock:
            return self._busy > 0 or time.monotonic() - self._still_since < HUNT_STILL_SECONDS

    def wait(self, seconds: float) -> bool:
        return not self._controller._stop_event.wait(max(0.0, seconds))

    def stopped(self) -> bool:
        return self._controller._stop_event.is_set()

    def _point_head(self, pan: int) -> bool:
        controller = self._controller
        pan = max(PAN_LIMITS[0], min(PAN_LIMITS[1], int(pan)))
        if pan == self._pan:
            return True
        # Turning the head moves the camera only when the camera rides on it;
        # a fixed camera keeps its still frames and long exposures meanwhile
        # (the head turns often to aim the lidar at the cat).
        with self._acting() if controller.hunt_camera_on_head else contextlib.nullcontext():
            controller._pan_tilt.set_inhibited(True)
            try:
                controller._pan_tilt.set_angles(pan, self._tilt, override=True)
            finally:
                controller._pan_tilt.set_inhibited(False)
            turned = abs(pan - self._pan)
            self._pan = pan
            return not controller._stop_event.wait(HUNT_HEAD_SETTLE_SECONDS + HUNT_HEAD_SECONDS_PER_DEGREE * turned)

    # -- seeing: the watcher's feed --------------------------------------

    def look(self, pan: int) -> Look:
        """Point the head and take what the next frame shows."""
        if not self._point_head(pan):
            return Look(None)
        sighting = self.watcher.wait_for(time.monotonic(), HUNT_LOOK_TIMEOUT_SECONDS)
        return sighting.look if sighting is not None else Look(None)

    def camera_pans(self) -> bool:
        return self._controller.hunt_camera_on_head

    def sighting(self) -> FeedSighting | None:
        return self.watcher.latest()

    def interrupt_on_cat(self, enabled: bool) -> None:
        self.watcher.set_interrupts(enabled)

    def spotted(self) -> FeedSighting | None:
        return self.watcher.spotted()

    def clear_spotted(self) -> None:
        self.watcher.clear_spotted()

    # -- ranging ------------------------------------------------------------

    def ping(self, pan: int, precise: bool = False) -> float | None:
        if not self._point_head(pan):
            return None
        return self._ping_now(precise)

    def ranges_precisely(self) -> bool:
        return self._controller._ranging.lidar_ready()

    def turn_power(self):
        """The car's learned turn power, shared with mapping (and kept)."""
        return self._controller._turn_power

    def range_ahead(self, precise: bool = False) -> float | None:
        """The latest range straight ahead: no waiting once the head faces
        ahead (both sensors keep measuring). The nearer of lidar and sonar,
        or with `precise` the lidar's narrow beam (to measure the cat)."""
        controller = self._controller
        if self._pan != 90:
            if not self._point_head(90):
                return None
            if controller._stop_event.wait(max(0.0, controller._ranging.settle_seconds(precise) - HUNT_HEAD_SETTLE_SECONDS)):
                return None
        return controller._ranging.now(precise)

    def _ping_now(self, precise: bool = False) -> float | None:
        controller = self._controller
        if controller._stop_event.wait(max(0.0, controller._ranging.settle_seconds(precise) - HUNT_HEAD_SETTLE_SECONDS)):
            return None
        readings = []
        for _ in range(3):
            reading = controller._ranging.now(precise)
            if reading is not None:
                readings.append(reading)
            if controller._stop_event.wait(SCAN_SAMPLE_INTERVAL_SECONDS if not controller._ranging.lidar_ready() else LIDAR_SAMPLE_INTERVAL_SECONDS):
                return None
        return statistics.median(readings) if readings else None

    def observe(self) -> None:
        """A sonar sweep to find its place on the map; the watcher keeps
        looking throughout (and a cat cuts it short)."""
        controller = self._controller
        self._point_head(STRIP_CENTRE_PAN)
        if not controller._drive.set_inhibited(True):
            raise RuntimeError("Drive was not stopped before a range scan")
        self.watcher.collect(True)
        try:
            with self._acting():
                controller._capture_observation(mode="hunt")
        finally:
            controller._drive.set_inhibited(False)
            collected = self.watcher.collect(False)
        self._pan = int(controller._pan_tilt.status()["pan"])
        if len(collected) >= 2:
            controller._show_panorama(SweepStrip({pan: (picture,) for pan, picture in collected.items()}, STRIP_CENTRE_PAN, self._tilt, (0.0, 0.0, 0.0)))

    def knows_where_it_is(self) -> bool:
        return bool(self._controller.localized()["known"])

    def pose(self) -> Pose:
        controller = self._controller
        with controller._lock:
            pose, last = controller._pose, controller._last_odometry
        return compose(pose, relative(last, controller._drive.odometry())) if last else pose

    def odometry(self) -> Pose:
        return self._controller._drive.odometry()

    def grid(self) -> CoverageGrid:
        return self._controller._grid

    def landmarks(self) -> list[tuple[str, float, float]]:
        with self._controller._map_lock:
            return [(landmark.label, landmark.x, landmark.y) for landmark in self._controller._semantic.confirmed()]

    # -- moving ---------------------------------------------------------------

    def turn(self, degrees: float, precise: bool) -> bool:
        controller = self._controller
        if abs(degrees) < 1.0:
            return not self.stopped()
        with self._acting():
            if precise:
                pieces = max(1, math.ceil(abs(degrees) / AUTO_TURN_PIECE_DEGREES - 1e-9))
                return all(controller._turn_and_align(degrees / pieces) for _ in range(pieces))
            return controller._turn_by(degrees, correct=False)

    def move(self, forward_cm: float, sideways_cm: float, speed: float = 1.0) -> bool:
        distance = math.hypot(forward_cm, sideways_cm)
        if distance < 1.0:
            return not self.stopped()
        controller = self._controller
        speed = max(0.3, min(1.0, speed))
        if abs(sideways_cm) <= abs(forward_cm) * 0.05:
            seconds = controller._step_seconds(distance)
        else:
            # Sideways travel is slower and has no calibration of its own.
            seconds = AUTO_FORWARD_SECONDS * distance / AUTO_STEP_CM * (1.0 + 0.3 * abs(sideways_cm) / distance)
        throttle = AUTO_FORWARD_COMMAND * speed * forward_cm / distance
        strafe = AUTO_FORWARD_COMMAND * speed * sideways_cm / distance
        with self._acting():
            return controller._move_burst(throttle, seconds / speed, strafe=strafe)

    def arc(self, strafe: float, steering: float, seconds: float) -> bool:
        with self._acting():
            return self._controller._move_burst(0.0, seconds, steering=steering, strafe=strafe)

    def command(self, throttle: float, strafe: float, steering: float) -> None:
        """Keep moving like this (the drive's watchdog stops the car if the
        hunt stops renewing the command)."""
        self._controller._drive.set_command(throttle, strafe, steering)
        with self._lock:
            self._still_since = time.monotonic()

    def halt(self) -> None:
        drive = self._controller._drive
        drive.stop()
        if not drive.wait_until_stopped():
            raise RuntimeError("Drive output did not stop after the ramp-down interval")
        with self._lock:
            self._still_since = time.monotonic()

    def battery_ok(self) -> bool:
        level = self._controller.battery_level() if self._controller.battery_level else None
        return level != "critical"

    def report(self, status: dict[str, object]) -> None:
        self.status = status
        controller = self._controller
        with controller._lock:
            controller._hunt_status = status
            controller._hunt_camera_status = {**self.watcher.status(), "place_fixes": dict(controller.place_fixes)}
            controller._plan = str(status.get("action") or "")
            if controller._enabled:
                controller._state = f"hunt_{status.get('phase')}"


class CarCalibrationRig:
    """The car as calibration sees it: turns, bursts, pictures and ranges."""

    def __init__(self, controller: "MappingController") -> None:
        self._controller = controller
        status = controller._pan_tilt.status()
        self._centre, self._tilt = int(status["pan"]), int(status["tilt"])
        self._origin: Pose | None = None

    def heading_odometry(self) -> float:
        return self._controller._drive.odometry()[2]

    def distance_odometry(self) -> float:
        odometry = self._controller._drive.odometry()
        if self._origin is None:
            self._origin = odometry
        return relative(self._origin, odometry)[1]

    def turn(self, degrees: float, rotation_scale: float) -> bool:
        return self._controller._turn_by(degrees, rotation_scale)

    def burst(self, throttle: float, seconds: float) -> bool:
        return self._controller._move_burst(throttle, seconds)

    def photograph(self, pan_offsets: Sequence[int]) -> dict[int, tuple[bytes, ...]]:
        # Offsets are counterclockwise of where the head faces.
        pans = [self._centre - offset for offset in pan_offsets if PAN_LIMITS[0] <= self._centre - offset <= PAN_LIMITS[1]]
        views = self._controller._capture_views(pans, self._tilt, self._centre, frames=1 if len(pans) > 1 else AUTO_ALIGN_FRAMES)
        return {self._centre - pan: frames for pan, frames in views.items()}

    def measure_turn(self, strip: dict[int, tuple[bytes, ...]], current: tuple[bytes, ...] | None) -> float | None:
        return self._controller._measure_turn({self._centre - offset: frames for offset, frames in strip.items()}, current, self._centre)

    def last_move_cm(self) -> float | None:
        with self._controller._lock:
            move = self._controller._last_move
        return move.get("ahead_cm") if move else None

    def range_ahead(self) -> float | None:
        controller = self._controller
        controller._pan_tilt.set_inhibited(True)
        try:
            controller._pan_tilt.set_angles(self._centre, self._tilt, override=True)
            # Let a running-median sensor forget readings from before the move.
            controller._stop_event.wait(max(0.06, controller._ranging.settle_seconds(precise=True)))
            # The lidar measures the wall ahead to a centimetre or two.
            return controller._ranging.now(precise=True)
        finally:
            controller._pan_tilt.set_inhibited(False)

    def clearance_all_round(self) -> float | None:
        scan, _ = self._controller._sweep(self._tilt, self._centre, ())
        distances = [float(item["distance_cm"]) for item in scan or [] if item["distance_cm"] is not None]
        return min(distances) if distances else None

    def stopped(self) -> bool:
        return self._controller._stop_event.is_set()

    def report(self, message: str) -> None:
        with self._controller._lock:
            self._controller._plan = message
