"""Mapping: the controller that runs the car's mapping, exploring, hunting
and calibration sessions, and keeps the map.

A session runs on its own thread. At each stop the car sweeps the head's
range sensors (the lidar every degree when fitted, else the sonar every 3),
takes pictures, works out how it moved since the last stop (wheels, camera
views, and fitting the sweep to the map), and adds the sweep to the pose
graph, the surface map, the coverage grid and the map of what the cameras
recognized. Exploring decides where to stop next, in one of two ways:
"frontier" (a sweep ahead, then a move toward the most promising unexplored
edge) or "spin" (a full circle of sweeps, then a drive to the open place a
full turn would show most from).

The pieces it uses:
- ranging: the head's sonar and lidar together, and how sweeps are paced
- pictures: judging stop pictures and measuring turns between views
- explore_plan: the frontier explorer's next move, and way-ahead checks
- exploration: the coverage grid (frontiers, routes, spin stations, matching)
- surfaces: walls and features from sweeps; localization: poses and fusing
- places: recognizing stops from pictures; the pose graph
- semantics: what the cameras recognized, placed on the map
- store: the SQLite database; pointcloud: lidar points for the dashboard
- bodies: the car as the hunt and calibration engines see it
- hunt, calibration, watch: those engines, and the hunt camera's watcher
"""

import copy
import json
import logging
import math
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from robocar.calibration import Calibration, CalibrationStore, run_calibration, seconds_for
from robocar.drive import (
    FORWARD_CM_PER_SECOND_AT_FULL_DUTY,
    MOTOR_DEADBAND_FRACTION,
    MOTOR_RAMP_SECONDS,
    TURN_DEGREES_PER_SECOND_AT_FULL_DUTY,
    PAN_LIMITS,
    CoastModel,
    DriveController,
    TurnPower,
    bearing_pan,
    pan_bearing,
    pan_scale,
    set_pan_scale,
)
from robocar.exploration import SPIN_AWAY_CM, CoverageGrid, load_learner, save_learner
from robocar.explore_plan import (
    AUTO_HEADING_TOLERANCE_DEGREES,
    AUTO_MIN_STEP_CM,
    AUTO_TURN_MIN_CLEARANCE_CM,
    AUTO_WALL_STANDOFF_CM,
    CELL_AREA_M2,
    direction_clearances,
    forward_clearance,
    hazard_within,
    is_lidar_scan,
    plan_explore_step,
)
from robocar.hunt import HuntEngine, load_memory, save_memory
from robocar.hunt import Sighting as FeedSighting
from robocar.localization import (
    MotionPrediction,
    Pose,
    VisualMotion,
    compose,
    pose_sigma,
    predict_motion,
    propagate,
    register_scan,
    relative,
    wrap_degrees,
)
from robocar.pictures import (
    EXPOSURE_LADDER,
    EXPOSURE_MAX_ATTEMPTS,
    EXPOSURE_START_INDEX,
    PictureQuality,
    ViewKey,
    _steadiness_frame,
    assess_picture,
    combine_view_estimates,
    estimate_view_motion,
    exposure_label,
    image_texture_score,
)
from robocar.places import (
    CAMERA_HORIZONTAL_FOV_DEGREES,
    PlaceMemory,
    PlaceRecognizer,
    locate_from_places,
    make_keyframe,
)
from robocar.range_sensor import RangeSensorController
from robocar.pointcloud import CLOUD_CELL_CM, PointCloud, lidar_points
from robocar.room_plan import MethodRecords
from robocar.room_plan import choose as choose_option
from robocar.room_plan import options as room_options
from robocar.ranging import (
    LIDAR_SAMPLE_INTERVAL_SECONDS,
    LIDAR_SAMPLE_SPREAD_CM,
    LIDAR_SAMPLES_PER_ANGLE,
    LIDAR_SCAN_ANGLES,
    LIDAR_STEP_SETTLE_SECONDS,
    SCAN_FIRST_SETTLE_SECONDS,
    SCAN_LEAD_IN_PAN,
    HeadRanging,
    estimate_pan_scale,
)
from robocar.store import MappingStore
from robocar.semantics import (
    CAMERA_HFOV_DEG,
    HEAD_CAMERA_HFOV_DEG,
    SemanticMap,
    Sighting,
    hazards_from,
    mark_dynamic,
    sightings_from,
    static_scan,
)
from robocar.surfaces import IntegrationReport, SurfaceMap, extract_echoes, pose_rotation, sweep_seen_from
from robocar.vision_debug import render_detections, render_features, render_lidar_overlay, render_matches, render_panorama
from robocar.watch import Watcher

logger = logging.getLogger(__name__)


def camera_hfov(camera: object) -> float:
    """How wide a camera sees, by its sensor: the OV5647 on the head is a
    little narrower than the HQ camera behind its 6 mm lens."""
    model = str(getattr(camera, "model", "") or "")
    return HEAD_CAMERA_HFOV_DEG if "ov5647" in model else CAMERA_HFOV_DEG



# Extra camera views taken during the sweep when the forward view cannot be
# trusted to measure motion.
SIDE_VIEW_PANS = (54, 126)
TEXTURE_POOR_SCORE = 120.0
SIDE_MIN_ANGLE_DEGREES = 30
STOP_SETTLE_SECONDS = 0.75
AUTO_MAX_SECONDS = 240
AUTO_MAX_MOVES = 60
EXPLORE_STRATEGIES = ("room", "frontier", "spin")
# Spin-and-roam: four sweeps a quarter turn apart make a full circle (each
# sweep sees 60 degrees either side, so neighbours overlap by 30 degrees and
# each can be fitted to the last). A quarter whose view the map already
# knows (fewer than SPIN_SKIP_GAIN_CELLS unknown cells) is skipped. Between
# stations it drives in legs, sweeping ahead every SPIN_SWEEP_EVERY_CM to
# stay placed, and has arrived within SPIN_ARRIVED_CM. Spinning takes time,
# so a session may run longer than a frontier one.
SPIN_QUARTERS = (90.0, 180.0, 270.0)
# Turning on the spot sweeps the car's corners round this far from its centre.
SPIN_TURN_RADIUS_CM = 20.0
SPIN_SKIP_GAIN_CELLS = 60.0
SPIN_SWEEP_EVERY_CM = 120.0
# With turns and distances measured (gyro and accelerometer) the wheels'
# guess is not what places the car, so it can go further between sweeps.
SPIN_SWEEP_EVERY_MEASURED_CM = 300.0
# Long drives are one burst, watched all the way (see _move_burst): stopped
# when anything ahead comes within the stand-off plus this (the car's
# stopping distance), when the car veers this far off its line, or when it
# has not been moving for this long although driving.
WATCH_STOP_MARGIN_CM = 15.0
WATCH_MAX_VEER_DEGREES = 20.0
WATCH_STALL_SECONDS = 1.5
WATCH_STALL_SPEED_CM_S = 3.0
TRAVEL_MAX_SECONDS = 8.0
# Bursts this short use the step timing (calibrated); longer ones the line
# through measured bursts, or the wheels' speed.
TRAVEL_FROM_STEPS_CM = 30.0
SPIN_ARRIVED_CM = 25.0
SPIN_MAX_LEGS = 12
SPIN_MAX_SECONDS = 900
SPIN_MAX_MOVES = 200
AUTO_FORWARD_COMMAND = 1.0
AUTO_FORWARD_SECONDS = 0.25
# With a motion calibration, each exploration step is sized to this distance.
AUTO_STEP_CM = 25.0
# The car believes it knows where it is on the map while: there is a map to
# match against; a sweep has matched it (or it recognized a place) since the
# service started; its pose estimate, which grows less certain with every
# move on wheel counts, is this certain; and it has driven no further than
# this on wheel counts since that last match. (Matches are rare with the
# real sonar, so recency is judged by distance, not by stops.)
LOCALIZED_MIN_WALLS = 2
# ...or this many obstacle cells on the coverage map.
LOCALIZED_MIN_OCCUPIED_CELLS = 40
LOCALIZED_MAX_SIGMA_CM = 30.0
LOCALIZED_MAX_HEADING_SIGMA_DEG = 12.0
LOCALIZED_MAX_DEAD_RECKONING_CM = 400.0
LOCALIZED_MAX_DEAD_RECKONING_DEG = 540.0
# While mapping with a separate forward camera, the cat seen in its feed is
# remembered (for the hunt) at most this often.
MAPPING_CAT_EVERY_SECONDS = 5.0
AUTO_TURN_COMMAND = 1.0
AUTO_TURN_PIECE_DEGREES = 50.0
AUTO_MAX_TURN_SECONDS = 3.0
# With a gyro, turns are measured as they happen. After each turn the car
# waits for it to settle, learns how far it coasted after the motors were
# cut, and corrects anything left over beyond the tolerance with gentler
# turns. A turn that has not got going after a while is stuck (a wheel
# against something) and is given up.
GYRO_TURN_TOLERANCE_DEGREES = 1.5
GYRO_TURN_MAX_CORRECTIONS = 3
# Gyro turns ask for a turn rate and get the power for it from what the gyro
# has seen (drive.TurnPower): full power spins the car at hundreds of degrees
# a second within a quarter second, far too fast to stop on a small turn,
# while too little does not turn it at all. Bigger turns go faster; the
# corrections afterwards slowly. A turn the gyro sees not starting gets more
# power, a step at a time.
GYRO_TURN_RATE_DPS = 150.0
GYRO_BIG_TURN_RATE_DPS = 220.0
GYRO_BIG_TURN_DEGREES = 60.0
GYRO_CORRECTION_RATE_DPS = 50.0
GYRO_STALL_POWER_STEP = 0.15
# A turn this long has reached its speed: its peak rate shows what its power does.
GYRO_LEARN_POWER_SECONDS = 0.6
# After a gyro turn the car is settled once it turns less than this in a
# tenth of a second (waiting at most the longer time).
GYRO_SETTLED_DEGREES = 0.2
GYRO_SETTLE_MAX_SECONDS = 0.6
GYRO_STALL_SECONDS = 0.6
GYRO_STALL_DEGREES = 3.0
# Rotation scales learned while mapping are kept within these bounds: the
# wheels' guess may be far out, a gyro's scale only by a few percent.
WHEEL_ROTATION_SCALE_LIMITS = (0.5, 2.0)
GYRO_ROTATION_SCALE_LIMITS = (0.9, 1.1)
# A move ends early when the IMU feels a knock this hard (in g), after its
# start: on the car a full-power start alone measured 1.4-1.5 g, as did
# stopping, so only a real crash goes past this. Knocks over KNOCK_G are
# still reported, for tuning.
BUMP_STOP_G = 2.5
BUMP_IGNORE_START_SECONDS = 0.5
# The learned coast is saved after this many more measured turns.
COAST_SAVE_EVERY = 5
# After a burst the car is given this long to come to rest before the
# accelerometer's sum is closed; the latest measured forward bursts (this
# many) time the next ones, once there are enough of different lengths.
MOVE_SETTLE_SECONDS = 0.15
BURSTS_KEPT = 30
BURSTS_NEEDED = 4
# Translation scales learned while mapping stay within these: the wheels'
# guess may be far out, measured moves only by a little.
WHEEL_TRANSLATION_SCALE_LIMITS = (0.5, 2.0)
MEASURED_TRANSLATION_SCALE_LIMITS = (0.85, 1.15)
# Spin and roam: a station's sweeps, joined into one full circle by the
# gyro's turns, are fitted to the map as it was before the station (a whole
# circle fits far more surely than a quarter) and the station is tied to the
# stop before it by the result, trusted to about this much.
CIRCLE_FIT_MIN_SWEEPS = 3
CIRCLE_FIT_SIGMA = (3.0, 3.0, 1.0)
# With only the lidar (its beam is 2 degrees wide), the way ahead is checked
# with readings this far either side of straight ahead, this far apart: a
# car-wide corridor 45 cm out is about 18 degrees either side.
LIDAR_FAN_HALF_DEGREES = 20
LIDAR_FAN_STEP_DEGREES = 2
LIDAR_FAN_SETTLE_SECONDS = 0.03
# The pan servo's scale is learned from lidar sweeps either side of a gyro
# turn on the spot (moved at most this far, turned this much), blended in
# this much at a time.
PAN_SCALE_MAX_MOVE_CM = 2.0
PAN_SCALE_TURN_RANGE_DEG = (15.0, 100.0)
PAN_SCALE_BLEND = 0.3
# Before each turn the camera photographs a strip across the pan, from
# straight ahead to past the destination. Afterwards the new forward view is
# matched against every strip picture to measure the turn actually made, and
# small corrective turns remove the remaining error.
AUTO_STRIP_STEP_DEGREES = 10
# The full pan range on the turn side, so even a large overshoot is measured.
AUTO_STRIP_REACH_DEGREES = 60
# The strip and its panorama are centred on straight ahead.
STRIP_CENTRE_PAN = 90
AUTO_STRIP_MAX_YAW_DEGREES = 25.0
AUTO_ALIGN_TOLERANCE_DEGREES = 3.0
AUTO_ALIGN_MAX_CORRECTIONS = 2
# A measured turn must be backed by two strip pictures that agree, must be
# roughly what the wheels saw, and may only be corrected by so much at once;
# otherwise the wheels are trusted.
AUTO_STRIP_AGREEMENT_DEGREES = 4.0
AUTO_ALIGN_PLAUSIBLE_RATIO = (0.5, 2.0)
AUTO_ALIGN_MAX_CORRECTION_DEGREES = 20.0
AUTO_ALIGN_FRAMES = 2
# While exploring, the strip pictures for the first turn piece are taken
# during the stop's sonar sweep (which passes every pan anyway) rather than in
# a second pass of the head. They are only reused while the car and head are
# still where they were: within this much movement.
SWEEP_STRIP_MAX_MOVE_CM = 1.0
SWEEP_STRIP_MAX_TURN_DEG = 1.0
# Strip pictures are taken at the sweep's nearest step (it steps every 3
# degrees) rather than adding stops of their own; the turn check uses each
# picture's actual angle, so being off by up to this much costs nothing.
SWEEP_STRIP_SNAP_DEG = 1.5
PANORAMA_HFOV_DEGREES = CAMERA_HORIZONTAL_FOV_DEGREES
AUTO_ALIGN_SETTLE_SECONDS = 0.15
AUTO_VIEW_STEADY_TIMEOUT_SECONDS = 1.0
CAMERA_MOTION_FRAME_COUNT = 3
# When the pose graph moves any stop by more than this, the map is rebuilt
# from every sweep at its corrected pose.
MAP_REBUILD_MOVE_CM = 4.0
# A new run carries on the map from the last stop. The car may have been
# moved by hand in between, so its first stop starts this uncertain and is
# tied to the last stop only this loosely; its first sweep against the known
# walls (or a recognized place) pins it down.
CONTINUE_POSITION_SIGMA_CM = 25.0
# A run started while the service has kept tracking the wheels since the
# last stop keeps its pose and certainty, but at least this uncertain: the
# car may have been lifted and set down a little way off.
CONTINUE_SAME_SESSION_SIGMA_CM = 10.0
CONTINUE_SAME_SESSION_HEADING_SIGMA_DEG = 5.0
# Fitting a sweep to the coverage map when wall registration fails: how far
# to search (twice the pose's uncertainty, within these bounds) and how sure
# a good fit leaves the pose.
GRID_MATCH_MIN_REACH_CM = 12.0
GRID_MATCH_MAX_REACH_CM = 32.0
GRID_MATCH_MIN_REACH_DEG = 8.0
GRID_MATCH_MAX_REACH_DEG = 24.0
GRID_MATCH_SIGMA_CM = 12.0
GRID_MATCH_HEADING_SIGMA_DEG = 5.0
GRID_MATCH_COVARIANCE = np.diag(
    [GRID_MATCH_SIGMA_CM**2, GRID_MATCH_SIGMA_CM**2, math.radians(GRID_MATCH_HEADING_SIGMA_DEG) ** 2]
)
# A lidar sweep fits far tighter: in simulated rooms trusted fits landed
# within 2 cm and 1 degree RMS (worst 4 cm, 3 degrees).
LIDAR_GRID_MATCH_COVARIANCE = np.diag([5.0**2, 5.0**2, math.radians(2.0) ** 2])
# A fit is combined with the wheel-based estimate, weighted by how sure each
# is, and dropped when the two disagree beyond what both allow (99% of a
# chi-squared with 3 degrees of freedom): a rare trusted fit still comes out
# badly twisted.
GRID_MATCH_GATE = 11.34
# Placing the car from the hunt camera's live picture (matched to two or
# more remembered places whose directions toward it cross): how sure the
# position and heading are (tests: within 10 cm and 3 degrees).
CAMERA_FIX_HEADING_SIGMA_DEG = 3.0
CAMERA_FIX_POSITION_SIGMA_CM = 20.0
# From only two places nothing cross-checks them: counted less sure (in
# tests one in eleven such fixes was off by about 25 cm and 6 degrees).
CAMERA_FIX_TWO_PLACES_HEADING_SIGMA_DEG = 6.0
CAMERA_FIX_TWO_PLACES_POSITION_SIGMA_CM = 35.0
# Between stops the pose is dead-reckoned and grows less certain: per cm
# driven, about this share in position and this many degrees of heading,
# and per degree turned, this share of a degree.
DRIFT_POSITION_PER_CM = 0.1
DRIFT_HEADING_DEG_PER_CM = 0.02
DRIFT_HEADING_PER_DEGREE = 0.05
CONTINUE_HEADING_SIGMA_DEG = 20.0
MAP_REBUILD_TURN_DEG = 2.0
VIEW_HEADING_SIGMA_AFTER_MATCH = 3.0
# Before keeping a picture, take quick frames until two in a row match, so
# the car and head have stopped moving.
CAMERA_STEADY_DIFFERENCE = 2.5
CAMERA_STEADY_TIMEOUT_SECONDS = 2.0
CAMERA_STEADY_INTERVAL_SECONDS = 0.1
CAMERA_NOISE_GROWTH = 1.15
# Let the car stop rocking and the head stop moving before the first picture.
CAMERA_SETTLE_SECONDS = 0.3
CAMERA_MOTION_FRAME_INTERVAL_SECONDS = 0.05


class SnapshotCamera(Protocol):
    def capture_snapshot(self) -> bytes | None: ...


class ObjectDetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> list[dict[str, object]]: ...


def fuse_pose(
    prior: Pose, prior_covariance: np.ndarray, measured: Pose, measured_covariance: np.ndarray, gate: float
) -> tuple[Pose, np.ndarray] | None:
    """The prior pose corrected by a measurement of it (a Kalman update),
    or None when they disagree by more than `gate` (squared Mahalanobis)."""
    innovation = np.array(
        [measured[0] - prior[0], measured[1] - prior[1], math.radians(wrap_degrees(measured[2] - prior[2]))]
    )
    combined = prior_covariance + measured_covariance
    try:
        distance = float(innovation @ np.linalg.solve(combined, innovation))
        gain = prior_covariance @ np.linalg.inv(combined)
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(distance) or distance > gate:
        return None
    step = gain @ innovation
    pose = (prior[0] + float(step[0]), prior[1] + float(step[1]), prior[2] + math.degrees(float(step[2])))
    covariance = (np.eye(3) - gain) @ prior_covariance
    return pose, (covariance + covariance.T) / 2.0


@dataclass(frozen=True)
class SweepStrip:
    """Turn-measurement pictures taken during a stop's sweep, and where from."""

    pictures: dict[int, tuple[bytes, ...]]  # by pan
    centre_pan: int
    tilt: int
    odometry: Pose


class PanTilt(Protocol):
    def set_angles(
        self,
        pan: int,
        tilt: int,
        *,
        override: bool = False,
    ) -> tuple[int, int]: ...

    def set_inhibited(self, inhibited: bool) -> None: ...

    def status(self) -> dict[str, object]: ...



def _guess_step_seconds(distance_cm: float) -> float:
    """Burst length expected to cover a distance, before any calibration.

    A burst ramps up for its length and down for MOTOR_RAMP_SECONDS after; at
    full duty that covers roughly speed x (burst + ramp) / 2 for short bursts.
    """
    speed = FORWARD_CM_PER_SECOND_AT_FULL_DUTY * (1 - MOTOR_DEADBAND_FRACTION)
    return max(0.05, min(0.8, 2 * distance_cm / speed - MOTOR_RAMP_SECONDS))


def _load_turning(path) -> dict[str, object]:
    """What turning and moving have learned (see MappingController._save_learned)."""
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable turning file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _coast_from(data: dict[str, object], path) -> CoastModel:
    try:
        return CoastModel.from_json(data.get("coast", data))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        logger.info("Turn coast in %s not used: %s", path, exc)
        return CoastModel()


class MappingController:
    def __init__(
        self,
        camera: SnapshotCamera,
        drive: DriveController,
        pan_tilt: PanTilt,
        range_sensor: RangeSensorController,
        store: MappingStore,
        detector: "ObjectDetectorLike | None" = None,
        camera_on_head: bool = True,
        hunt_camera: "SnapshotCamera | None" = None,
        lidar: RangeSensorController | None = None,
    ) -> None:
        self._camera = camera
        # The hunt watches through this one (the forward camera when there
        # are two); mapping and its sweeps use `camera`.
        self._hunt_camera = hunt_camera if hunt_camera is not None else camera
        # A camera on the pan head turns with it (picture strips measure
        # turns, side views, the sweep panorama); a fixed one always looks
        # straight ahead, whichever way the head points.
        self.camera_on_head = camera_on_head
        # Recognizes objects in each stop's picture (on the AI HAT+ when present).
        self._detector = detector
        self._semantic = SemanticMap()
        # People and animals seen at the latest stop, as (bearing, range).
        self._hazards: list[tuple[float, float | None]] = []
        # Turn-strip pictures from the latest sweep, until a turn uses them.
        self._sweep_strip: SweepStrip | None = None
        self._drive = drive
        self._pan_tilt = pan_tilt
        self._range_sensor = range_sensor
        # The lidar on the head (None without one): the sweeps' ranges and
        # precise single ranges. The sonar keeps watching too, for glass and
        # anything else light passes or misses.
        self._lidar = lidar
        self._ranging = HeadRanging(range_sensor, lidar)
        self._store = store
        self._lock = threading.Lock()
        self._map_lock = threading.Lock()
        self._stop_event = threading.Event()
        # Set by the hunt's camera watcher when it sees the cat while
        # seeking: turns, moves and sweeps end early (it is not a stop).
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._state = "idle"
        self._error: str | None = None
        latest = self._store.latest()
        self._last_id = latest[0] if latest else None
        self._last_observation = latest[1] if latest else None
        self._run_id: str | None = None
        self._mode = "idle"
        self._move_count = 0
        self._plan: str | None = None
        store_path = getattr(self._store, "path", None)
        self._calibration_store = CalibrationStore(
            store_path.parent / "calibration.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._calibration: Calibration | None = self._calibration_store.load()
        # What exploring has learned (how much each kind of frontier really
        # shows, and what effort buys), kept across runs beside the database.
        self._learner_path = (
            store_path.parent / "exploration.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._learner = load_learner(self._learner_path)
        # The cat hunt: what it has learned, its latest status and camera frame.
        self._hunt_memory_path = (
            store_path.parent / "hunt.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._hunt_memory = load_memory(self._hunt_memory_path)
        # How turns coast, learned from the gyro, kept beside the database.
        self._coast_path = (
            store_path.parent / "turning.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        learned = _load_turning(self._coast_path)
        self._coast = _coast_from(learned, self._coast_path)
        try:
            self._turn_power = TurnPower.from_json(learned["turn_power"]) if learned.get("turn_power") else TurnPower()
        except (ValueError, TypeError, KeyError) as exc:
            logger.info("Turn power in %s not used: %s", self._coast_path, exc)
            self._turn_power = TurnPower()
        # The pan servo's scale: kept with the car's own servo only.
        self._pan_scale_learned: float | None = None
        self._pan_scale_samples = 0
        saved_scale = learned.get("pan_scale")
        if isinstance(saved_scale, (int, float)) and self._pan_tilt.status()["mode"] == "hardware":
            self._pan_scale_learned = set_pan_scale(saved_scale)
        # Moves measured by the accelerometer: the board's forward direction,
        # and (seconds, centimetres) of forward bursts, for timing the next.
        imu = self._drive.imu
        forward = learned.get("forward_axis")
        if imu is not None and isinstance(forward, list) and len(forward) == 3:
            imu.set_forward_axis(forward)
        boost = learned.get("learned_boost")
        if isinstance(boost, (int, float)):
            self._drive.set_learned_boost(boost)
        self._bursts: list[tuple[float, float]] = [
            (float(seconds), float(cm)) for seconds, cm in learned.get("bursts", []) if isinstance(seconds, (int, float))
        ][-BURSTS_KEPT:]
        self._last_move: dict[str, object] | None = None
        self._move_abort: str | None = None
        # Room mapping: what each way of mapping has delivered, and its latest choices.
        try:
            self._records = MethodRecords.from_json(dict(learned.get("mapping_methods") or {}))
        except (TypeError, ValueError):
            self._records = MethodRecords()
        self._room_choices: list[dict[str, object]] = []
        self._last_scan: list[dict[str, object]] | None = None
        self._hunt_status: dict[str, object] | None = None
        self._hunt_frame: bytes | None = None
        self._hunt_frame_at: float = 0.0
        # The hunt camera's learned settle delay and how many pictures it retook.
        self._hunt_camera_status: dict[str, object] = {}
        # Set by the web app: "ok", "getting_low", "low", "critical" or None.
        self.battery_level: Callable[[], str | None] | None = None
        self._reset_localization()
        self._surface_map = SurfaceMap()
        self._grid = CoverageGrid()
        self._cloud = PointCloud()
        self._explore_target: tuple[float, float] | None = None
        self._path: list[Pose] = []
        self._history: list[list[Pose]] = []
        self._links: list[tuple[Pose, Pose]] = []
        self._map_run_id: str | None = None
        # The place memory is only used from the mapping thread (and at start
        # up); the dashboard reads the snapshots kept below instead.
        self._places_lock = threading.Lock()
        self._places = PlaceMemory(self._store)
        self._places.load()
        # The forward camera's place pictures, for placing the car by what
        # the hunt's camera sees.
        self._forward_places = PlaceRecognizer()
        for keyframe in self._store.forward_keyframes():
            self._forward_places.add(replace(keyframe, hfov_deg=camera_hfov(self._hunt_camera)))
        self.place_fixes: dict[str, object] = {"tried": 0, "used": 0, "refused": 0, "last": None}
        # The forward camera watched while mapping, for the cat.
        self._mapping_watcher: Watcher | None = None
        self._mapping_cats = 0
        self._last_mapping_cat = -math.inf
        self._place_counts = self._places.counts()
        self._last_place: dict[str, object] | None = None
        # The exposure that worked best last time is where the next picture starts.
        self._exposure_index = EXPOSURE_START_INDEX
        self._vision: dict[str, dict[str, object]] = {}
        # Exploring: how ("frontier" or "spin"), and for spin-and-roam where
        # it has turned full circles this run and how the current one goes.
        self._strategy = "frontier"
        self._stations: list[tuple[float, float]] = []
        self._spin_progress: dict[str, object] | None = None
        self._replay_latest_run()

    def _reset_localization(self) -> None:
        self._pose: Pose = (0.0, 0.0, 0.0)
        self._pose_covariance = np.zeros((3, 3))
        self._last_odometry: Pose | None = None
        self._last_camera_views: dict[ViewKey, tuple[bytes, ...]] | None = None
        # Start from what calibration measured; each session keeps refining it.
        # The rotation scale is for whatever the heading follows now (the
        # gyro or the wheels), and starts again should that change.
        self._rotation_source = self._drive.heading_source()
        # Likewise distances: measured by the accelerometer, or the wheels'.
        self._translation_source = "accelerometer" if self._drive.measures_distance() else "wheels"
        self._odometry_scale = (
            self._initial_translation_scale(self._translation_source),
            self._initial_rotation_scale(self._rotation_source),
        )
        # How far turns coast after the motors are cut (learned from the
        # gyro, and kept from session to session while the service runs).
        self._coast = getattr(self, "_coast", None) or CoastModel()
        self._last_bump: dict[str, object] | None = None
        self._last_localization: dict[str, object] | None = None
        # Camera-verified rotation since the last stop, for turns too large
        # for stop-to-stop image comparison.
        self._verified_turn: float | None = None
        self._last_node: int | None = None
        self._joining_run = False
        # Stops since the pose was last pinned to the map (a sonar sweep
        # matched to mapped walls, or a recognized place); None if never.
        self._stops_since_fix: int | None = None
        # Wheel travel (cm, degrees) between the last fix and the last stop.
        self._travel_since_fix = (0.0, 0.0)

    def start(self) -> None:
        self._start("survey", self._run_survey)

    def start_autonomous(self, attended: bool, strategy: str = "room") -> None:
        """Explore on its own (attended). "room" (the main way): turn a full
        circle on the spot sweeping, drive straight across to the farthest
        open place it showed, and spin again; frontier steps where there is
        nowhere to spin. "frontier": only frontier steps (a sweep ahead, then
        a move toward the most promising unexplored edge). "spin": spin and
        roam without the frontier steps (see _run_spin_explore)."""
        if strategy not in EXPLORE_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(EXPLORE_STRATEGIES)}")
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before exploring")
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
        if drive_status["mode"] != "hardware":
            raise RuntimeError("Autonomous exploration requires hardware mode")
        if any(
            abs(float(drive_status[axis])) > 0.01
            for axis in ("throttle", "strafe", "steering")
        ):
            raise RuntimeError("Stop the car before starting autonomous exploration")
        if not self._ranging.lidar_ready():
            raise RuntimeError("Mapping needs the lidar working (sweeps are lidar sweeps)")
        with self._lock:
            self._strategy = strategy
        runners = {
            "room": self._run_room_mapping,
            "spin": self._run_spin_explore,
            "frontier": self._run_autonomous,
        }
        self._start("explore", runners[strategy])

    def start_hunt(self, attended: bool) -> None:
        """Hunt for the cat in the mapped room and play with it (attended)."""
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before the hunt")
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
        if drive_status["mode"] != "hardware":
            raise RuntimeError("The hunt requires hardware mode")
        if any(abs(float(drive_status[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")):
            raise RuntimeError("Stop the car before starting the hunt")
        if not self._ranging.lidar_ready():
            raise RuntimeError("The hunt needs the lidar working")
        if self._detector is None or not getattr(self._detector, "available", True):
            raise RuntimeError("The hunt needs object detection to spot the cat")
        if self._store.count() == 0:
            raise RuntimeError("Map the room first: the hunt plans its search on the map")
        self._start("hunt", self._run_hunt)

    def _run_hunt(self) -> None:
        from robocar.bodies import CarHuntBody  # bodies builds on this module

        self._hold_camera(True)
        self._interrupt.clear()
        body = CarHuntBody(self)
        # Its eyes: detection on the camera feed the whole time (and the
        # dashboard's live picture).
        body.start()
        try:
            self._drive.set_inhibited(False)
            engine = HuntEngine(body, self._hunt_memory)
            reason = engine.run()
            with self._lock:
                self._state = f"hunt_{reason}"
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("The hunt stopped after an error")
                with self._lock:
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            body.close()
            self._interrupt.clear()
            self._drive.stop()
            try:
                save_memory(self._hunt_memory_path, self._hunt_memory)
            except OSError as exc:
                logger.warning("Could not save what the hunt learned: %s", exc)
            # What its turns taught about turning power.
            self._save_learned()
            self._set_enabled_false()
            self._hold_camera(False)

    def _publish_hunt_frame(self, frame: np.ndarray, detections: list[dict[str, object]], status: dict[str, object]) -> None:
        """The hunt's latest picture for the dashboard: what it detected, and
        what it is up to."""
        try:
            image = frame.copy()
            for detection in detections:
                x, y, width, height = (int(value) for value in detection["box"])
                cat = detection.get("label") == "cat"
                colour = (60, 170, 255) if cat else (45, 220, 90)
                cv2.rectangle(image, (x, y), (x + width, y + height), colour, 3 if cat else 2)
                cv2.putText(image, f"{detection['label']} {float(detection['confidence']):.0%}", (x, max(y - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
            phase = str(status.get("phase") or "").upper()
            action = str(status.get("action") or "")[:70]
            cv2.rectangle(image, (0, image.shape[0] - 44), (image.shape[1], image.shape[0]), (16, 24, 32), -1)
            cv2.putText(image, phase, (8, image.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (93, 179, 242), 2, cv2.LINE_AA)
            cv2.putText(image, action, (8, image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 243, 247), 1, cv2.LINE_AA)
            encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encoded:
                with self._lock:
                    self._hunt_frame = jpeg.tobytes()
                    self._hunt_frame_at = time.time()
        except Exception:
            logger.debug("Could not draw the hunt frame", exc_info=True)

    def hunt_frame(self) -> bytes | None:
        with self._lock:
            return self._hunt_frame

    def hunt_status(self) -> dict[str, object]:
        with self._lock:
            running = self._enabled and self._mode == "hunt"
            status = dict(self._hunt_status or {})
            frame_at = self._hunt_frame_at
        return {
            "running": running,
            "status": status or None,
            "frame_at": frame_at or None,
            "frame_age_seconds": round(time.time() - frame_at, 1) if frame_at else None,
            "camera": dict(self._hunt_camera_status),
            "learned": {
                "favourites": self._hunt_memory.favourites(5),
                "hunts": self._hunt_memory.hunts,
                "finds": self._hunt_memory.finds,
                "cat_spots": [[round(v, 1) for v in spot] for spot in self._hunt_memory.cat_spots[:8]],
            },
        }

    def start_calibration(self, attended: bool) -> None:
        """Measure real turns and steps (car facing a flat wall about 1-1.5 m away)."""
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before calibrating")
        if not self.camera_on_head:
            raise RuntimeError(
                "Turn calibration measures with pictures from the panning head; "
                "with the camera fixed facing forward it is not available yet"
            )
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
        if drive_status["mode"] != "hardware":
            raise RuntimeError("Calibration requires hardware mode")
        if any(abs(float(drive_status[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")):
            raise RuntimeError("Stop the car before calibrating")
        if not self._ranging.lidar_ready():
            raise RuntimeError("Calibration needs the lidar working")
        self._start("calibrate", self._run_calibration)

    def calibration(self) -> dict[str, object] | None:
        with self._lock:
            return self._calibration.to_json() if self._calibration else None

    def _run_calibration(self) -> None:
        self._hold_camera(True)
        try:
            self._set_state("calibrating")
            self._drive.set_inhibited(False)
            with self._lock:
                previous = self._calibration
            from robocar.bodies import CarCalibrationRig  # bodies builds on this module

            with self._lock:
                self._last_move = None
            result = run_calibration(
                CarCalibrationRig(self),
                previous,
                _guess_step_seconds,
                self._drive.heading_source(),
                feel_steps=self._drive.gyro_ready(),
            )
            if self._stop_event.is_set():
                return
            if result.turn_ratios or result.step_durations:
                self._calibration_store.save(result)
                with self._lock:
                    self._calibration = result
                    self._state = "calibrated"
                    self._rotation_source = result.turn_source
                    self._odometry_scale = (
                        result.forward_ratio or self._odometry_scale[0],
                        result.mean_turn_ratio() or self._odometry_scale[1],
                    )
            else:
                self._set_state("calibration_failed")
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Calibration stopped after an error")
                with self._lock:
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.stop()
            self._set_enabled_false()
            self._hold_camera(False)

    def _start(self, mode: str, target: Callable[[], None]) -> None:
        with self._lock:
            if self._enabled or (self._thread is not None and self._thread.is_alive()):
                raise RuntimeError("A mapping session is already active")
            self._enabled = True
            self._mode = mode
            self._state = "waiting_for_stop" if mode == "survey" else "scanning"
            self._error = None
            self._run_id = str(uuid.uuid4())
            self._move_count = 0
            self._plan = None
            self._last_place = None
            with self._map_lock:
                self._explore_target = None
            self._stop_event.clear()
            run_id = self._run_id

            def run() -> None:
                # Calibration only measures motion; mapping runs build on
                # everything mapped so far rather than starting over.
                if mode != "calibrate":
                    self._continue_map(run_id)
                target()

            self._thread = threading.Thread(
                target=run,
                name=f"mapping-{mode}",
                daemon=True,
            )
            self._thread.start()

    def _continue_map(self, run_id: str) -> None:
        """Start a run from the newest stop of any earlier run, with its map."""
        with self._places_lock:
            last = max(self._places.graph.nodes, default=None)
        if last is None:
            with self._lock:
                self._reset_localization()
            with self._map_lock:
                self._surface_map = SurfaceMap()
                self._grid = CoverageGrid()
                self._cloud = PointCloud()
                self._semantic = SemanticMap()
                self._path, self._history, self._links = [], [], []
                self._map_run_id = run_id
            return
        self._rebuild_map(last, run_id)
        with self._places_lock:
            pose = self._places.graph.nodes[last]
        with self._lock:
            # Wheel odometry since that stop, the previous views and the
            # learned scales carry over when the service has kept running.
            tracked = self._last_node == last and self._last_odometry is not None
            self._pose = pose
            if tracked:
                # The wheels have been counted since that stop: the next stop
                # adds whatever they saw, so the estimate stays as sure as it
                # was, but no surer than the floor. Only the variances are
                # topped up (adding to the diagonal keeps it a covariance).
                floor = np.array(
                    [
                        CONTINUE_SAME_SESSION_SIGMA_CM**2,
                        CONTINUE_SAME_SESSION_SIGMA_CM**2,
                        math.radians(CONTINUE_SAME_SESSION_HEADING_SIGMA_DEG) ** 2,
                    ]
                )
                shortfall = np.clip(floor - np.diag(self._pose_covariance), 0.0, None)
                self._pose_covariance = self._pose_covariance + np.diag(shortfall)
            else:
                self._pose_covariance = np.diag(
                    [CONTINUE_POSITION_SIGMA_CM**2, CONTINUE_POSITION_SIGMA_CM**2, math.radians(CONTINUE_HEADING_SIGMA_DEG) ** 2]
                )
                self._stops_since_fix = None
                self._travel_since_fix = (0.0, 0.0)
            self._last_node = last
            self._joining_run = not tracked
            self._last_localization = None
            self._verified_turn = None

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            if self._state != "error":
                self._state = "idle"
            self._stop_event.set()
            thread = self._thread
        self._drive.inhibit_and_stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._drive.set_inhibited(False)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def status(self) -> dict[str, object]:
        self._follow_heading_source()
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
            status = {
                "enabled": self._enabled,
                "state": self._state,
                "error": self._error,
                "observation_count": self._store.count(),
                "last_observation": self._last_observation,
                "last_id": self._last_id,
                "mode": self._mode,
                "strategy": self._strategy,
                "spin": dict(self._spin_progress) if self._spin_progress else None,
                "room_choices": list(self._room_choices[-3:]),
                "mapping_methods": self._records.to_json(),
                "run_id": self._run_id,
                "pose_model": "wheel odometry + camera yaw/travel + sonar surface registration",
                "move_count": self._move_count,
                "plan": self._plan,
                "calibrated_at": self._calibration.created_at if self._calibration else None,
                "exploration_learning": self._learner.to_json(),
                "hazards": [
                    {"bearing_deg": round(bearing, 1), "range_cm": None if distance is None else round(distance, 1)}
                    for bearing, distance in self._hazards
                ],
                "pose": [round(value, 1) for value in self._pose],
                "pose_sigma_cm": round(position_sigma, 1),
                "heading_sigma_deg": round(heading_sigma, 1),
                "odometry_scale": [round(value, 3) for value in self._odometry_scale],
                "turns_measured_by": self._rotation_source,
                "distances_measured_by": self._translation_source,
                "last_move": self._last_move,
                "bursts_measured": len(self._bursts),
                "coast": self._coast.summary(),
                "turn_power": self._turn_power.summary(),
                "pan_scale": round(pan_scale(), 3),
                "pan_scale_samples": self._pan_scale_samples,
                "last_bump": self._last_bump,
                "localization": self._last_localization,
                "places": dict(self._place_counts),
                "last_place": self._last_place,
                "forward_places": len(self._forward_places.keyframes),
                "cats_seen_while_mapping": self._mapping_cats,
                "camera_place_fixes": dict(self.place_fixes),
            }
        with self._map_lock:
            status["surface_count"] = len(self._surface_map.walls())
            status["feature_count"] = len(self._surface_map.features())
            status["landmark_count"] = len(self._semantic.confirmed())
        detector_info = getattr(self._detector, "info", None)
        status["detector"] = detector_info() if callable(detector_info) else None
        status["localized"] = self.localized()
        return status

    def _start_mapping_watch(self) -> None:
        """With a separate forward camera, watch its feed while mapping: the
        cat seen there goes into the hunt's memory of where it likes to be.
        (With one camera, mapping has the camera to itself.)"""
        detect = getattr(self._detector, "detect", None)
        grab = getattr(self._hunt_camera, "capture_frame", None)
        if self._hunt_camera is self._camera or detect is None or grab is None:
            return
        self._mapping_watcher = Watcher(
            grab,
            detect,
            lambda: STRIP_CENTRE_PAN,
            self._drive.odometry,
            lambda: any(abs(float(self._drive.status()[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")),
            set_exposure=getattr(self._hunt_camera, "set_exposure", None),
            on_sighting=self._mapping_sighting,
        )
        self._mapping_watcher.start()

    def _stop_mapping_watch(self) -> None:
        watcher, self._mapping_watcher = self._mapping_watcher, None
        if watcher is None:
            return
        watcher.stop()
        if self._mapping_cats:
            try:
                save_memory(self._hunt_memory_path, self._hunt_memory)
            except OSError as exc:
                logger.warning("Could not save where the cat was seen: %s", exc)

    def _mapping_sighting(self, sighting: FeedSighting) -> None:
        cat = sighting.look.cat
        if cat is None or sighting.at - self._last_mapping_cat < MAPPING_CAT_EVERY_SECONDS:
            return
        self._last_mapping_cat = sighting.at
        with self._lock:
            reference, last_odometry = self._pose, self._last_odometry
            if last_odometry is None:
                return  # not on the map yet
            self._mapping_cats += 1
        pose = compose(reference, relative(last_odometry, sighting.odometry))
        distance = self._hunt_memory.size_scale / max(cat.box_height, 1.0)
        heading = math.radians(pose[2] + cat.bearing_deg)
        x, y = pose[0] - math.sin(heading) * distance, pose[1] + math.cos(heading) * distance
        self._hunt_memory.remember_cat(x, y)
        logger.info("Saw the cat while mapping, at about (%.0f, %.0f) cm", x, y)

    def _hunt_recognizer(self) -> PlaceRecognizer:
        """Remembered places from the camera the hunt watches through."""
        return self._forward_places if self._hunt_camera is not self._camera else self._places.recognizer

    def place_fix_from_picture(self, jpeg: bytes, camera_bearing: float, odometry: Pose) -> dict[str, object] | None:
        """Correct the pose from a picture the hunt's camera took (with the
        car's odometry then), by matching it to remembered places; what
        happened, or None when nothing matched."""
        with self._lock:
            reference, last_odometry, covariance = self._pose, self._last_odometry, self._pose_covariance.copy()
            last_node, run_id = self._last_node, self._run_id or ""
        if last_odometry is None or last_node is None:
            return None
        query = make_keyframe(-1, run_id, jpeg, camera_bearing, camera_hfov(self._hunt_camera))
        if query is None:
            return None
        travelled = relative(last_odometry, odometry)
        estimate = compose(reference, travelled)
        moved, turned = math.hypot(travelled[0], travelled[1]), abs(wrap_degrees(travelled[2]))
        grown = covariance + np.diag(
            [
                (DRIFT_POSITION_PER_CM * moved) ** 2,
                (DRIFT_POSITION_PER_CM * moved) ** 2,
                math.radians(DRIFT_HEADING_DEG_PER_CM * moved + DRIFT_HEADING_PER_DEGREE * turned) ** 2,
            ]
        )
        with self._places_lock:
            recognizer = self._hunt_recognizer()
            graph = self._places.graph
            # Only places in the car's own map frame.
            poses = {node: graph.nodes[node] for node in graph.component(last_node)} if last_node in graph.nodes else {}
            fix = locate_from_places(query, recognizer, poses, estimate) if poses else None
        with self._lock:
            self.place_fixes["tried"] = int(self.place_fixes["tried"]) + 1
        if fix is None:
            return None
        if len(fix.places) >= 3:
            position_sigma, heading_sigma = CAMERA_FIX_POSITION_SIGMA_CM, CAMERA_FIX_HEADING_SIGMA_DEG
        else:
            position_sigma, heading_sigma = CAMERA_FIX_TWO_PLACES_POSITION_SIGMA_CM, CAMERA_FIX_TWO_PLACES_HEADING_SIGMA_DEG
        fused = fuse_pose(
            estimate,
            grown,
            (fix.position[0], fix.position[1], fix.heading_deg),
            np.diag([position_sigma**2, position_sigma**2, math.radians(heading_sigma) ** 2]),
            GRID_MATCH_GATE,
        )
        result = {**fix.summary(), "used": fused is not None}
        with self._lock:
            if fused is None:
                self.place_fixes["refused"] = int(self.place_fixes["refused"]) + 1
            elif self._last_odometry == last_odometry:  # no stop since the picture
                # The fix becomes the new reference: pose at the picture,
                # with the odometry then.
                self._pose, self._pose_covariance = fused
                self._last_odometry = odometry
                self._stops_since_fix = 0
                self._travel_since_fix = (0.0, 0.0)
                self.place_fixes["used"] = int(self.place_fixes["used"]) + 1
            self.place_fixes["last"] = result
        return result

    def _match_to_grid(self, scan: list[dict[str, object]], prior_pose: Pose, prior_covariance: np.ndarray):
        """The sweep fitted to the coverage map near the predicted pose,
        searching as far as the prediction is uncertain (None without a map)."""
        position_sigma, heading_sigma = pose_sigma(prior_covariance)
        reach_cm = min(GRID_MATCH_MAX_REACH_CM, max(GRID_MATCH_MIN_REACH_CM, 2.0 * position_sigma))
        reach_deg = min(GRID_MATCH_MAX_REACH_DEG, max(GRID_MATCH_MIN_REACH_DEG, 2.0 * heading_sigma))
        with self._map_lock:
            if int(self._grid.occupied().sum()) < LOCALIZED_MIN_OCCUPIED_CELLS:
                return None
            try:
                return self._grid.match_scan(scan, prior_pose, reach_cm, reach_deg)
            except Exception:  # a failed match must never stop the stop
                logger.warning("Matching the sweep to the coverage map failed", exc_info=True)
                return None

    def localized(self) -> dict[str, object]:
        """Whether the car believes it knows roughly where it is and which way
        it faces on the map, and why (not)."""
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
            since = self._stops_since_fix
            travelled, rotated = self._travel_since_fix
            last_odometry = self._last_odometry
            translation_scale = self._odometry_scale[0]
        with self._map_lock:
            walls = len(self._surface_map.walls())
            mapped = walls >= LOCALIZED_MIN_WALLS or int(self._grid.occupied().sum()) >= LOCALIZED_MIN_OCCUPIED_CELLS
        moved, turned = travelled, rotated
        if last_odometry is not None:
            change = relative(last_odometry, self._drive.odometry())
            # Corrected as each stop's travel is: the learned distance scale
            # and the calibrated turn ratio.
            moved += math.hypot(change[0], change[1]) * translation_scale
            rotation = wrap_degrees(change[2])
            turned += abs(rotation * self._rotation_ratio(rotation)) if rotation else 0.0
        reasons = []
        if not mapped:
            reasons.append("no map to match against yet")
        elif since is None:
            reasons.append("has not matched itself to the map yet")
        if position_sigma > LOCALIZED_MAX_SIGMA_CM:
            reasons.append(f"position uncertain (±{position_sigma:.0f} cm)")
        if heading_sigma > LOCALIZED_MAX_HEADING_SIGMA_DEG:
            reasons.append(f"heading uncertain (±{heading_sigma:.0f}°)")
        if moved > LOCALIZED_MAX_DEAD_RECKONING_CM:
            reasons.append(f"drove {moved / 100:.1f} m on wheel counts since it last matched the map")
        if turned > LOCALIZED_MAX_DEAD_RECKONING_DEG:
            reasons.append(f"turned {turned:.0f}° on wheel counts since it last matched the map")
        if reasons:
            detail = "; ".join(reasons)
        else:
            when = (
                "at the last stop"
                if since == 0
                else f"{since} stops ago, {moved / 100:.1f} m of wheel counts since"
            )
            detail = f"matched to the map {when}: ±{position_sigma:.0f} cm, ±{heading_sigma:.0f}°"
        return {
            "known": not reasons,
            "detail": detail,
            "sigma_cm": round(position_sigma, 1),
            "heading_sigma_deg": round(heading_sigma, 1),
            "stops_since_fix": since,
        }

    def observations(self) -> list[dict[str, object]]:
        return self._store.recent()

    def surface_map(self) -> dict[str, object]:
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
        with self._map_lock:
            data = self._surface_map.to_dict()
            data["run_id"] = self._map_run_id
            data["path"] = [[round(value, 1) for value in pose] for pose in self._path]
            data["history"] = [[[round(value, 1) for value in pose] for pose in path] for path in self._history]
            if self._grid.log_odds.size:
                data["grid"] = {
                    "bounds": [round(value, 1) for value in self._grid.bounds()],
                    "explored": round(self._grid.explored_fraction(), 3),
                }
                data["room"] = self._grid.known_room()
            data["cloud_points"] = len(self._cloud)
            data["stations"] = [[round(x, 1), round(y, 1)] for x, y in self._stations]
            data["frontier_target"] = (
                [round(self._explore_target[0], 1), round(self._explore_target[1], 1)] if self._explore_target else None
            )
            data["links"] = [
                [[round(start[0], 1), round(start[1], 1)], [round(end[0], 1), round(end[1], 1)]]
                for start, end in self._links
            ]
            data.update(self._semantic.to_json(self._grid.wall_between))
        with self._lock:
            last_cat = (self._hunt_status or {}).get("last_cat_position")
        data["cat"] = {
            "last_seen": last_cat,
            "spots": [[round(v, 1) for v in spot] for spot in self._hunt_memory.cat_spots[:8]],
        }
        data["pose_sigma_cm"] = round(position_sigma, 1)
        data["heading_sigma_deg"] = round(heading_sigma, 1)
        return data

    def lidar_cloud(self) -> dict[str, object]:
        """The lidar points of the map now shown, for the dashboard."""
        with self._map_lock:
            return {"run_id": self._map_run_id, "cell_cm": CLOUD_CELL_CM, "points": self._cloud.to_json()}

    def image(self, observation_id: int) -> bytes | None:
        return self._store.image(observation_id)

    def reset(self) -> None:
        with self._lock:
            if self._enabled or (self._thread is not None and self._thread.is_alive()):
                raise RuntimeError("Stop mapping before resetting its database")
            self._store.reset()
            self._last_id = None
            self._last_observation = None
            self._run_id = None
            self._mode = "idle"
            self._state = "idle"
            self._error = None
            self._move_count = 0
            self._plan = None
            self._reset_localization()
            with self._places_lock:
                self._places = PlaceMemory(self._store)
                self._place_counts = self._places.counts()
                self._forward_places = PlaceRecognizer()
            self._last_place = None
            with self._map_lock:
                self._surface_map = SurfaceMap()
                self._grid = CoverageGrid()
                self._cloud = PointCloud()
                self._semantic = SemanticMap()
                self._explore_target = None
                self._path = []
                self._history = []
                self._links = []
                self._map_run_id = None

    def close(self) -> None:
        self.stop()
        if self._coast.observations or self._pan_scale_samples or self._bursts or self._turn_power.observations:
            self._save_learned()
        self._store.close()

    def _replay_latest_run(self) -> None:
        """Rebuild the newest run's map, joined with every session it recognized."""
        run_id = self._store.latest_run_id()
        with self._places_lock:
            nodes = [node for node, run in self._places.graph.runs.items() if run == run_id]
        if not nodes:
            return
        latest = max(nodes)
        self._rebuild_map(latest, run_id)
        with self._places_lock:
            pose = self._places.graph.nodes[latest]
            position_sigma, heading_sigma = self._places.stop(latest).pose_sigma
        self._pose = pose
        self._pose_covariance = np.diag(
            [position_sigma**2, position_sigma**2, math.radians(heading_sigma) ** 2]
        )

    def _rebuild_map(self, node: int, run_id: str) -> IntegrationReport | None:
        """Replay every sweep in the node's connected sessions at their optimized poses."""
        with self._places_lock:
            component = self._places.graph.component(node)
            stops = [
                (other, self._places.stop(other), self._places.graph.nodes[other])
                for other in sorted(component)
            ]
            links = [
                (self._places.graph.nodes[source], self._places.graph.nodes[target])
                for source, target, _ in self._places.loop_links(component)
            ]
        surface_map = SurfaceMap()
        grid = CoverageGrid()
        cloud = PointCloud()
        semantic = SemanticMap()
        path: list[Pose] = []
        history: dict[str, list[Pose]] = {}
        report = None
        for other, stop, pose in stops:
            report = surface_map.integrate(stop.scan, pose, *stop.pose_sigma)
            grid.integrate(stop.scan, pose, surface_map.beam_width_deg)
            cloud.add(lidar_points(static_scan(stop.scan), pose))
            semantic.add(stop.sightings, pose)
            if stop.run_id == run_id:
                path.append(pose)
            else:
                history.setdefault(stop.run_id, []).append(pose)
        with self._map_lock:
            self._surface_map = surface_map
            self._grid = grid
            self._cloud = cloud
            self._semantic = semantic
            self._path = path
            self._history = list(history.values())
            self._links = links
            self._map_run_id = run_id
        return report

    def _hold_camera(self, held: bool) -> None:
        for camera in {id(self._camera): self._camera, id(self._hunt_camera): self._hunt_camera}.values():
            hold = getattr(camera, "hold", None)
            if hold is not None:
                hold(held)

    def _run_survey(self) -> None:
        capture_pending = True
        stopped_since: float | None = None
        self._hold_camera(True)
        self._start_mapping_watch()
        try:
            while not self._stop_event.is_set():
                drive_status = self._drive.status()
                moving = any(
                    abs(float(drive_status[axis])) > 0.01
                    for axis in ("throttle", "strafe", "steering")
                )
                if moving:
                    capture_pending = True
                    stopped_since = None
                    self._set_state("waiting_for_stop")
                elif capture_pending:
                    if stopped_since is None:
                        stopped_since = time.monotonic()
                    elif time.monotonic() - stopped_since >= STOP_SETTLE_SECONDS:
                        if self._drive.set_inhibited(True):
                            self._set_state("capturing")
                            self._capture_observation()
                            self._drive.set_inhibited(False)
                            capture_pending = False
                            stopped_since = None
                            self._set_state("waiting_for_stop")
                self._stop_event.wait(0.1)
        except Exception as exc:
            logger.exception("Mapping survey stopped after an error")
            self._drive.stop()
            with self._lock:
                self._enabled = False
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.set_inhibited(False)
            self._stop_mapping_watch()
            self._hold_camera(False)

    def _run_autonomous(self) -> None:
        """Frontier exploring on its own (strategy "frontier")."""
        started_at = time.monotonic()
        self._learner.start_run()
        self._hold_camera(True)
        self._start_mapping_watch()
        try:
            outcome = self._frontier_explore(lambda: self._frontier_limits_reached(started_at))
            if outcome in ("explored", "blocked"):
                self._set_state(outcome)
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Autonomous mapping stopped after an error")
            self._drive.stop()
            if not self._stop_event.is_set():
                with self._lock:
                    self._enabled = False
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._finish_exploring()

    def _frontier_limits_reached(self, started_at: float) -> bool:
        if time.monotonic() - started_at >= AUTO_MAX_SECONDS:
            self._set_state("time_limit")
            return True
        with self._lock:
            if self._move_count >= AUTO_MAX_MOVES:
                self._state = "move_limit"
                return True
        return False

    def _finish_exploring(self) -> None:
        self._drive.stop()
        if not self._stop_event.is_set():
            self._drive.set_inhibited(False)
        self._set_enabled_false()
        self._stop_mapping_watch()
        self._hold_camera(False)
        with self._map_lock:
            self._learner.finish(self._grid.known_count())
        try:
            save_learner(self._learner_path, self._learner)
        except OSError as exc:
            logger.warning("Could not save exploration learning: %s", exc)

    def _frontier_explore(
        self,
        limits_reached: Callable[[], bool],
        max_moves: int | None = None,
        until: Callable[[], bool] | None = None,
    ) -> str:
        """Frontier steps: a sweep ahead, then a move toward the most promising
        unexplored edge, again and again. Ends "explored" (nothing left),
        "blocked", "limit" (`limits_reached`, or `max_moves` made), "found"
        (`until` became true: room mapping goes back to spinning) or
        "stopped"."""
        turns_without_progress = 0
        backup_bursts = 0
        just_backed_up = False
        moves = 0
        while not self._stop_event.is_set():
            if limits_reached():
                return "limit"
            if max_moves is not None and moves >= max_moves:
                return "limit"
            if not self._drive.set_inhibited(True):
                raise RuntimeError("Drive was not stopped before a range scan")
            self._set_state("scanning")
            scan = self._capture_observation(mode="explore")
            if self._stop_event.is_set():
                break
            if not any(item["distance_cm"] is not None for item in scan):
                self._set_state("sensor_fault")
                raise RuntimeError("No valid range readings; motors remain stopped")
            if until is not None and moves and until():
                return "found"
            # A missing straight-ahead echo (often a smooth wall at a steep
            # angle) counts as blocked: the planner turns rather than drives.
            with self._lock:
                pose = self._pose
                hazards = list(self._hazards)
            with self._map_lock:
                step = plan_explore_step(
                    scan,
                    pose,
                    self._grid,
                    turns_without_progress,
                    backup_bursts,
                    just_backed_up,
                    self._explore_target,
                    self._learner,
                    hazards,
                )
                self._explore_target = step.target
            with self._lock:
                self._plan = step.reason
            if step.action == "done":
                return "explored"
            if step.action == "blocked":
                return "blocked"

            self._drive.set_inhibited(False)
            if step.action == "forward":
                self._set_state("moving")
                if not self._move_burst(AUTO_FORWARD_COMMAND, self._travel_seconds(step.distance_cm), watch=True):
                    break
                turns_without_progress = 0
                just_backed_up = False
            elif step.action == "turn":
                self._set_state("turning")
                pieces = max(1, math.ceil(abs(step.degrees) / AUTO_TURN_PIECE_DEGREES - 1e-9))
                if not all(self._turn_and_align(step.degrees / pieces) for _ in range(pieces)):
                    break
                turns_without_progress += 1
                # Only a change of direction earns another backing-up.
                backup_bursts = 0
                just_backed_up = False
                if step.distance_cm > 0:
                    # Lined up with a waypoint the map shows clear: one
                    # check ahead must agree before driving on.
                    ahead = self._ping_ahead()
                    if ahead is not None and ahead - step.distance_cm >= AUTO_WALL_STANDOFF_CM:
                        self._set_state("moving")
                        if not self._move_burst(AUTO_FORWARD_COMMAND, self._travel_seconds(step.distance_cm), watch=True):
                            break
                        turns_without_progress = 0
            elif step.action == "strafe":
                self._set_state("sliding")
                seconds = AUTO_FORWARD_SECONDS * abs(step.distance_cm) / AUTO_STEP_CM
                if not self._move_burst(0, seconds, strafe=math.copysign(AUTO_FORWARD_COMMAND, step.distance_cm)):
                    break
            else:
                self._set_state("backing_up")
                if not self._move_burst(-AUTO_FORWARD_COMMAND, AUTO_FORWARD_SECONDS):
                    break
                backup_bursts += 1
                turns_without_progress = 0
                just_backed_up = True
            self._count_move()
            moves += 1
            self._drive.set_inhibited(True)
            self._set_state("scanning")
            if self._stop_event.wait(0.1):
                break
        return "stopped"

    # -- spin and roam ----------------------------------------------------

    def _current_pose(self) -> Pose:
        """The pose now: the last stop's, moved on by the wheels since."""
        with self._lock:
            pose, last = self._pose, self._last_odometry
        return compose(pose, relative(last, self._drive.odometry())) if last else pose

    def _spin_limits_reached(self, started_at: float) -> bool:
        if time.monotonic() - started_at >= SPIN_MAX_SECONDS:
            self._set_state("time_limit")
            return True
        with self._lock:
            if self._move_count >= SPIN_MAX_MOVES:
                self._state = "move_limit"
                return True
        return False

    def _sweep_here(self) -> list[dict[str, object]] | None:
        """A stop: sweep, place the car on the map, add what it saw. None if stopped."""
        if not self._drive.set_inhibited(True):
            raise RuntimeError("Drive was not stopped before a range scan")
        self._set_state("scanning")
        scan = self._capture_observation(mode="explore")
        if self._stop_event.is_set() or not scan:
            return None
        if not any(item["distance_cm"] is not None for item in scan):
            self._set_state("sensor_fault")
            raise RuntimeError("No valid range readings; motors remain stopped")
        return scan

    def _turn_to_heading(self, heading: float) -> bool:
        """Turn on the spot to face a map heading, in camera-checked pieces."""
        degrees = wrap_degrees(heading - self._current_pose()[2])
        if abs(degrees) < 2.0:
            return True
        self._drive.set_inhibited(False)
        try:
            self._set_state("turning")
            pieces = max(1, math.ceil(abs(degrees) / AUTO_TURN_PIECE_DEGREES - 1e-9))
            for _ in range(pieces):
                if not self._turn_and_align(degrees / pieces):
                    return False
            self._count_move()
            return True
        finally:
            self._drive.set_inhibited(True)

    def _room_to_turn(self, scan: list[dict[str, object]]) -> bool:
        """Whether turning on the spot is safe: the latest sweep shows room
        ahead and to the sides, and the map shows nothing confirmed within
        the car's turning circle behind or beside it, where the sweep cannot
        see."""
        clearances = direction_clearances(scan)
        if clearances and min(clearances.values()) < AUTO_TURN_MIN_CLEARANCE_CM:
            return False
        pose = self._current_pose()
        with self._map_lock:
            return self._grid.clear_all_round(pose, SPIN_TURN_RADIUS_CM)

    def _station_base(self) -> tuple[CoverageGrid, int | None]:
        """The map and the latest stop as they are before a station's sweeps."""
        with self._map_lock:
            grid = copy.deepcopy(self._grid)
        with self._lock:
            return grid, self._last_node

    def _spin(self, scan: list[dict[str, object]], base: tuple[CoverageGrid, int | None] | None = None) -> bool:
        """Turn a full circle where the car stands, sweeping a quarter turn
        apart. `scan` is the sweep already taken facing the first quarter;
        `base` the map and stop from before it, to fit the whole circle to.
        False if stopped."""
        with self._lock:
            start = self._pose
            sweeps = [(self._last_node, scan)]
        here = (start[0], start[1])
        swept = 1
        skipped = 0
        for quarter in SPIN_QUARTERS:
            if self._stop_event.is_set():
                return False
            heading = start[2] + quarter
            with self._map_lock:
                # With nothing mapped yet every way is new.
                gain = (
                    self._grid.information_gain(here, heading, self._grid.worth_seeing(), self._grid._blocked())
                    if self._grid.log_odds.size
                    else math.inf
                )
            if gain < SPIN_SKIP_GAIN_CELLS:
                skipped += 1
                continue
            if not self._room_to_turn(scan):
                with self._lock:
                    self._plan = "too close to something to turn on the spot; ending this spin"
                break
            with self._lock:
                self._plan = f"spinning: sweep {swept + 1} facing {quarter:.0f}° round"
                self._spin_progress = {**(self._spin_progress or {}), "quarter": quarter, "swept": swept}
            if not self._turn_to_heading(heading):
                return False
            scan = self._sweep_here()
            if scan is None:
                return False
            with self._lock:
                sweeps.append((self._last_node, scan))
            swept += 1
        fit = self._fit_circle(base, sweeps) if base is not None else None
        with self._lock:
            self._spin_progress = {**(self._spin_progress or {}), "swept": swept, "skipped": skipped, "circle_fit": fit}
        return True

    def _fit_circle(
        self, base: tuple[CoverageGrid, int | None], sweeps: list[tuple[int | None, list[dict[str, object]]]]
    ) -> dict[str, object] | None:
        """Fit a station's sweeps, as one circle, to the map from before it;
        if it fits, tie the station to the stop before it and redraw the map.
        A summary for the status, or None when there was nothing to fit to."""
        grid, base_node = base
        nodes = [node for node, _ in sweeps if node is not None]
        if base_node is None or len(nodes) < CIRCLE_FIT_MIN_SWEEPS or int(grid.occupied().sum()) < LOCALIZED_MIN_OCCUPIED_CELLS:
            return None
        with self._places_lock:
            poses = {node: self._places.graph.nodes.get(node) for node in nodes}
            base_pose = self._places.graph.nodes.get(base_node)
        if base_pose is None or any(pose is None for pose in poses.values()):
            return None
        last = nodes[-1]
        frame = poses[last]
        circle: list[dict[str, object]] = []
        for node, scan in sweeps:
            if node is not None:
                # Placed relative to the last by the gyro's turns between them.
                circle += sweep_seen_from(static_scan(scan), poses[node], frame)
        with self._lock:
            covariance = self._pose_covariance.copy()
            run_id = self._run_id
        position_sigma, heading_sigma = pose_sigma(covariance)
        reach_cm = min(GRID_MATCH_MAX_REACH_CM, max(GRID_MATCH_MIN_REACH_CM, 2.0 * position_sigma))
        reach_deg = min(GRID_MATCH_MAX_REACH_DEG, max(GRID_MATCH_MIN_REACH_DEG, 2.0 * heading_sigma))
        try:
            match = grid.match_scan(circle, frame, reach_cm, reach_deg)
        except Exception:  # a failed fit must never stop exploring
            logger.warning("Fitting the full circle to the map failed", exc_info=True)
            return None
        summary: dict[str, object] = {"sweeps": len(nodes), "score": None if match is None else round(match.score, 2), "used": False}
        if match is None or not match.good:
            return summary
        with self._places_lock:
            tied = self._places.add_constraint(
                base_node, last, relative(base_pose, match.pose), CIRCLE_FIT_SIGMA, "circle", match.hits
            )
            settled = self._places.graph.nodes[last]
        if not tied or run_id is None:
            return summary
        self._rebuild_map(last, run_id)
        moved = relative(frame, settled)
        with self._lock:
            self._pose = settled
            self._pose_covariance = np.diag(
                [CIRCLE_FIT_SIGMA[0] ** 2, CIRCLE_FIT_SIGMA[1] ** 2, math.radians(CIRCLE_FIT_SIGMA[2]) ** 2]
            )
            self._stops_since_fix = 0
            self._travel_since_fix = (0.0, 0.0)
        logger.info(
            "Full circle of %d sweeps fitted the map (score %.2f): station moved %.1f cm, %+.1f degrees",
            len(nodes), match.score, math.hypot(moved[0], moved[1]), moved[2],
        )
        return {**summary, "used": True, "moved_cm": round(math.hypot(moved[0], moved[1]), 1), "turned_deg": round(moved[2], 1)}

    def _drive_to_station(self, goal: tuple[float, float], started_at: float) -> str:
        """Drive to a station along the map's route: "arrived", "blocked"
        (no way, or something in the way), or "stopped"."""
        travelled = 0.0
        sweep_every = SPIN_SWEEP_EVERY_MEASURED_CM if self._drive.measures_distance() and self._drive.gyro_ready() else SPIN_SWEEP_EVERY_CM
        for _ in range(SPIN_MAX_LEGS):
            if self._stop_event.is_set():
                return "stopped"
            if self._spin_limits_reached(started_at):
                return "stopped"
            pose = self._current_pose()
            if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= SPIN_ARRIVED_CM:
                return "arrived"
            if travelled >= sweep_every:
                # Long enough on the wheels alone: sweep ahead to stay placed.
                if self._sweep_here() is None:
                    return "stopped"
                travelled = 0.0
                pose = self._current_pose()
            with self._map_lock:
                leg = self._grid.route(pose, goal)
            if leg is None:
                return "blocked"
            bearing, length = leg
            if abs(bearing) > AUTO_HEADING_TOLERANCE_DEGREES:
                if not self._turn_to_heading(pose[2] + bearing):
                    return "stopped"
            with self._lock:
                hazards = list(self._hazards)
                stop_heading = self._pose[2]
            if hazard_within(hazards, wrap_degrees(self._current_pose()[2] - stop_heading)):
                with self._lock:
                    self._plan = "a person or animal is close ahead; not driving at them"
                return "blocked"
            ahead = self._ping_ahead()
            facing = self._current_pose()
            with self._map_lock:
                mapped = self._grid.clear_run(facing, length)
            if ahead is None:
                return "blocked"
            # As far as the way runs straight, the lidar saw clear and the map
            # knows is open: across the room in one go if it is all open.
            step = min(length, ahead - AUTO_WALL_STANDOFF_CM, mapped)
            if step < AUTO_MIN_STEP_CM:
                return "blocked"
            self._drive.set_inhibited(False)
            try:
                self._set_state("moving")
                with self._lock:
                    self._plan = f"driving {step:.0f} cm toward the next place to spin ({math.hypot(goal[0] - facing[0], goal[1] - facing[1]):.0f} cm away)"
                if not self._move_burst(AUTO_FORWARD_COMMAND, self._travel_seconds(step), watch=True):
                    return "stopped"
            finally:
                self._drive.set_inhibited(True)
            self._count_move()
            now = self._current_pose()
            travelled += math.hypot(now[0] - facing[0], now[1] - facing[1])
            if self._move_abort is not None:
                with self._lock:
                    self._plan = f"stopped short ({self._move_abort}); looking again"
        return "arrived" if math.hypot(goal[0] - self._current_pose()[0], goal[1] - self._current_pose()[1]) <= SPIN_ARRIVED_CM * 2 else "blocked"

    def _run_room_mapping(self) -> None:
        """Room mapping, the main way: a full circle where it stands, then
        again and again whichever of the ways of carrying on promises the
        most new floor for the time (see room_plan): a full circle across
        the room (preferred), a full circle nearby, or a step toward an
        unexplored edge. Each way's promises are checked against what it
        really brought, and the timings against what things really took."""
        started_at = time.monotonic()
        self._learner.start_run()
        self._hold_camera(True)
        self._start_mapping_watch()
        with self._lock:
            self._stations = []
            self._spin_progress = {"stations": 0}
            self._room_choices = []
        unreachable: list[tuple[float, float]] = []
        frontier_spent = False
        limits = lambda: self._spin_limits_reached(started_at)
        best_rate = 0.0  # the most new floor a second any action has brought this run
        try:
            base = self._station_base()
            began = time.monotonic()
            scan = self._sweep_here()
            if scan is None:
                return
            self._records.note_timing("sweep_seconds", time.monotonic() - began)
            if not self._spin_station_here(scan, base, started_at):
                return
            # Rates go by the expected times (learned from real ones), as the
            # options are judged by them.
            with self._map_lock:
                best_rate = self._grid.known_count() / self._records.spin_seconds
            while not self._stop_event.is_set() and not limits():
                with self._map_lock:
                    found = room_options(
                        self._grid, self._current_pose(), [*self._stations, *unreachable], self._learner, self._records, frontier_spent
                    )
                option = choose_option(found, self._records, best_rate)
                self._note_choice(option, found)
                if option is None:
                    self._set_state("explored")
                    with self._lock:
                        self._plan = "nothing left that would show much new floor: the room is mapped"
                    break
                with self._lock:
                    self._plan = f"next: {option.describe()}"
                    self._explore_target = option.station.target if option.station else option.frontier.target if option.frontier else None
                with self._map_lock:
                    known_before = self._grid.known_count()
                began = time.monotonic()
                if option.method == "frontier":
                    outcome = self._frontier_explore(limits, max_moves=1)
                    if outcome in ("explored", "blocked"):
                        # Nothing for it to do from here; it may be offered
                        # again once the map has grown.
                        frontier_spent = True
                    elif outcome == "stopped":
                        break
                    else:
                        self._records.note_timing("sweep_seconds", time.monotonic() - began)
                else:
                    assert option.station is not None
                    start = self._current_pose()
                    outcome = self._drive_to_station(option.station.target, started_at)
                    if outcome == "stopped":
                        break
                    driven = math.hypot(self._current_pose()[0] - start[0], self._current_pose()[1] - start[1])
                    if outcome == "blocked":
                        unreachable.append(option.station.target)
                    elif driven > 20.0:
                        self._records.note_timing("drive_cm_per_second", driven / max(time.monotonic() - began, 1e-3))
                    base = self._station_base()
                    scan = self._sweep_here()
                    if scan is None:
                        break
                    with self._lock:
                        here = (self._pose[0], self._pose[1])
                        new_place = all(math.hypot(here[0] - x, here[1] - y) >= SPIN_AWAY_CM for x, y in self._stations)
                    if new_place:
                        spun_at = time.monotonic()
                        if not self._spin_station_here(scan, base, started_at):
                            break
                        self._records.note_timing("spin_seconds", time.monotonic() - spun_at)
                    frontier_spent = False
                with self._map_lock:
                    delivered = self._grid.known_count() - known_before
                self._records.note(option.method, option.gain, delivered)
                best_rate = max(best_rate, delivered / option.seconds)
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Room mapping stopped after an error")
            self._drive.stop()
            if not self._stop_event.is_set():
                with self._lock:
                    self._enabled = False
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._save_learned()
            self._finish_exploring()

    def _spin_station_here(self, scan: list[dict[str, object]], base, started_at: float) -> bool:
        """A full circle where the car stands (already swept once); False if stopped."""
        if self._spin_limits_reached(started_at):
            return False
        with self._lock:
            self._stations.append((self._pose[0], self._pose[1]))
            self._spin_progress = {**(self._spin_progress or {}), "stations": len(self._stations), "swept": 1}
        return self._spin(scan, base)

    def _note_choice(self, option, found) -> None:
        """Keep the latest choices (and what was weighed) for the status."""
        entry = {
            "chose": option.method if option else None,
            "options": [
                {"method": other.method, "cells": round(other.gain), "seconds": round(other.seconds, 1), "score": round(other.score, 2)}
                for other in found[:3]
            ],
        }
        with self._lock:
            self._room_choices = (self._room_choices + [entry])[-12:]

    def _run_spin_explore(self) -> None:
        """Spin and roam: turn a full circle sweeping, pick the open place
        that a full turn would show the most new floor from (for the driving
        it takes), drive there, and spin again, until nowhere is worth it.


        A full circle at each station maps the place all round and gives the
        sweeps plenty of overlap to fit to the map; a station is chosen only
        where there is room to turn on the spot."""
        started_at = time.monotonic()
        self._learner.start_run()
        self._hold_camera(True)
        self._start_mapping_watch()
        with self._lock:
            self._stations = []
            self._spin_progress = {"stations": 0}
        unreachable: list[tuple[float, float]] = []
        try:
            base = self._station_base()
            scan = self._sweep_here()
            spin_here = True
            while scan is not None and not self._stop_event.is_set():
                if self._spin_limits_reached(started_at):
                    break
                if spin_here:
                    with self._lock:
                        self._stations.append((self._pose[0], self._pose[1]))
                        self._spin_progress = {"stations": len(self._stations), "swept": 1}
                    if not self._spin(scan, base):
                        break
                with self._map_lock:
                    station = self._grid.spin_station(self._current_pose(), avoid=[*self._stations, *unreachable])
                    self._explore_target = station.target if station else None
                if station is None:
                    self._set_state("explored")
                    with self._lock:
                        self._plan = "nowhere left that a full turn would show much new"
                    break
                with self._lock:
                    self._plan = (
                        f"next spin {station.path_cm:.0f} cm away, could show {station.gain * CELL_AREA_M2:.1f} m² of new floor"
                    )
                    self._spin_progress = {**(self._spin_progress or {}), "next": [round(v, 1) for v in station.target]}
                outcome = self._drive_to_station(station.target, started_at)
                if outcome == "stopped":
                    break
                if outcome == "blocked":
                    unreachable.append(station.target)
                base = self._station_base()
                scan = self._sweep_here()
                if scan is None:
                    break
                # Spin only somewhere new: a blocked drive may have got nowhere.
                with self._lock:
                    here = (self._pose[0], self._pose[1])
                    spin_here = all(math.hypot(here[0] - x, here[1] - y) >= SPIN_AWAY_CM for x, y in self._stations)
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Spin-and-roam exploring stopped after an error")
            self._drive.stop()
            if not self._stop_event.is_set():
                with self._lock:
                    self._enabled = False
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.stop()
            if not self._stop_event.is_set():
                self._drive.set_inhibited(False)
            self._set_enabled_false()
            self._stop_mapping_watch()
            self._hold_camera(False)
            with self._map_lock:
                self._learner.finish(self._grid.known_count())
            try:
                save_learner(self._learner_path, self._learner)
            except OSError as exc:
                logger.warning("Could not save exploration learning: %s", exc)

    def _ping_ahead(self) -> float | None:
        """How far ahead is clear, drive locked meanwhile: a fan of lidar
        readings across the car's width (its beam is narrow), and the sonar's
        reading too when it is switched on (glass). None without the lidar."""
        if not self._drive.set_inhibited(True):
            return None
        status = self._pan_tilt.status()
        centre, tilt = int(status["pan"]), int(status["tilt"])
        self._pan_tilt.set_inhibited(True)
        try:
            if not self._ranging.lidar_ready():
                return None
            clear = self._lidar_fan_ahead(centre, tilt)
            if clear is not None and self._ranging.sonar_ready():
                # The sonar, when switched on, sees glass the lidar misses.
                sonar = self._ranging.read(self._range_sensor)
                if sonar is not None:
                    clear = min(clear, sonar)
            return clear
        finally:
            self._pan_tilt.set_inhibited(False)
            self._drive.set_inhibited(False)

    def _lidar_fan_ahead(self, centre: int, tilt: int) -> float | None:
        """Clear distance in a car-wide corridor ahead, from lidar readings
        every few degrees across it; None if straight ahead gave nothing."""
        assert self._lidar is not None
        pans = sorted(
            {
                min(PAN_LIMITS[1], max(PAN_LIMITS[0], int(round(bearing_pan(bearing)))))
                for bearing in range(-LIDAR_FAN_HALF_DEGREES, LIDAR_FAN_HALF_DEGREES + 1, LIDAR_FAN_STEP_DEGREES)
            }
            | {90}
        )
        items = []
        try:
            # Approached from one side, as sweeps are, so backlash is the same.
            self._pan_tilt.set_angles(max(PAN_LIMITS[0], pans[0] - 4), tilt, override=True)
            if self._stop_event.wait(SCAN_FIRST_SETTLE_SECONDS / 2):
                return None
            for pan in pans:
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._stop_event.wait(LIDAR_FAN_SETTLE_SECONDS):
                    return None
                items.append(
                    {"angle_deg": pan_bearing(pan), "distance_cm": self._ranging.read(self._lidar), "sensor": "lidar"}
                )
        finally:
            self._pan_tilt.set_angles(centre, tilt, override=True)
        return forward_clearance(items)

    def _step_seconds(self, distance_cm: float = AUTO_STEP_CM) -> float:
        """Burst length for an exploration step: from the forward bursts the
        accelerometer has measured, else from calibration when there is one."""
        with self._lock:
            calibration = self._calibration
            bursts = list(self._bursts)
        if len(bursts) >= BURSTS_NEEDED:
            learned = seconds_for(bursts, distance_cm)
            if learned is not None:
                return learned
        measured = calibration.step_duration(distance_cm) if calibration else None
        # Uncalibrated, scale the basic step; ramps make longer bursts go a
        # little further than this, which errs short.
        return measured if measured is not None else AUTO_FORWARD_SECONDS * distance_cm / AUTO_STEP_CM

    def _turn_and_align(self, degrees: float) -> bool:
        """Turn by `degrees`, then use the camera to measure and correct the turn."""
        if not self.camera_on_head or self._drive.gyro_ready():
            # Measuring needs pictures across the turn from the panning head;
            # with a gyro the turn is measured (and corrected) as it happens.
            return self._turn_by(degrees)
        pan_status = self._pan_tilt.status()
        centre_pan, tilt = int(pan_status["pan"]), int(pan_status["tilt"])
        sign = 1 if degrees >= 0 else -1
        # Photograph toward where the car is turning (lower pans look left).
        pans = [
            centre_pan - sign * step
            for step in range(0, AUTO_STRIP_REACH_DEGREES + 1, AUTO_STRIP_STEP_DEGREES)
            if PAN_LIMITS[0] <= centre_pan - sign * step <= PAN_LIMITS[1]
        ]
        strip = self._strip_from_sweep(pans, centre_pan, tilt)
        if strip is None:
            strip = self._capture_views(pans, tilt, centre_pan, frames=1)
        odometry_before = self._drive.odometry()[2]
        if not self._turn_by(degrees):
            return False
        actual: float | None = None
        current_frames: tuple[bytes, ...] | None = None
        corrections = 0
        for attempt in range(AUTO_ALIGN_MAX_CORRECTIONS + 1):
            current = self._capture_views([centre_pan], tilt, centre_pan, frames=AUTO_ALIGN_FRAMES)
            current_frames = current.get(centre_pan)
            actual = self._measure_turn(strip, current_frames, centre_pan)
            if actual is None:
                break
            odometry_turned = self._drive.odometry()[2] - odometry_before
            expected = odometry_turned * self._rotation_ratio(degrees)
            low, high = AUTO_ALIGN_PLAUSIBLE_RATIO
            if abs(expected) > 3.0 and not low <= actual / expected <= high:
                logger.info("Camera turn %.1f disagrees with wheels %.1f; trusting the wheels", actual, expected)
                actual = None
                break
            # Positive means the car went too far counterclockwise.
            error = actual - degrees
            if abs(error) <= AUTO_ALIGN_TOLERANCE_DEGREES or attempt == AUTO_ALIGN_MAX_CORRECTIONS:
                break
            # Correct with the turn ratio this very turn showed, so a badly
            # calibrated turn rate cannot make the correction overshoot too.
            ratio = actual / odometry_turned if abs(odometry_turned) > 2.0 else None
            correction = max(-AUTO_ALIGN_MAX_CORRECTION_DEGREES, min(AUTO_ALIGN_MAX_CORRECTION_DEGREES, -error))
            self._set_state("aligning")
            if not self._turn_by(correction, ratio if ratio is not None and 0.2 <= ratio <= 5.0 else None):
                return False
            corrections += 1
        self._show_turn(strip, current_frames, centre_pan, degrees, actual, corrections)
        if actual is None:
            return True
        odometry_turned = self._drive.odometry()[2] - odometry_before
        with self._lock:
            self._verified_turn = (self._verified_turn or 0.0) + actual
            translation_scale, rotation_scale = self._odometry_scale
            if abs(odometry_turned) > 5.0 and 0.3 <= actual / odometry_turned <= 3.0:
                low, high = WHEEL_ROTATION_SCALE_LIMITS
                rotation_scale = min(high, max(low, 0.7 * rotation_scale + 0.3 * actual / odometry_turned))
                self._odometry_scale = (translation_scale, rotation_scale)
        return True

    def _show_turn(
        self,
        strip: dict[int, tuple[bytes, ...]],
        current: tuple[bytes, ...] | None,
        centre_pan: int,
        planned: float,
        actual: float | None,
        corrections: int,
    ) -> None:
        if not strip or not current:
            return
        aim = actual if actual is not None else planned
        best_pan = min(strip, key=lambda pan: abs(centre_pan - pan - aim))
        lines = [
            f"planned {planned:+.0f} deg, camera measured "
            + (f"{actual:+.1f} deg" if actual is not None else "nothing usable (wheels trusted)"),
            f"left: strip picture at {centre_pan - best_pan:+d} deg; right: view after the turn"
            + (f"; {corrections} correction(s)" if corrections else ""),
        ]
        self._set_vision("turn", "Turn check", lines, lambda: render_matches(strip[best_pan][0], current[0], "Turn check", lines))

    def _recognize(
        self,
        image_jpeg: bytes | None,
        camera_bearing: float,
        scan: list[dict[str, object]],
        hfov_deg: float = CAMERA_HFOV_DEG,
    ) -> list[Sighting]:
        """Objects in a stop's picture, placed by bearing and sonar range.
        Recognition is a bonus: any trouble with it leaves the stop without."""
        if self._detector is None or not image_jpeg:
            return []
        try:
            frame = cv2.imdecode(np.frombuffer(image_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return []
            detections = self._detector.detect(frame)
            sightings = sightings_from(detections, frame.shape[1], camera_bearing, scan, hfov_deg)
        except Exception:
            logger.exception("Object recognition failed at this stop; carrying on without it")
            return []
        if detections:
            drawn = [
                {**detection, "dynamic": detection.get("label") in {s.label for s in sightings if s.dynamic}}
                for detection in detections
            ]
            lines = [
                ", ".join(
                    f"{s.label} {s.bearing_deg:+.0f} deg" + (f" {s.range_cm:.0f} cm" if s.range_cm is not None else "")
                    for s in sightings[:4]
                )
                or "nothing that matters for the map",
            ]
            if any(s.dynamic for s in sightings):
                lines.append("people/animals (orange) are kept out of the map")
            self._set_vision(
                "objects", "What the camera recognized", lines, lambda: render_detections(image_jpeg, drawn, "What the camera recognized", lines)
            )
        return sightings

    def _set_vision(self, name: str, title: str, lines: list[str], render: Callable[[], bytes | None]) -> None:
        """Store a dashboard picture; drawing problems never stop mapping."""
        try:
            jpeg = render()
        except Exception:
            logger.warning("Could not draw the %s picture", name, exc_info=True)
            return
        if jpeg:
            with self._lock:
                self._vision[name] = {
                    "title": title,
                    "caption": " · ".join(lines),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "jpeg": jpeg,
                }

    def coverage_png(self) -> bytes | None:
        """The coverage grid as a transparent PNG, north up."""
        with self._map_lock:
            if not self._grid.log_odds.size:
                return None
            image = self._grid.to_image()
        encoded, png = cv2.imencode(".png", image)
        return png.tobytes() if encoded else None

    def vision_panels(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {"name": name, "title": panel["title"], "caption": panel["caption"], "updated_at": panel["updated_at"]}
                for name, panel in self._vision.items()
            ]

    def vision_image(self, name: str) -> bytes | None:
        with self._lock:
            panel = self._vision.get(name)
            return panel["jpeg"] if panel else None

    def _wait_until_steady(self, timeout: float = CAMERA_STEADY_TIMEOUT_SECONDS) -> bool:
        """Take quick frames until two in a row match: the car and head are still."""
        previous = _steadiness_frame(self._camera.capture_snapshot())
        if previous is None:
            return True
        deadline = time.monotonic() + timeout
        earlier: np.ndarray | None = None
        while time.monotonic() < deadline:
            if self._stop_event.wait(CAMERA_STEADY_INTERVAL_SECONDS):
                return False
            current = _steadiness_frame(self._camera.capture_snapshot())
            if current is None:
                return True
            change = float(np.abs(current - previous).mean())
            if change < CAMERA_STEADY_DIFFERENCE:
                return True
            # In dim light sensor noise alone exceeds the threshold. Noise does
            # not build up over two frames the way movement does, so a change
            # from two frames back no bigger than from one frame back means
            # the view is still and only noisy.
            if earlier is not None:
                two_back = float(np.abs(current - earlier).mean())
                if two_back <= CAMERA_NOISE_GROWTH * change:
                    return True
            earlier, previous = previous, current
        logger.info("Camera still changing after %.1f s; taking the picture anyway", timeout)
        return False

    def _capture_best_picture(
        self, steady_timeout: float = CAMERA_STEADY_TIMEOUT_SECONDS, *, already_steady: bool = False
    ) -> tuple[bytes | None, dict[str, object] | None]:
        """Take a picture, check its levels, and retake with adjusted exposure
        until it is decent or the settings run out; keep the best one.

        `already_steady` skips the wait for matching frames first (the caller
        has held everything still long enough); a blurred picture is still
        retaken after waiting for steadiness."""
        set_exposure = getattr(self._camera, "set_exposure", None)
        if set_exposure is None:
            return self._camera.capture_snapshot(), None
        with self._lock:
            index = self._exposure_index
        set_exposure(EXPOSURE_LADDER[index])
        if not already_steady:
            self._wait_until_steady(steady_timeout)
        tried: dict[int, PictureQuality] = {}
        retook_blurred = False
        best: tuple[int, bytes, PictureQuality] | None = None
        for _ in range(EXPOSURE_MAX_ATTEMPTS):
            set_exposure(EXPOSURE_LADDER[index])
            jpeg = self._camera.capture_snapshot()
            if jpeg is None:
                break
            quality = assess_picture(jpeg)
            if quality is None:
                # Not a picture we can measure; keep it rather than nothing.
                return jpeg, None
            tried[index] = quality
            if best is None or quality.score > best[2].score:
                best = (index, jpeg, quality)
            if quality.decent:
                break
            if quality.blurred and not quality.too_dark and not quality.too_bright and not retook_blurred:
                # Still moving or vibrating: wait and retake at the same exposure.
                retook_blurred = True
                del tried[index]
                self._wait_until_steady(steady_timeout)
                continue
            if quality.too_dark:
                step = 1
            elif quality.too_bright:
                step = -1
            else:
                # Levels are fine but detail is thin: try pulling more out of
                # the shadows, or the highlights in a bright picture.
                step = 1 if quality.brightness < 120 else -1
            following = index + step
            if not 0 <= following < len(EXPOSURE_LADDER) or following in tried:
                break
            index = following
        if best is None:
            return None, None
        best_index, jpeg, quality = best
        set_exposure(EXPOSURE_LADDER[best_index])
        with self._lock:
            self._exposure_index = best_index
        info: dict[str, object] = {
            "setting": EXPOSURE_LADDER[best_index],
            "attempts": len(tried),
            **quality.summary(),
        }
        return jpeg, info

    @staticmethod
    def _measure_turn(
        strip: dict[int, tuple[bytes, ...]],
        current: tuple[bytes, ...] | None,
        centre_pan: int,
    ) -> float | None:
        """Combine the turn each strip picture implies: its direction relative
        to the centre view plus the rotation still left between it and the
        current forward view."""
        if not current:
            return None
        readings: list[tuple[float, float]] = []
        for pan, frames in strip.items():
            key = (pan, 0)
            visual = combine_view_estimates(estimate_view_motion({key: frames}, {key: current}))
            if visual is not None and abs(visual.yaw_deg) <= AUTO_STRIP_MAX_YAW_DEGREES:
                # Pictures the car ended up nearly facing overlap most.
                offset = pan_bearing(pan) - pan_bearing(centre_pan)
                readings.append((offset + visual.yaw_deg, 1.0 / (1.0 + abs(visual.yaw_deg))))
        if not readings:
            return None
        agreeing = max(
            (
                [other for other in readings if abs(other[0] - reading[0]) <= AUTO_STRIP_AGREEMENT_DEGREES]
                for reading in readings
            ),
            key=lambda group: (len(group), sum(weight for _, weight in group)),
        )
        if len(agreeing) < 2:
            return None
        total = sum(weight for _, weight in agreeing)
        return sum(turn * weight for turn, weight in agreeing) / total

    def _capture_views(
        self,
        pans: Sequence[int],
        tilt: int,
        return_pan: int,
        frames: int,
    ) -> dict[int, tuple[bytes, ...]]:
        """Photograph at each pan in ascending order, approaching every position
        from below so servo backlash is the same for every picture.

        Each picture gets the same exposure check as a stop's picture: retaken
        brighter, darker or steadier until decent. It starts from the setting
        that last worked, so in even light that is usually one shot, and any
        extra frames reuse the setting chosen."""
        views: dict[int, tuple[bytes, ...]] = {}
        if not pans:
            return views
        ordered = sorted(pans)
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(max(PAN_LIMITS[0], ordered[0] - 6), tilt, override=True)
            for pan in ordered:
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._stop_event.wait(AUTO_ALIGN_SETTLE_SECONDS):
                    return views
                first, _ = self._capture_best_picture(AUTO_VIEW_STEADY_TIMEOUT_SECONDS)
                pictures = [first] + [self._camera.capture_snapshot() for _ in range(frames - 1)]
                if all(picture is not None for picture in pictures):
                    views[pan] = tuple(pictures)
            return views
        finally:
            try:
                if ordered[-1] != return_pan or len(ordered) > 1:
                    self._pan_tilt.set_angles(max(PAN_LIMITS[0], return_pan - 6), tilt, override=True)
                    self._pan_tilt.set_angles(return_pan, tilt, override=True)
            finally:
                self._pan_tilt.set_inhibited(False)

    def _save_learned(self) -> None:
        """Keep what turning and moving have learned, beside the database."""
        if self._coast_path is None:
            return
        imu = self._drive.imu
        forward = imu.forward_axis() if imu is not None else None
        data = {
            "coast": self._coast.to_json(),
            "turn_power": self._turn_power.to_json(),
            "learned_boost": round(self._drive.learned_boost(), 3),
            "mapping_methods": self._records.to_json(),
            "pan_scale": self._pan_scale_learned,
            "forward_axis": list(forward) if forward is not None else None,
            "bursts": [[round(seconds, 3), round(cm, 1)] for seconds, cm in self._bursts],
        }
        try:
            temporary = self._coast_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data))
            temporary.replace(self._coast_path)
        except OSError as exc:
            logger.warning("Could not save what turning learned: %s", exc)

    def _initial_rotation_scale(self, source: str) -> float:
        """Real degrees per odometry degree to start from, for turns measured
        by `source` ("gyro" or "wheels"): calibration's, if it measured that."""
        calibration = getattr(self, "_calibration", None)
        if source == "wheels" and calibration is not None and calibration.turn_source == source and calibration.mean_turn_ratio():
            return float(calibration.mean_turn_ratio())
        # A gyro's own scale is right to a few percent; lidar fits refine it.
        return 1.0

    def _follow_heading_source(self) -> str:
        """What measures turns now; the rotation scale starts again when that
        changes (and likewise distances and the translation scale)."""
        source = self._drive.heading_source()
        moves = "accelerometer" if self._drive.measures_distance() else "wheels"
        with self._lock:
            if source != self._rotation_source:
                logger.info("Turns are now measured by the %s", source)
                self._rotation_source = source
                self._odometry_scale = (self._odometry_scale[0], self._initial_rotation_scale(source))
            if moves != self._translation_source:
                logger.info("Distances are now measured by the %s", moves)
                self._translation_source = moves
                self._odometry_scale = (self._initial_translation_scale(moves), self._odometry_scale[1])
        return source

    def _initial_translation_scale(self, source: str) -> float:
        calibration = getattr(self, "_calibration", None)
        if source == "wheels" and calibration is not None and calibration.forward_ratio:
            return float(calibration.forward_ratio)
        return 1.0

    def _rotation_limits(self) -> tuple[float, float]:
        with self._lock:
            source = self._rotation_source
        return GYRO_ROTATION_SCALE_LIMITS if source == "gyro" else WHEEL_ROTATION_SCALE_LIMITS

    def _rotation_ratio(self, degrees: float) -> float:
        """Real degrees per odometry degree for a turn of this size and direction:
        the calibrated ratio, nudged by what this session has learned since."""
        source = self._follow_heading_source()
        with self._lock:
            session = self._odometry_scale[1]
            calibration = self._calibration
        if source == "gyro" or calibration is None or calibration.turn_source != source:
            return session
        reference = calibration.mean_turn_ratio()
        measured = calibration.turn_ratio(degrees)
        if measured is not None and reference:
            return measured * session / reference
        return session

    @property
    def hunt_camera_on_head(self) -> bool:
        """Whether the hunt's camera turns with the head (only when it is the head camera)."""
        return self.camera_on_head and self._hunt_camera is self._camera

    def _picture_bearing(self, pan: int) -> float:
        """Which way, from the car's heading, a picture taken with the head at `pan` looks."""
        return pan_bearing(pan) if self.camera_on_head else 0.0

    def _halt_wait(self, seconds: float) -> bool:
        """Wait; True if a stop or an interrupt (the hunt saw the cat) cut it short."""
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            if self._interrupt.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._stop_event.is_set()
            if self._stop_event.wait(min(remaining, 0.02)):
                return True

    def _turn_by(self, degrees: float, rotation_scale: float | None = None, *, correct: bool = True) -> bool:
        """Rotate in place by about `degrees` (counterclockwise), closed loop on
        the odometry's heading. With a gyro that heading is measured, so what
        is left after the turn settles is corrected with gentler turns
        (unless `correct` is off: a quick turn, as the hunt makes)."""
        if self._stop_event.is_set():
            return False
        if rotation_scale is None:
            rotation_scale = self._rotation_ratio(degrees)
        # Odometry under-reads or over-reads turns by the learned scale.
        target = degrees / max(rotation_scale, 1e-3)
        gyro = self._drive.gyro_ready()
        start_heading = self._drive.odometry()[2]
        if gyro:
            rate = GYRO_BIG_TURN_RATE_DPS if abs(degrees) >= GYRO_BIG_TURN_DEGREES else GYRO_TURN_RATE_DPS
            command = self._turn_power.command_for(rate)
        else:
            command = AUTO_TURN_COMMAND
        for attempt in range(GYRO_TURN_MAX_CORRECTIONS + 1 if gyro and correct else 1):
            remaining = target - (self._drive.odometry()[2] - start_heading)
            if attempt and abs(remaining) * rotation_scale <= GYRO_TURN_TOLERANCE_DEGREES:
                break
            outcome = self._turn_once(remaining, command, gyro)
            if outcome == "stopped":
                return False
            if outcome in ("interrupted", "stalled") or not gyro:
                break
            command = self._turn_power.command_for(GYRO_CORRECTION_RATE_DPS)
        return True

    def _wait_until_still(self) -> bool:
        """Wait for a gyro turn's coast to die away; False if stopped or interrupted."""
        deadline = time.monotonic() + GYRO_SETTLE_MAX_SECONDS
        previous = self._drive.odometry()[2]
        while time.monotonic() < deadline:
            if self._halt_wait(0.1):
                return False
            heading = self._drive.odometry()[2]
            if abs(heading - previous) < GYRO_SETTLED_DEGREES:
                return True
            previous = heading
        return True

    def _turn_once(self, degrees: float, command: float, gyro: bool) -> str:
        """One turn by `degrees` of odometry heading: "done", "timeout",
        "stalled" (the gyro saw it not turning), "interrupted" or "stopped"."""
        if self._interrupt.is_set():
            return "interrupted"
        target = abs(degrees)
        steering = math.copysign(command, degrees)
        peak_rate = TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * DriveController._wheel_speed_fraction(
            float(self._drive.status()["max_duty"]) * command
        )
        start_heading = self._drive.odometry()[2]
        started = time.monotonic()
        previous_time, previous_heading = started, start_heading
        cut_turned = cut_rate = fastest = 0.0
        stall_check = (started, start_heading)
        outcome = "done"
        try:
            self._drive.set_command(0, 0, steering)
            while True:
                if self._halt_wait(0.01):
                    # A stop fails the turn; an interrupt just ends it early.
                    return "stopped" if self._stop_event.is_set() else "interrupted"
                now = time.monotonic()
                heading = self._drive.odometry()[2]
                rate = abs(heading - previous_heading) / max(now - previous_time, 1e-3)
                previous_time, previous_heading = now, heading
                turned = abs(heading - start_heading)
                fastest = max(fastest, rate)
                if gyro:
                    # Measured turns: what this car has been seen to coast.
                    coast = self._coast.predict(rate)
                    if now - stall_check[0] >= GYRO_STALL_SECONDS:
                        if abs(heading - stall_check[1]) < GYRO_STALL_DEGREES:
                            # Not turning at this power: learn it, and push harder.
                            self._turn_power.note_stall(command)
                            if command >= TurnPower.MAX_COMMAND - 1e-9:
                                logger.warning("Turn stuck: the gyro saw %.1f degrees at full power", turned)
                                return "stalled"
                            command = min(TurnPower.MAX_COMMAND, command + GYRO_STALL_POWER_STEP)
                            steering = math.copysign(command, degrees)
                            logger.info("Turn not starting; power up to %.2f", command)
                        stall_check = (now, heading)
                else:
                    # The motors ramp down from their current speed after the
                    # stop, adding rotation in proportion to how fast it is turning.
                    coast = rate * MOTOR_RAMP_SECONDS * min(1.0, rate / max(peak_rate, 1e-3)) / 2
                if turned + coast >= target:
                    cut_turned, cut_rate = turned, rate
                    break
                if now - started >= AUTO_MAX_TURN_SECONDS:
                    outcome = "timeout"
                    break
                self._drive.set_command(0, 0, steering)
        finally:
            # A gyro turn cuts the motors at once: it stops where measured.
            self._drive.stop(ramp=not gyro)
            if not self._drive.wait_until_stopped():
                raise RuntimeError("Drive output did not stop after the ramp-down interval")
        if gyro:
            if not self._wait_until_still():
                return "stopped" if self._stop_event.is_set() else "interrupted"
            if outcome == "done":
                coasted = abs(self._drive.odometry()[2] - start_heading) - cut_turned
                self._coast.learn(cut_rate, max(coasted, 0.0))
                if time.monotonic() - started >= GYRO_LEARN_POWER_SECONDS and fastest > 0:
                    self._turn_power.note_turning(command, fastest)
                if self._coast.observations % COAST_SAVE_EVERY == 0:
                    self._save_learned()
        return outcome

    def _move_burst(
        self,
        throttle: float,
        duration: float,
        *,
        steering: float = 0,
        strafe: float = 0,
        watch: bool = False,
    ) -> bool:
        """Drive for `duration`; True unless stopped. With `watch` (long
        drives straight ahead) the lidar, gyro and accelerometer are watched
        all the way and the move is ended early if anything goes wrong (see
        WATCH_STOP_MARGIN_CM); why is kept in self._move_abort."""
        self._move_abort = None
        if self._stop_event.is_set():
            return False
        if self._interrupt.is_set():
            return True
        with self._lock:
            self._last_move = None
        token = self._drive.begin_move()
        completed = False
        start_heading = self._drive.odometry()[2]
        moving_seen = time.monotonic()
        try:
            started = time.monotonic()
            self._drive.set_command(throttle, strafe, steering)
            deadline = started + duration
            refresh_interval = min(
                float(self._drive.status()["watchdog_timeout_seconds"]) / 2,
                0.05,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    completed = True
                    return True
                if self._halt_wait(min(refresh_interval, remaining)):
                    return not self._stop_event.is_set()
                knock = self._drive.knock_since(started + BUMP_IGNORE_START_SECONDS)
                if knock >= BUMP_STOP_G:
                    # Ran into something the sweep did not show (or something
                    # ran into the car): stop pushing; the move is over.
                    logger.warning("Bumped into something (%.1f g); move ended early", knock)
                    with self._lock:
                        self._last_bump = {
                            "g": round(knock, 2),
                            "at": datetime.now(timezone.utc).isoformat(),
                            "command": [round(throttle, 2), round(strafe, 2), round(steering, 2)],
                        }
                    return True
                if watch:
                    reason = self._watch_problem(started, start_heading, moving_seen)
                    if reason == "moving":
                        moving_seen = time.monotonic()
                    elif reason is not None:
                        logger.info("Move ended early: %s", reason)
                        self._move_abort = reason
                        return True
                if time.monotonic() < deadline:
                    self._drive.set_command(throttle, strafe, steering)
        finally:
            self._drive.stop()
            if not self._drive.wait_until_stopped():
                raise RuntimeError("Drive output did not stop after the ramp-down interval")
            if token is not None:
                self._finish_measured_move(token, throttle, strafe, steering, duration if completed else None)

    def _watch_problem(self, started: float, start_heading: float, moving_seen: float) -> str | None:
        """What is wrong with a watched drive now: None if nothing ("moving"
        when the IMU sees it going, to note the time)."""
        ahead = self._ranging.now(precise=True)
        if ahead is not None and ahead < AUTO_WALL_STANDOFF_CM + WATCH_STOP_MARGIN_CM:
            return f"something {ahead:.0f} cm ahead"
        if self._drive.gyro_ready():
            veer = abs(self._drive.odometry()[2] - start_heading)
            if veer > WATCH_MAX_VEER_DEGREES:
                return f"pushed {veer:.0f}° off course"
        imu = self._drive.imu
        if imu is not None and imu.ready():
            now = time.monotonic()
            if imu.speed_cm_s() >= WATCH_STALL_SPEED_CM_S:
                return "moving"
            if now - started >= WATCH_STALL_SECONDS and now - moving_seen >= WATCH_STALL_SECONDS:
                return "not moving although driving (stuck?)"
        return None

    def _travel_seconds(self, distance_cm: float) -> float:
        """Burst length to drive `distance_cm` straight ahead, however far."""
        if distance_cm <= TRAVEL_FROM_STEPS_CM:
            return self._step_seconds(distance_cm)
        with self._lock:
            bursts = list(self._bursts)
        seconds = None
        if len(bursts) >= BURSTS_NEEDED and len({round(s, 3) for s, _ in bursts}) >= 2:
            count = len(bursts)
            mean_s = sum(s for s, _ in bursts) / count
            mean_d = sum(d for _, d in bursts) / count
            spread = sum((s - mean_s) ** 2 for s, _ in bursts)
            slope = sum((s - mean_s) * (d - mean_d) for s, d in bursts) / spread if spread > 0 else 0.0
            if slope > 0:
                seconds = mean_s + (distance_cm - mean_d) / slope
        if seconds is None:
            speed = FORWARD_CM_PER_SECOND_AT_FULL_DUTY * DriveController._wheel_speed_fraction(
                float(self._drive.status()["max_duty"]) * AUTO_FORWARD_COMMAND
            )
            seconds = distance_cm / max(speed, 1.0) + MOTOR_RAMP_SECONDS / 2
        return min(TRAVEL_MAX_SECONDS, max(0.05, seconds))

    def _finish_measured_move(self, token, throttle: float, strafe: float, steering: float, seconds: float | None) -> None:
        """Close the accelerometer's measure of a burst once the car is at rest,
        and learn burst lengths from straight forward ones that ran their
        whole length (`seconds`, None if cut short)."""
        self._stop_event.wait(MOVE_SETTLE_SECONDS)
        straight = abs(steering) < 1e-6
        before = self._drive.odometry()
        measure, used = self._drive.finish_move(token, throttle if straight else 0.0, strafe if straight else 0.0)
        if measure is None:
            return
        wheels = math.hypot(before[0] - token[0][0], before[1] - token[0][1])
        summary = {**measure.summary(), "used": used, "wheels_cm": round(wheels, 1)}
        with self._lock:
            self._last_move = summary
            forward = straight and abs(strafe) < 1e-6 and abs(throttle - AUTO_FORWARD_COMMAND) < 1e-6
            if used and forward and seconds is not None and measure.ahead_cm:
                self._bursts = (self._bursts + [(seconds, float(measure.ahead_cm))])[-BURSTS_KEPT:]
                learned_burst = True
            else:
                learned_burst = False
        if learned_burst or (measure.ahead_cm is None and self._drive.measures_distance()):
            self._save_learned()

    def _capture_observation(self, mode: str = "survey") -> list[dict[str, object]]:
        self._store.ensure_capacity()
        if self._halt_wait(CAMERA_SETTLE_SECONDS):
            return []
        pan_status = self._pan_tilt.status()
        original_pan = int(pan_status["pan"])
        original_tilt = int(pan_status["tilt"])
        image_jpeg, exposure = self._capture_best_picture()
        # With a separate forward camera, a picture from it too: furniture is
        # recognized in it, and it becomes a place the hunt's camera can
        # recognize later.
        forward_jpeg = self._hunt_camera.capture_snapshot() if self._hunt_camera is not self._camera else None
        if image_jpeg is None:
            raise RuntimeError("Camera snapshot failed; see camera status for details")
        if exposure is not None:
            exposure_lines = [
                f"{exposure_label(exposure['setting'])} after {exposure['attempts']} tr"
                + ("y" if exposure["attempts"] == 1 else "ies")
                + ("" if exposure["decent"] else " (best available)"),
                f"brightness {exposure['brightness']:.0f}/255, contrast {exposure['contrast']:.0f}, "
                f"clipped {100 * float(exposure['clipped']):.1f}%",
            ]
            self._set_vision(
                "stop", "Stop picture", exposure_lines, lambda: render_features(image_jpeg, "Stop picture", exposure_lines)
            )
        centre_frames = [image_jpeg]
        for _ in range(CAMERA_MOTION_FRAME_COUNT - 1):
            if self._stop_event.wait(CAMERA_MOTION_FRAME_INTERVAL_SECONDS):
                return []
            frame = self._camera.capture_snapshot()
            if frame is None:
                logger.warning("Extra camera frame unavailable for motion comparison")
                continue
            centre_frames.append(frame)
        # A fixed camera's picture faces ahead whichever way the head points.
        centre_key = (original_pan if self.camera_on_head else STRIP_CENTRE_PAN, original_tilt)
        views: dict[ViewKey, tuple[bytes, ...]] = {centre_key: tuple(centre_frames)}

        odometry = self._drive.odometry()
        with self._lock:
            previous_views = self._last_camera_views
            previous_odometry = self._last_odometry
            translation_scale, rotation_scale = self._odometry_scale
        raw_odometry = relative(previous_odometry, odometry) if previous_odometry else (0.0, 0.0, 0.0)
        odometry_delta = (
            raw_odometry[0] * translation_scale,
            raw_odometry[1] * translation_scale,
            raw_odometry[2] * (self._rotation_ratio(raw_odometry[2] * rotation_scale) if raw_odometry[2] else rotation_scale),
        )
        view_estimates = estimate_view_motion(previous_views or {}, views)
        texture = image_texture_score(image_jpeg)
        # Take side pictures during the sweep when the forward view is too
        # bland, failed to measure motion, or disagrees with the wheels, and
        # keep taking them while the previous stop needed them too.
        need_side_views = texture < TEXTURE_POOR_SCORE or (
            previous_views is not None
            and (
                not view_estimates
                or predict_motion(
                    odometry_delta, combine_view_estimates(view_estimates), gyro=self._drive.heading_source() == "gyro"
                ).visual_rejected
                or any(key != centre_key for key in previous_views)
            )
        )
        side_pans = (
            tuple(pan for pan in SIDE_VIEW_PANS if pan != original_pan) if need_side_views and self.camera_on_head else ()
        )
        # A strip of pictures across the head's range, taken on the way past
        # during the sweep: a turn while exploring may reuse it, and the
        # dashboard shows it as a panorama.
        # (The hunt's camera watcher sees during its sweeps instead.)
        strip_pans = self._sweep_strip_pans() if mode in ("explore", "survey") and self.camera_on_head else []
        scan, side_views = self._sweep(original_tilt, original_pan, side_pans, strip_pans)
        if self._sweep_strip is not None:
            # The car stands still for a stop, so the odometry read for it holds.
            self._sweep_strip = replace(self._sweep_strip, odometry=odometry)
            self._show_panorama(self._sweep_strip)
        if scan is None:
            return []
        # What the camera recognizes: people and animals are kept out of the
        # map (mapping_scan), furniture becomes landmarks.
        if forward_jpeg is not None:
            sightings = self._recognize(forward_jpeg, 0.0, scan, camera_hfov(self._hunt_camera))
        else:
            sightings = self._recognize(image_jpeg, self._picture_bearing(original_pan), scan, camera_hfov(self._camera))
        scan = mark_dynamic(scan, sightings)
        mapping_scan = static_scan(scan)
        if is_lidar_scan(scan):
            self._show_lidar_overlay(scan, forward_jpeg, image_jpeg, original_pan)
        if side_views:
            views.update(side_views)
            view_estimates += estimate_view_motion(previous_views or {}, side_views)
        visual = combine_view_estimates(view_estimates)
        with self._lock:
            verified_turn, self._verified_turn = self._verified_turn, None
        if visual is None and verified_turn is not None:
            # The turn was too large for the views to overlap, but the camera
            # already measured it against the destination view.
            visual = VisualMotion(verified_turn, None, AUTO_ALIGN_FRAMES, 0.0)
        prediction_source = self._drive.heading_source()
        self._learn_pan_scale(scan, raw_odometry, prediction_source)
        prediction = predict_motion(odometry_delta, visual, gyro=prediction_source == "gyro")
        previous_centre = (previous_views or {}).get(centre_key)
        if previous_centre:
            motion_lines = [
                "left: previous stop; right: this stop",
                "camera turn "
                + (f"{visual.yaw_deg:+.1f} deg" if visual is not None else "not measurable")
                + (" (used)" if prediction.visual_used else " (rejected)" if prediction.visual_rejected else "")
                + f"; {self._drive.heading_source()} {odometry_delta[2]:+.1f} deg",
            ]
            self._set_vision(
                "motion",
                "Stop to stop",
                motion_lines,
                lambda: render_matches(previous_centre[0], image_jpeg, "Stop to stop", motion_lines),
            )

        with self._lock:
            run_id = self._run_id
            previous_pose, previous_covariance = self._pose, self._pose_covariance
        if run_id is None:
            raise RuntimeError("Mapping run ID was not initialized")
        if previous_odometry is None:
            prior_pose, prior_covariance = previous_pose, previous_covariance
        else:
            prior_pose, prior_covariance = propagate(previous_pose, previous_covariance, prediction)

        with self._map_lock:
            echoes = [echo for run in extract_echoes(mapping_scan, self._surface_map.beam_width_deg) for echo in run]
            registration = register_scan(echoes, prior_pose, prior_covariance, self._surface_map)
        pose, covariance = registration.pose, registration.covariance
        grid_match = None
        grid_fused = False
        if not registration.accepted:
            # Walls too few or too cluttered to line up: fit the whole sweep
            # to the coverage map instead (furniture and corners count too).
            grid_match = self._match_to_grid(mapping_scan, prior_pose, prior_covariance)
            if grid_match is not None and grid_match.good:
                match_covariance = LIDAR_GRID_MATCH_COVARIANCE if is_lidar_scan(mapping_scan) else GRID_MATCH_COVARIANCE
                fused = fuse_pose(prior_pose, prior_covariance, grid_match.pose, match_covariance, GRID_MATCH_GATE)
                if fused is not None:
                    pose, covariance = fused
                    grid_fused = True
        fixed = registration.accepted or grid_fused
        position_sigma, heading_sigma = pose_sigma(covariance)
        scale = self._learned_odometry_scale(raw_odometry, previous_pose, pose, covariance, prediction, registration.accepted)

        with self._map_lock:
            counts = (len(self._surface_map.walls()), len(self._surface_map.features()))
        estimate = {
            "odometry_delta": [round(value, 2) for value in raw_odometry],
            "odometry_scale": [round(value, 3) for value in scale],
            "visual": None
            if visual is None
            else {
                "yaw_deg": round(visual.yaw_deg, 2),
                "travel_bearing_deg": None
                if visual.travel_bearing_deg is None
                else round(visual.travel_bearing_deg, 1),
                "estimates": visual.estimates,
                "spread_deg": round(visual.spread_deg, 2),
                "views": sorted({key[0] for key, _ in view_estimates}),
            },
            "visual_used": prediction.visual_used,
            "visual_rejected": prediction.visual_rejected,
            "travel_bearing_used": prediction.bearing_used,
            "texture_score": round(texture, 1),
            "side_views": sorted(key[0] for key in side_views),
            "predicted_pose": [round(value, 2) for value in prior_pose],
            "registration": registration.summary(),
            "grid_match": None if grid_match is None else {**grid_match.summary(), "used": grid_fused},
            "pose_sigma": [round(position_sigma, 2), round(heading_sigma, 2)],
            "surfaces_before": {"walls": counts[0], "features": counts[1]},
            "exposure": exposure,
            "detections": [sighting.to_json() for sighting in sightings],
        }
        with self._lock:
            previous_node = self._last_node
            joining, self._joining_run = self._joining_run, False
        edge = None
        if previous_node is not None:
            sigma = self._edge_sigma(prediction, fixed)
            if joining:
                sigma = (
                    max(sigma[0], CONTINUE_POSITION_SIGMA_CM),
                    max(sigma[1], CONTINUE_POSITION_SIGMA_CM),
                    max(sigma[2], CONTINUE_HEADING_SIGMA_DEG),
                )
            # From where the previous stop sits in the graph (the hunt may
            # have corrected the pose since, from what its camera saw).
            with self._places_lock:
                anchor = self._places.graph.nodes.get(previous_node, previous_pose)
            edge = (previous_node, relative(anchor, pose), sigma)
            estimate["edge"] = {
                "from": previous_node,
                "delta": [round(value, 3) for value in edge[1]],
                "sigma": [round(value, 3) for value in edge[2]],
            }
        observation_id = self._store.save(image_jpeg, scan, run_id, mode, pose, estimate)

        # Remember this place and try to recognize it from earlier stops.
        if forward_jpeg is not None:
            forward_keyframe = make_keyframe(observation_id, run_id, forward_jpeg, 0.0, camera_hfov(self._hunt_camera))
            if forward_keyframe is not None:
                self._store.save_forward_keyframe(forward_keyframe)
                with self._places_lock:
                    self._forward_places.add(forward_keyframe)
        keyframe = make_keyframe(observation_id, run_id, image_jpeg, self._picture_bearing(original_pan), camera_hfov(self._camera))
        if keyframe is not None:
            self._store.save_keyframe(keyframe)
        with self._places_lock:
            update = self._places.add_stop(
                observation_id, run_id, pose, scan, (position_sigma, heading_sigma), keyframe, edge, sightings
            )
            event = update.event
            # The pose graph has the final word, using every constraint so far.
            pose = self._places.graph.nodes[observation_id]
            if event is not None:
                covariance = np.diag(
                    [event.sigma[0] ** 2, event.sigma[1] ** 2, math.radians(event.sigma[2]) ** 2]
                )
            elif update.views:
                covariance = covariance.copy()
                covariance[2, 2] = min(covariance[2, 2], math.radians(VIEW_HEADING_SIGMA_AFTER_MATCH) ** 2)
            place_counts = self._places.counts()
        rebuild = event is not None or update.moved_cm > MAP_REBUILD_MOVE_CM or update.moved_deg > MAP_REBUILD_TURN_DEG
        if event is not None:
            logger.info("Recognized place %s: %s", event.keyframe_id, event.summary())
            remembered = self._store.image(event.keyframe_id)
            if remembered:
                place_lines = [
                    f"left: remembered stop {event.keyframe_id}"
                    + (" from an earlier session" if event.kind == "relocalized" else "")
                    + "; right: now",
                    f"{event.inliers} camera inliers; pose moved "
                    f"{math.hypot(event.correction[0], event.correction[1]):.0f} cm, {event.correction[2]:+.1f} deg"
                    + ("; sonar confirmed" if event.metric else ""),
                ]
                self._set_vision(
                    "place",
                    "Recognized place",
                    place_lines,
                    lambda: render_matches(remembered, image_jpeg, "Recognized place", place_lines),
                )
        if rebuild:
            report = self._rebuild_map(observation_id, run_id)
        else:
            with self._map_lock:
                report = self._surface_map.integrate(mapping_scan, pose, position_sigma, heading_sigma)
                self._grid.integrate(mapping_scan, pose, self._surface_map.beam_width_deg)
                self._cloud.add(lidar_points(mapping_scan, pose))
                self._semantic.add(sightings, pose)
                self._path.append(pose)
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._pose, self._pose_covariance = pose, covariance
            if fixed or event is not None:
                self._stops_since_fix = 0
                self._travel_since_fix = (0.0, 0.0)
            elif self._stops_since_fix is not None:
                self._stops_since_fix += 1
                self._travel_since_fix = (
                    self._travel_since_fix[0] + math.hypot(odometry_delta[0], odometry_delta[1]),
                    self._travel_since_fix[1] + abs(odometry_delta[2]),
                )
            self._hazards = hazards_from(sightings)
            self._last_node = observation_id
            self._last_odometry = odometry
            self._last_scan = scan
            self._last_camera_views = views
            self._odometry_scale = scale
            self._place_counts = place_counts
            if event is not None:
                self._last_place = event.summary()
            self._last_localization = {
                "sensor": "lidar" if is_lidar_scan(scan) else "sonar",
                "turns_from": prediction_source,
                "visual_used": prediction.visual_used,
                "visual_rejected": prediction.visual_rejected,
                "side_views": bool(side_views),
                "registration": registration.summary(),
                "grid_match": None if grid_match is None else {**grid_match.summary(), "used": grid_fused},
                "place": event.summary() if event is not None else None,
                "views_matched": update.views,
                "graph_adjusted_cm": round(update.moved_cm, 1),
                "surfaces_created": report.created if not rebuild and report else None,
                "surfaces_refined": report.fused if not rebuild and report else None,
                "surfaces_retired": report.retired if not rebuild and report else None,
            }
            self._last_observation = captured_at
            self._last_id = observation_id
        return scan

    @staticmethod
    def _edge_sigma(prediction: MotionPrediction, registered: bool) -> tuple[float, float, float]:
        """Uncertainty of the motion between two stops, for the pose graph."""
        covariance = prediction.covariance
        factor = 0.5 if registered else 1.0
        return (
            factor * math.sqrt(covariance[0, 0]) + 0.5,
            factor * math.sqrt(covariance[1, 1]) + 0.5,
            max(0.3, factor * math.degrees(math.sqrt(covariance[2, 2]))),
        )

    # -- ranging -------------------------------------------------------------

    def _scan_angles(self) -> tuple[int, ...]:
        """Sweeps are lidar sweeps: none without a working lidar."""
        return LIDAR_SCAN_ANGLES if self._ranging.lidar_ready() else ()

    def _lidar_sample(self, pan: int) -> dict[str, object]:
        """One lidar sweep reading: the median of a few frames, with the
        signal strength and what the sonar says meanwhile (for glass)."""
        assert self._lidar is not None
        samples: list[float] = []
        strengths: list[float] = []
        errors: list[str] = []
        for index in range(LIDAR_SAMPLES_PER_ANGLE):
            if index:
                if self._halt_wait(LIDAR_SAMPLE_INTERVAL_SECONDS):
                    break
            try:
                distance = self._lidar.sample_now()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            strength = self._lidar.details().get("signal")
            if distance is None:
                errors.append("weak or saturated return")
                continue
            samples.append(distance)
            if isinstance(strength, (int, float)):
                strengths.append(float(strength))
        sonar = None
        if self._ranging.sonar_ready():
            try:
                sonar = self._range_sensor.sample_now()
            except Exception:
                sonar = None
        # Frames that disagree a lot straddle an edge: no reading there.
        usable = len(samples) >= min(2, LIDAR_SAMPLES_PER_ANGLE) and (
            max(samples) - min(samples) <= LIDAR_SAMPLE_SPREAD_CM + 0.03 * min(samples)
        )
        return {
            "angle_deg": round(pan_bearing(pan), 2),
            "pan": pan,
            "distance_cm": round(statistics.median(samples), 1) if usable else None,
            "samples": len(samples),
            "error": None if usable else (errors[-1] if errors else "frames disagree"),
            "sensor": "lidar",
            "signal": round(statistics.median(strengths)) if strengths else None,
            "sonar_cm": None if sonar is None else round(sonar, 1),
        }

    def _sweep(
        self,
        tilt: int,
        original_pan: int,
        view_pans: Sequence[int],
        strip_pans: Sequence[int] = (),
    ) -> tuple[list[dict[str, object]] | None, dict[ViewKey, tuple[bytes, ...]]]:
        """Sweep the lidar every degree, grabbing camera views at the requested
        pans and turn-strip pictures at `strip_pans` (kept in self._sweep_strip)."""
        scan: list[dict[str, object]] = []
        views: dict[ViewKey, tuple[bytes, ...]] = {}
        strip: dict[int, tuple[bytes, ...]] = {}
        self._sweep_strip = None
        scan_angles = set(self._scan_angles())
        pans = sorted(set(view_pans) | set(strip_pans) | scan_angles)
        # Side views are compared with the previous stop's, so they keep the
        # stop's exposure even if a strip picture needed another one.
        with self._lock:
            stop_exposure = self._exposure_index
        set_exposure = getattr(self._camera, "set_exposure", None)
        if not pans:
            return scan, views
        # The lidar steps a degree at a time and reads a few frames after each step.
        settle = LIDAR_STEP_SETTLE_SECONDS
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(SCAN_LEAD_IN_PAN, tilt, override=True)
            if self._halt_wait(SCAN_FIRST_SETTLE_SECONDS):
                return None, views
            for pan in pans:
                if self._stop_event.is_set() or self._interrupt.is_set():
                    return None, views
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._halt_wait(settle):
                    return None, views
                if pan in strip_pans:
                    # The same picture checks as a strip taken in a pass of
                    # its own, so turns are measured as well. The head has
                    # only stepped a few degrees and is held still at least
                    # as long as that pass would, so the up-front wait for
                    # matching frames is skipped (blur still forces a retake).
                    if self._stop_event.wait(max(0.0, AUTO_ALIGN_SETTLE_SECONDS - settle)):
                        return None, views
                    picture, _ = self._capture_best_picture(AUTO_VIEW_STEADY_TIMEOUT_SECONDS, already_steady=True)
                    if picture is not None:
                        strip[pan] = (picture,)
                if pan in view_pans:
                    if strip and set_exposure is not None:
                        set_exposure(EXPOSURE_LADDER[stop_exposure])
                    frame = self._camera.capture_snapshot()
                    if frame is not None:
                        views[(pan, tilt)] = (frame,)
                if pan not in scan_angles:
                    continue
                if (pan in strip_pans or pan in view_pans) and self._halt_wait(LIDAR_STEP_SETTLE_SECONDS):
                    return None, views
                scan.append(self._lidar_sample(pan))
                if self._stop_event.is_set():
                    return None, views
        finally:
            try:
                self._pan_tilt.set_angles(original_pan, tilt, override=True)
            finally:
                self._pan_tilt.set_inhibited(False)
        if strip:
            # Where the car stood is filled in by the caller, which has read it.
            self._sweep_strip = SweepStrip(strip, original_pan, tilt, (math.nan, math.nan, math.nan))
        return scan, views

    def _sweep_strip_pans(self) -> list[int]:
        """Pans for strip pictures during a sweep: every 10 degrees out to 60
        each side of straight ahead (wherever the head happened to point),
        moved to the sweep's nearest step."""
        stops = set(self._scan_angles()) | {SCAN_LEAD_IN_PAN} if self._ranging.enabled() else set()
        pans = set()
        for offset in range(-AUTO_STRIP_REACH_DEGREES, AUTO_STRIP_REACH_DEGREES + 1, AUTO_STRIP_STEP_DEGREES):
            wanted = STRIP_CENTRE_PAN + offset
            if not PAN_LIMITS[0] <= wanted <= PAN_LIMITS[1]:
                continue
            nearest = min(stops, key=lambda pan: abs(pan - wanted), default=wanted)
            pans.add(nearest if abs(nearest - wanted) <= SWEEP_STRIP_SNAP_DEG else wanted)
        return sorted(pans)

    def _show_lidar_overlay(
        self, scan: list[dict[str, object]], forward_jpeg: bytes | None, image_jpeg: bytes, original_pan: int
    ) -> None:
        """The stop's lidar readings drawn over the pictures: the forward
        camera's when there is one, and the head camera's."""
        valid = [item for item in scan if isinstance(item.get("distance_cm"), (int, float))]
        lines = [
            f"lidar: {len(valid)} of {len(scan)} directions measured"
            + (f", nearest {min(float(item['distance_cm']) for item in valid):.0f} cm" if valid else "")
        ]
        if forward_jpeg is not None:
            self._set_vision(
                "lidar",
                "Lidar over the forward picture",
                lines,
                lambda: render_lidar_overlay(forward_jpeg, scan, 0.0, camera_hfov(self._hunt_camera), "Lidar over the forward picture", lines),
            )
        else:
            self._set_vision(
                "lidar",
                "Lidar over the stop picture",
                lines,
                lambda: render_lidar_overlay(
                    image_jpeg, scan, self._picture_bearing(original_pan), camera_hfov(self._camera), "Lidar over the stop picture", lines
                ),
            )

    def _show_panorama(self, strip: SweepStrip) -> None:
        views = [(pan_bearing(pan), frames[0]) for pan, frames in strip.pictures.items() if frames]
        if len(views) < 2:
            return
        widest = max(bearing for bearing, _ in views) + PANORAMA_HFOV_DEGREES / 2
        lines = [f"{len(views)} pictures taken during the sonar sweep, joined: about {widest:.0f} deg each side of ahead"]
        self._set_vision(
            "panorama",
            "Sweep panorama",
            lines,
            lambda: render_panorama(views, PANORAMA_HFOV_DEGREES, "Sweep panorama", lines),
        )

    def _strip_from_sweep(self, pans: Sequence[int], centre_pan: int, tilt: int) -> dict[int, tuple[bytes, ...]] | None:
        """The sweep's pictures nearest each of `pans`, if every one was taken
        and the car and head have not moved since. Used once: the turn moves
        the car."""
        sweep_strip, self._sweep_strip = self._sweep_strip, None
        if sweep_strip is None or sweep_strip.centre_pan != centre_pan or sweep_strip.tilt != tilt:
            return None
        moved = relative(sweep_strip.odometry, self._drive.odometry())
        if math.hypot(moved[0], moved[1]) > SWEEP_STRIP_MAX_MOVE_CM or abs(moved[2]) > SWEEP_STRIP_MAX_TURN_DEG:
            return None
        strip: dict[int, tuple[bytes, ...]] = {}
        for wanted in pans:
            nearest = min(sweep_strip.pictures, key=lambda pan: abs(pan - wanted))
            if abs(nearest - wanted) > SWEEP_STRIP_SNAP_DEG:
                return None
            strip[nearest] = sweep_strip.pictures[nearest]
        return strip

    def _learn_pan_scale(self, scan: list[dict[str, object]], raw_odometry: Pose, source: str) -> None:
        """After a gyro-measured turn on the spot, see how far the lidar's view
        moved across the pan, and refine the pan servo's scale from it."""
        with self._lock:
            previous = self._last_scan
        if (
            previous is None
            or source != "gyro"
            or self._pan_tilt.status()["mode"] != "hardware"
            or math.hypot(raw_odometry[0], raw_odometry[1]) > PAN_SCALE_MAX_MOVE_CM
            or not PAN_SCALE_TURN_RANGE_DEG[0] <= abs(raw_odometry[2]) <= PAN_SCALE_TURN_RANGE_DEG[1]
        ):
            return
        estimate = estimate_pan_scale(previous, scan, raw_odometry[2])
        if estimate is None:
            return
        measured, error, readings = estimate
        current = pan_scale()
        updated = set_pan_scale(current + PAN_SCALE_BLEND * (measured - current))
        with self._lock:
            self._pan_scale_learned = updated
            self._pan_scale_samples += 1
        logger.info(
            "Pan servo scale %.3f from a %.0f-degree turn (%d readings, %.1f cm); now %.3f",
            measured, raw_odometry[2], readings, error, updated,
        )
        self._save_learned()

    def _learned_odometry_scale(
        self,
        raw_odometry: Pose,
        previous_pose: Pose,
        pose: Pose,
        covariance: np.ndarray,
        prediction: MotionPrediction,
        registered: bool,
    ) -> tuple[float, float]:
        """Nudge wheel odometry scale toward what sonar and camera measured."""
        with self._lock:
            translation_scale, rotation_scale = self._odometry_scale
        actual = relative(previous_pose, pose)
        distance = math.hypot(raw_odometry[0], raw_odometry[1])
        if registered and distance > 10.0:
            unit = np.array(raw_odometry[:2]) / distance
            world_unit = pose_rotation(previous_pose[2]) @ unit
            along_sigma = math.sqrt(max(float(world_unit @ covariance[:2, :2] @ world_unit), 0.0))
            if along_sigma < 0.2 * distance:
                ratio = float(np.array(actual[:2]) @ unit) / distance
                with self._lock:
                    measured = self._translation_source == "accelerometer"
                low, high = MEASURED_TRANSLATION_SCALE_LIMITS if measured else WHEEL_TRANSLATION_SCALE_LIMITS
                translation_scale = min(high, max(low, 0.8 * translation_scale + 0.2 * ratio))
        if (registered or prediction.visual_used) and abs(raw_odometry[2]) > 5.0:
            heading_sigma = math.degrees(math.sqrt(max(float(covariance[2, 2]), 0.0)))
            if heading_sigma < 0.2 * abs(raw_odometry[2]):
                ratio = wrap_degrees(actual[2]) / raw_odometry[2]
                low, high = self._rotation_limits()
                if 0.3 <= ratio <= 3.0:
                    rotation_scale = min(high, max(low, 0.8 * rotation_scale + 0.2 * ratio))
        return translation_scale, rotation_scale

    def _count_move(self) -> None:
        with self._lock:
            self._move_count += 1

    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._enabled:
                self._state = state

    def _set_enabled_false(self) -> None:
        with self._lock:
            self._enabled = False
