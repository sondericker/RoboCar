"""The cat hunt: find the cat in a mapped room, then play with it.

**Seeking.** A belief map says where the cat might be. It covers the floor
the coverage grid knows, and is weighted toward furniture cats like (a
couch, bed or chair the camera recognized) and toward spots where the cat
was found before. Every look with the camera lowers the belief over what
that look could actually see (walls block the view), and the belief slowly
drifts back toward its prior, because cats move. The next vantage point is
the one that could show the most belief per centimetre of travel. On the
way there the car checks ahead with a forward glance after every leg; on
arrival it does a full sonar sweep (re-checking the map and where it is)
and pans the camera across the room.

**Playing.** No canned routine. Each beat a policy picks one of many moves
(pounce, zig-zag, orbit, sidestep, spin, stalk, peekaboo...) with fresh
random parameters. It weighs what suits the distance to the cat, what the
car's slowly drifting mood calls for, what has not been done lately, and
what has learned to get the cat to react. After each move the cat's own
movement, with the car's movement taken out, scores that move. Every so
often the car backs off and waits about a minute, wiggling now and then,
before going back to bothering the cat. The scores, the camera's distance
calibration and the cat's favourite spots are kept between hunts.

**Safety.** The car never closes to less than MIN_CAT_DISTANCE_CM.
Forward moves are limited by the sonar and the map, sideways moves by a
glance to that side first, and reversing by space the map shows clear
(and never far). Nothing drives toward a person close ahead. A session
ends at its time limit, on a critically low battery, or on Stop.

Everything here talks to the car through HuntBody, so the same engine runs
the real car and the simulation the tests use.
"""

import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from robocar.exploration import CONFIRMED_ABOVE, CoverageGrid
from robocar.localization import Pose, compose, relative
from robocar.semantics import CAMERA_HFOV_DEG

logger = logging.getLogger(__name__)

CAT_LABELS = frozenset({"cat"})
# Distances, in cm from the sonar, kept from the cat.
MIN_CAT_DISTANCE_CM = 30.0
PLAY_NEAR_CM = 50.0
PLAY_FAR_CM = 110.0
PLAY_TARGET_CM = 75.0
# Camera: the OV5647's 640x480 mode is about 53 degrees across and 41 high.
CAMERA_VFOV_DEG = 41.0
# Distance from a cat's box height: distance = scale / box height (pixels).
# A sitting cat is about 25 cm tall; the scale is learned from the sonar.
DEFAULT_SIZE_SCALE = 25.0 * 240.0 / math.tan(math.radians(CAMERA_VFOV_DEG / 2))
SCALE_LIMITS = (0.5, 2.0)
SCALE_SAMPLES = 9
# A sonar range teaches the scale only when it agrees with the camera this
# well and the cat is this close (further out the beam catches walls first).
CALIBRATE_AGREEMENT = 0.25
CALIBRATE_WITHIN_CM = 150.0
# Seeking.
BELIEF_CELL_CM = 20.0
VIEW_RANGE_CM = 350.0
VANTAGE_SPACING_CM = 40.0
VANTAGE_OVERHEAD_CM = 120.0
BELIEF_DRIFT = 0.04  # share pulled back toward the prior after each look round
LOOK_PANS = (30, 60, 90, 120, 150)
SEEK_LEG_CM = 60.0
# Route legs are at least this long: a route's first step out of a wall's
# margin can be a few centimetres, too short to be worth a move on its own.
MIN_LEG_CM = 12.0
SEEK_STANDOFF_CM = 45.0
FURNITURE_CATS_LIKE = {"couch": 3.0, "bed": 3.0, "chair": 1.5, "potted plant": 1.2, "dining table": 1.5, "bench": 1.5}
# Playing.
BOUT_SECONDS = (40.0, 110.0)
REST_SECONDS = (45.0, 75.0)
REST_FIDGET_SECONDS = (8.0, 20.0)
LOST_AFTER_LOOKS = 3
HEAD_TRACK_LIMIT_DEG = 35.0  # turn the body once the head is this far off
SIDE_MARGIN_CM = 20.0  # sideways clearance kept, from the sonar on the head
BACK_LIMIT_CM = 25.0
MOVE_SECONDS_PER_CM = 1 / 60.0
REWARD_LEARNING_RATE = 0.25
NOVELTY_MEMORY = 6
SESSION_SECONDS = 20 * 60.0
# Driving near the cat needs a sighting this fresh, with the cat within this
# angle of ahead: the camera and sonar cannot see beside or behind the car.
FRESH_SECONDS = 1.5
CAT_AHEAD_DEG = 60.0
# ...unless the cat is known to be further than this (and seen this recently):
# then no move the car makes between looks can reach it.
CAT_FAR_CM = 120.0
CAT_FAR_FRESH_SECONDS = 5.0
# A vantage point the car could not get to is left alone this long.
UNREACHABLE_SECONDS = 120.0
UNREACHABLE_RADIUS_CM = 50.0


@dataclass(frozen=True)
class CatView:
    """The cat in one picture."""

    bearing_deg: float  # counterclockwise from the car's heading
    box_height: float  # pixels
    confidence: float
    box: tuple[int, int, int, int]
    frame_width: int


@dataclass(frozen=True)
class Look:
    """One picture: the cat if seen, and whether a person is close ahead."""

    cat: CatView | None
    person_ahead: bool = False


class HuntBody(Protocol):
    """What the hunt needs from the car (or a simulation of it)."""

    def now(self) -> float: ...

    def wait(self, seconds: float) -> bool: ...  # False when stopped

    def stopped(self) -> bool: ...

    def look(self, pan: int) -> Look: ...  # point the head, take a picture, detect

    def ping(self, pan: int) -> float | None: ...  # sonar range with the head at `pan`

    def observe(self) -> None: ...  # a full stop: sweep, localize, update the map

    def sweep_looks(self, pans: Sequence[int]) -> dict[int, Look]: ...  # looks from the last stop's sweep pictures

    def pose(self) -> Pose: ...  # best estimate on the map, now

    def odometry(self) -> Pose: ...

    def grid(self) -> CoverageGrid: ...

    def landmarks(self) -> list[tuple[str, float, float]]: ...

    def turn(self, degrees: float, precise: bool) -> bool: ...

    def move(self, forward_cm: float, sideways_cm: float, speed: float = 1.0) -> bool: ...  # sideways: + right

    def arc(self, strafe: float, steering: float, seconds: float) -> bool: ...

    def battery_ok(self) -> bool: ...

    def report(self, status: dict[str, object]) -> None: ...


def pan_bearing(pan: float) -> float:
    return 90.0 - pan


def pixel_bearing(column: float, width: int) -> float:
    focal = (width / 2) / math.tan(math.radians(CAMERA_HFOV_DEG / 2))
    return -math.degrees(math.atan((column - width / 2) / focal))


def cat_view_from(detections: Sequence[Mapping[str, object]], pan: int, frame_width: int, frame_height: int) -> Look:
    """The most confident cat in a picture's detections, and whether a
    person stands close in front."""
    cat = None
    person_ahead = False
    for detection in detections:
        x, y, width, height = (int(value) for value in detection["box"])
        label = str(detection.get("label"))
        bearing = pan_bearing(pan) + pixel_bearing(x + width / 2, frame_width)
        if label in CAT_LABELS and (cat is None or float(detection["confidence"]) > cat.confidence):
            cat = CatView(bearing, float(height), float(detection["confidence"]), (x, y, width, height), frame_width)
        elif label == "person" and abs(bearing) <= 25.0 and height >= 0.45 * frame_height:
            person_ahead = True
    return Look(cat, person_ahead)


# -- memory --------------------------------------------------------------------


@dataclass
class HuntMemory:
    """What hunts learn and keep: how well each move works, the camera's
    distance scale, and where the cat tends to be."""

    move_scores: dict[str, list[float]] = field(default_factory=dict)  # name: [mean reward, count]
    size_scale: float = DEFAULT_SIZE_SCALE
    # Recent box-height x sonar products; the scale is their median.
    scale_samples: list[float] = field(default_factory=list)
    cat_spots: list[list[float]] = field(default_factory=list)  # [x, y, weight]
    hunts: int = 0
    finds: int = 0

    def score(self, move: str) -> tuple[float, int]:
        mean, count = self.move_scores.get(move, [0.5, 0])
        return float(mean), int(count)

    def reward(self, move: str, value: float) -> None:
        mean, count = self.score(move)
        rate = max(REWARD_LEARNING_RATE, 1.0 / (count + 1))
        self.move_scores[move] = [round(mean + rate * (value - mean), 4), count + 1]

    def calibrate(self, box_height: float, sonar_cm: float) -> None:
        """Learn the box-size distance scale from a sonar range to the cat:
        the median of recent measurements, so a stray reading (the beam
        catching a wall first) cannot drag it about."""
        measured = box_height * sonar_cm
        if not SCALE_LIMITS[0] * DEFAULT_SIZE_SCALE <= measured <= SCALE_LIMITS[1] * DEFAULT_SIZE_SCALE:
            return
        self.scale_samples = (self.scale_samples + [measured])[-SCALE_SAMPLES:]
        ordered = sorted(self.scale_samples)
        self.size_scale = ordered[len(ordered) // 2]

    def remember_cat(self, x: float, y: float) -> None:
        for spot in self.cat_spots:
            if math.hypot(spot[0] - x, spot[1] - y) < 60.0:
                spot[0], spot[1] = 0.8 * spot[0] + 0.2 * x, 0.8 * spot[1] + 0.2 * y
                spot[2] = min(10.0, spot[2] + 1.0)
                return
        self.cat_spots.append([x, y, 1.0])
        # Old haunts fade so the newest habits count most.
        self.cat_spots = sorted(self.cat_spots, key=lambda spot: -spot[2])[:20]

    def fade(self) -> None:
        for spot in self.cat_spots:
            spot[2] *= 0.9

    def favourites(self, count: int = 3) -> list[dict[str, object]]:
        tried = [(name, mean, n) for name, (mean, n) in self.move_scores.items() if n >= 3]
        return [
            {"move": name, "score": round(mean, 2), "tries": int(n)}
            for name, mean, n in sorted(tried, key=lambda item: -item[1])[:count]
        ]

    def to_json(self) -> dict[str, object]:
        return {
            "move_scores": self.move_scores,
            "size_scale": round(self.size_scale, 1),
            "cat_spots": [[round(value, 1) for value in spot] for spot in self.cat_spots],
            "hunts": self.hunts,
            "finds": self.finds,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "HuntMemory":
        memory = cls()
        scores = data.get("move_scores")
        if isinstance(scores, Mapping):
            for name, value in scores.items():
                if isinstance(value, list) and len(value) == 2:
                    memory.move_scores[str(name)] = [float(value[0]), int(value[1])]
        scale = data.get("size_scale")
        if isinstance(scale, (int, float)) and SCALE_LIMITS[0] * DEFAULT_SIZE_SCALE <= scale <= SCALE_LIMITS[1] * DEFAULT_SIZE_SCALE:
            memory.size_scale = float(scale)
            memory.scale_samples = [float(scale)] * 3  # a start the next readings can move
        spots = data.get("cat_spots")
        if isinstance(spots, list):
            memory.cat_spots = [[float(v) for v in spot[:3]] for spot in spots if isinstance(spot, list) and len(spot) >= 3]
        memory.hunts = int(data.get("hunts") or 0)
        memory.finds = int(data.get("finds") or 0)
        return memory


def load_memory(path: Path | None) -> HuntMemory:
    if path is None or not path.exists():
        return HuntMemory()
    try:
        return HuntMemory.from_json(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Ignoring unreadable hunt memory %s: %s", path, exc)
        return HuntMemory()


def save_memory(path: Path | None, memory: HuntMemory) -> None:
    if path is None:
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(memory.to_json(), indent=1))
    temporary.replace(path)


# -- where the cat might be ------------------------------------------------------


class CatBelief:
    """A coarse map of where the cat might be, over the floor the grid knows."""

    def __init__(self, grid: CoverageGrid, landmarks: Sequence[tuple[str, float, float]], memory: HuntMemory) -> None:
        self.grid = grid
        low_x, low_y, high_x, high_y = grid.bounds() if grid.log_odds.size else (-100.0, -100.0, 100.0, 100.0)
        self.origin = np.array([low_x, low_y])
        self.shape = (max(1, int(math.ceil((high_y - low_y) / BELIEF_CELL_CM))), max(1, int(math.ceil((high_x - low_x) / BELIEF_CELL_CM))))
        ys = low_y + (np.arange(self.shape[0]) + 0.5) * BELIEF_CELL_CM
        xs = low_x + (np.arange(self.shape[1]) + 0.5) * BELIEF_CELL_CM
        self.centres_x, self.centres_y = np.meshgrid(xs, ys)
        prior = np.zeros(self.shape)
        if grid.log_odds.size:
            free, unknown = grid.free(), grid.unknown()
            for row in range(self.shape[0]):
                for column in range(self.shape[1]):
                    cell = grid.cell_of((self.centres_x[row, column], self.centres_y[row, column]))
                    if grid._inside(cell):
                        # Cats sit on the known floor, or hide where the map is unsure.
                        prior[row, column] = 1.0 if free[cell] else 0.25 if unknown[cell] else 0.0
        for label, x, y in landmarks:
            boost = FURNITURE_CATS_LIKE.get(label)
            if boost:
                near = np.hypot(self.centres_x - x, self.centres_y - y) < 100.0
                prior[near & (prior > 0)] *= boost
        for x, y, weight in memory.cat_spots:
            bump = 1.0 + 0.8 * weight * np.exp(-((self.centres_x - x) ** 2 + (self.centres_y - y) ** 2) / (2 * 60.0**2))
            prior[prior > 0] *= bump[prior > 0]
        total = prior.sum()
        self.prior = prior / total if total > 0 else prior
        self.belief = self.prior.copy()

    def _cell(self, point: Sequence[float]) -> tuple[int, int] | None:
        column = int((point[0] - self.origin[0]) // BELIEF_CELL_CM)
        row = int((point[1] - self.origin[1]) // BELIEF_CELL_CM)
        if 0 <= row < self.shape[0] and 0 <= column < self.shape[1]:
            return row, column
        return None

    def _walls(self) -> np.ndarray:
        return self.grid.log_odds > CONFIRMED_ABOVE if self.grid.log_odds.size else np.zeros((0, 0), bool)

    def visible(self, origin: Sequence[float], heading_deg: float, half_fov: float, step_deg: float = 2.0) -> set[tuple[int, int, int]]:
        """Belief cells a camera at `origin` facing `heading_deg` can see, with
        the distance (cm) to each."""
        walls = self._walls()
        seen: dict[tuple[int, int], int] = {}
        for angle in np.arange(heading_deg - half_fov, heading_deg + half_fov + 1e-9, step_deg):
            radians = math.radians(angle)
            ray = np.array([-math.sin(radians), math.cos(radians)])
            for distance in np.arange(10.0, VIEW_RANGE_CM, BELIEF_CELL_CM / 2):
                point = np.asarray(origin) + ray * distance
                if walls.size:
                    cell = self.grid.cell_of(point)
                    if self.grid._inside(cell) and walls[cell]:
                        break
                belief_cell = self._cell(point)
                if belief_cell is not None and belief_cell not in seen:
                    seen[belief_cell] = int(distance)
        return {(row, column, distance) for (row, column), distance in seen.items()}

    def looked(self, origin: Sequence[float], heading_deg: float, half_fov: float = CAMERA_HFOV_DEG / 2) -> None:
        """No cat in this view: lower the belief over what it could see."""
        for row, column, distance in self.visible(origin, heading_deg, half_fov):
            detect = 0.9 if distance < 150 else 0.75 if distance < 250 else 0.55
            self.belief[row, column] *= 1.0 - detect
        self.normalize()

    def drift(self) -> None:
        """Cats move: let the belief relax a little toward the prior."""
        self.belief = (1 - BELIEF_DRIFT) * self.belief + BELIEF_DRIFT * self.prior
        self.normalize()

    def normalize(self) -> None:
        total = self.belief.sum()
        if total > 0:
            self.belief /= total

    def boost(self, point: Sequence[float], radius: float = 80.0, weight: float = 4.0) -> None:
        """The cat was last seen here: it is probably still nearby."""
        near = np.hypot(self.centres_x - point[0], self.centres_y - point[1]) < radius
        self.belief[near & (self.prior > 0)] *= weight
        self.normalize()

    def mass(self, cells: set[tuple[int, int, int]]) -> float:
        return float(sum(self.belief[row, column] for row, column, _ in cells))

    def best_vantage(
        self, pose: Sequence[float], rng: random.Random, avoid: Sequence[tuple[float, float]] = ()
    ) -> tuple[tuple[float, float], float] | None:
        """The standpoint showing the most belief (all round) per cm of travel,
        away from any points in `avoid` (ones the car could not get to)."""
        if not self.grid.log_odds.size:
            return None
        distance, _, _ = self.grid.path_lengths(pose)
        traversable = self.grid.traversable()
        best = None
        step = max(1, int(round(VANTAGE_SPACING_CM / self.grid.cell)))
        offset = rng.randrange(step)
        for row in range(offset, traversable.shape[0], step):
            for column in range(offset, traversable.shape[1], step):
                if not traversable[row, column] or not math.isfinite(distance[row, column]):
                    continue
                point = self.grid.centre_of((row, column))
                if any(math.hypot(point[0] - x, point[1] - y) < UNREACHABLE_RADIUS_CM for x, y in avoid):
                    continue
                mass = self.mass(self.visible(point, 0.0, 180.0, step_deg=6.0))
                value = mass / (VANTAGE_OVERHEAD_CM + float(distance[row, column]))
                if best is None or value > best[1]:
                    best = (point, value, mass)
        if best is None:
            return None
        return best[0], best[2]


# -- the engine ----------------------------------------------------------------


@dataclass
class _Track:
    """Where the cat is, relative to the car."""

    bearing: float
    distance: float | None
    box_height: float
    seen_at: float
    misses: int = 0


class HuntEngine:
    def __init__(
        self,
        body: HuntBody,
        memory: HuntMemory,
        rng: random.Random | None = None,
        session_seconds: float = SESSION_SECONDS,
    ) -> None:
        self.body = body
        self.memory = memory
        self.rng = rng or random.Random()
        self.session_seconds = session_seconds
        self.started = body.now()
        self.phase = "starting"
        self.action = "getting ready"
        self.track: _Track | None = None
        self.mood = self.rng.uniform(0.4, 0.8)  # energy, drifting
        self.recent: deque[str] = deque(maxlen=NOVELTY_MEMORY)
        self.log: deque[dict[str, object]] = deque(maxlen=12)
        self.stats = {"looks": 0, "finds": 0, "moves": 0, "rests": 0, "play_seconds": 0.0, "moves_by_kind": {}}
        self.head = 90
        self.belief: CatBelief | None = None
        self.last_cat_position: tuple[float, float] | None = None
        self.end_reason: str | None = None
        self.unreachable: list[tuple[float, float, float]] = []  # x, y, until
        self.performed = ""  # the move that actually ran (moves fall back when there is no room)
        self.memory.hunts += 1
        self.memory.fade()

    # -- reporting ----------------------------------------------------------

    def _say(self, phase: str | None = None, action: str | None = None, *, log: bool = True) -> None:
        if phase:
            self.phase = phase
        if action:
            self.action = action
            if log:
                self.log.appendleft({"t": round(self.body.now() - self.started, 1), "phase": self.phase, "text": action})
        self.body.report(self.status())

    def status(self) -> dict[str, object]:
        track = self.track
        return {
            "phase": self.phase,
            "action": self.action,
            "mood": round(self.mood, 2),
            "elapsed_seconds": round(self.body.now() - self.started, 1),
            "cat": None
            if track is None
            else {
                "bearing_deg": round(track.bearing, 1),
                "distance_cm": None if track.distance is None else round(track.distance),
                "seen_seconds_ago": round(self.body.now() - track.seen_at, 1),
            },
            "last_cat_position": None if self.last_cat_position is None else [round(v, 1) for v in self.last_cat_position],
            "stats": {**self.stats, "play_seconds": round(float(self.stats["play_seconds"]), 1)},
            "log": list(self.log),
            "favourites": self.memory.favourites(),
            "end_reason": self.end_reason,
        }

    # -- the loop -----------------------------------------------------------

    def run(self) -> str:
        try:
            self._say("seeking", "starting the hunt: checking where I am")
            if not self._observe():
                return self._end("stopped")
            while True:
                reason = self._should_end()
                if reason:
                    return self._end(reason)
                if self.phase in ("seeking", "starting"):
                    self._seek()
                elif self.phase == "approaching":
                    self._approach()
                elif self.phase == "playing":
                    self._play_bout()
                elif self.phase == "resting":
                    self._rest()
                elif self.phase == "searching":
                    self._search_nearby()
        except _Stopped:
            return self._end("stopped")

    def _should_end(self) -> str | None:
        if self.body.stopped():
            return "stopped"
        if self.body.now() - self.started >= self.session_seconds:
            return "time_limit"
        if not self.body.battery_ok():
            return "battery_low"
        return None

    def _end(self, reason: str) -> str:
        self.end_reason = reason
        text = {
            "stopped": "hunt stopped",
            "time_limit": "playtime is over for now",
            "battery_low": "battery low: stopping the hunt",
        }.get(reason, reason)
        self._say("finished", text)
        return reason

    def _check(self, ok: bool) -> None:
        if not ok or self.body.stopped():
            raise _Stopped()

    # -- sensing ------------------------------------------------------------

    def _observe(self) -> bool:
        """A full stop: sweep, localize, update the map. The belief is built
        once per hunt, from the map as mapping left it, and kept: rebuilding
        it would forget where the car has already looked."""
        self.body.observe()
        if self.belief is None:
            self.belief = CatBelief(self.body.grid(), self.body.landmarks(), self.memory)
        return not self.body.stopped()

    def _look(self, pan: int, look: Look | None = None) -> Look:
        """Look toward `pan`: a picture taken now, or `look` from one taken
        a moment ago (the head is then left where it is)."""
        if look is None:
            self.head = pan
            look = self.body.look(pan)
        self.stats["looks"] += 1
        if look.cat is not None:
            self._see(look.cat)
        elif self.track is not None and abs(pan_bearing(pan) - self.track.bearing) <= CAMERA_HFOV_DEG / 2 - 5.0:
            # Only a look where the cat should have been counts as missing it.
            self.track.misses += 1
        return look

    def _see(self, cat: CatView) -> None:
        distance = self.memory.size_scale / max(cat.box_height, 1.0)
        self.track = _Track(cat.bearing_deg, distance, cat.box_height, self.body.now())
        pose = self.body.pose()
        heading = math.radians(pose[2] + cat.bearing_deg)
        self.last_cat_position = (pose[0] - math.sin(heading) * distance, pose[1] + math.cos(heading) * distance)

    def _range_to_cat(self) -> float | None:
        """With the cat straight ahead, the sonar measures it (and teaches the
        camera's distance scale)."""
        if self.track is None or abs(self.track.bearing) > 8.0:
            return None
        sonar = self.body.ping(90)
        if sonar is None:
            return None
        estimate = self.track.distance
        if estimate is None or abs(sonar - estimate) <= CALIBRATE_AGREEMENT * estimate:
            if sonar <= CALIBRATE_WITHIN_CM:
                self.memory.calibrate(self.track.box_height, sonar)
            self.track.distance = sonar
            return sonar
        return None

    def _lost(self) -> bool:
        return self.track is None or self.track.misses >= LOST_AFTER_LOOKS

    # -- seeking ------------------------------------------------------------

    def _seek(self) -> None:
        self._say("seeking", "looking around for the cat")
        if self._look_around():
            return
        assert self.belief is not None
        self.belief.drift()
        now = self.body.now()
        self.unreachable = [spot for spot in self.unreachable if spot[2] > now]
        vantage = self.belief.best_vantage(self.body.pose(), self.rng, [(x, y) for x, y, _ in self.unreachable])
        if vantage is None:
            self._say(action="nowhere new to look from here; turning round", log=True)
            self._check(self.body.turn(self.rng.choice((-1, 1)) * 120.0, precise=True))
            return
        (x, y), mass = vantage
        pose = self.body.pose()
        if math.hypot(x - pose[0], y - pose[1]) < 35.0:
            # The best place to look from is right here: face the most
            # promising direction not yet seen.
            sectors = {
                turn: self.belief.mass(self.belief.visible(pose[:2], pose[2] + turn, 45.0, step_deg=5.0))
                for turn in (90.0, 180.0, -90.0)
            }
            turn = max(sectors, key=sectors.get)
            self._say(action=f"nothing ahead; turning {turn:+.0f} degrees to look that way")
            self._check(self.body.turn(turn, precise=True))
            return
        self._say(action=f"heading to look from ({x:.0f}, {y:.0f}) cm: {100 * mass:.0f}% chance of a view of the cat")
        start = self.body.pose()
        if self._travel((x, y)):
            return  # the cat turned up on the way
        end = self.body.pose()
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 10.0:
            # Got nowhere: try somewhere else for a while, and get clear first.
            self.unreachable.append((x, y, self.body.now() + UNREACHABLE_SECONDS))
            self._unstick()
        self._check(self._observe())

    def _unstick(self) -> None:
        """Hemmed in: glance about with the sonar, turn to the clearest way
        (clear on the map too) and move out a little."""
        best = None
        pose = self.body.pose()
        for pan in (30, 60, 90, 120, 150):
            reading = self.body.ping(pan)
            if reading is None:
                continue
            turn = pan_bearing(pan)
            mapped = self.body.grid().clear_run((pose[0], pose[1], pose[2] + turn), SEEK_LEG_CM)
            room = min(reading - SEEK_STANDOFF_CM, mapped)
            if best is None or room > best[1]:
                best = (turn, room)
        self.head = 90
        if best is None or best[1] < 8.0:
            self._say(action="hemmed in here; turning round to find a way out")
            self._check(self.body.turn(180.0, precise=True))
            return
        turn, room = best
        self._say(action=f"couldn't get there; edging out the clearest way ({turn:+.0f} degrees)")
        if abs(turn) > 5.0:
            self._check(self.body.turn(turn, precise=True))
        self._safe_forward(min(25.0, room))

    def _look_around(self) -> bool:
        """Pan the camera across the room; True once the cat is found."""
        assert self.belief is not None
        pose = self.body.pose()
        pans = LOOK_PANS if self.rng.random() < 0.5 else tuple(reversed(LOOK_PANS))
        # Right after a stop, its sweep has already pictured these directions.
        swept = self.body.sweep_looks(pans)
        for pan in sorted(pans, key=lambda pan: pan not in swept):
            self._check(not self.body.stopped())
            look = self._look(pan, swept.get(pan))
            if look.cat is not None:
                return self._found()
            self.belief.looked(pose[:2], pose[2] + pan_bearing(pan))
        self.head = 90
        return False

    def _found(self) -> bool:
        self.stats["finds"] += 1
        self.memory.finds += 1
        if self.last_cat_position is not None:
            self.memory.remember_cat(*self.last_cat_position)
        distance = "" if self.track is None or self.track.distance is None else f", about {self.track.distance:.0f} cm away"
        self._say("approaching", f"found the cat{distance}!")
        return True

    def _travel(self, goal: tuple[float, float]) -> bool:
        """Drive toward a vantage point in legs; True if the cat shows up."""
        for _ in range(8):
            self._check(not self.body.stopped())
            pose = self.body.pose()
            if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < 25.0:
                return False
            leg = self.body.grid().route(pose, goal)
            if leg is None:
                return False
            bearing, distance = leg
            if abs(bearing) > 8.0:
                self._check(self.body.turn(bearing, precise=True))
            ahead = self.body.ping(90)
            room = min(SEEK_LEG_CM, max(distance, MIN_LEG_CM), self.body.grid().clear_run(self.body.pose(), SEEK_LEG_CM))
            if ahead is not None:
                room = min(room, ahead - SEEK_STANDOFF_CM)
            if room < 8.0:
                return False
            self._check(self.body.move(room, 0.0))
            if self._look(90).cat is not None:
                return self._found()
        return False

    # -- approaching --------------------------------------------------------

    def _face_cat(self, precise: bool = False) -> None:
        """Turn the body toward the cat and centre the head."""
        if self.track is None:
            return
        bearing = self.track.bearing
        if abs(bearing) > 6.0:
            self._check(self.body.turn(bearing, precise=precise))
            self.track.bearing = 0.0
        self._look(90)

    def _approach(self) -> None:
        self._face_cat(precise=False)
        for _ in range(6):
            if self._lost():
                self._say("searching", "lost sight of the cat; looking nearby")
                return
            self._range_to_cat()
            distance = self.track.distance if self.track else None
            if distance is None:
                self._say(action="can't tell how far the cat is; creeping closer")
                self._safe_forward(10.0)
            elif distance <= PLAY_FAR_CM:
                self._say("playing", f"in play range ({distance:.0f} cm): let's play!")
                self._bout_ends = self.body.now() + self.rng.uniform(*BOUT_SECONDS)
                return
            else:
                self._say(action=f"sneaking up on the cat ({distance:.0f} cm away)")
                self._safe_forward(min(SEEK_LEG_CM, distance - PLAY_TARGET_CM))
            look = self._look(90)
            if look.cat is not None and abs(look.cat.bearing_deg) > 10.0:
                self._face_cat()
        self._say("playing", "close enough: let's play!")
        self._bout_ends = self.body.now() + self.rng.uniform(*BOUT_SECONDS)

    # -- safety -------------------------------------------------------------

    def _forward_room(self) -> float:
        """How far forward is safe: the sonar ahead, the map, and the cat."""
        ahead = self.body.ping(90)
        room = self.body.grid().clear_run(self.body.pose(), SEEK_LEG_CM)
        if ahead is not None:
            room = min(room, ahead - MIN_CAT_DISTANCE_CM)
        if self.track is not None and self.track.distance is not None and abs(self.track.bearing) < 40.0:
            room = min(room, self.track.distance - MIN_CAT_DISTANCE_CM)
        return max(0.0, room)

    def _side_room(self, side: int) -> float:
        """Sideways clearance (side +1 right, -1 left): a glance that way (the
        head reaches 60 degrees off ahead, not fully sideways, so it can miss
        a wall right beside the car) and floor the map knows is clear."""
        reading = self.body.ping(150 if side > 0 else 30)
        self.head = 150 if side > 0 else 30
        if reading is None:
            return 0.0
        pose = self.body.pose()
        mapped = self.body.grid().clear_run((pose[0], pose[1], pose[2] - 90.0 * side), 40.0, known_only=True)
        return max(0.0, min(reading - SIDE_MARGIN_CM, mapped - 5.0, self._cat_limit(-90.0 * side)))

    def _back_room(self) -> float:
        """Reversing: the rear is not sensed, so only space the map shows clear."""
        pose = self.body.pose()
        behind = (pose[0], pose[1], pose[2] + 180.0)
        mapped = self.body.grid().clear_run(behind, BACK_LIMIT_CM + 10.0, known_only=True) - 10.0
        return max(0.0, min(BACK_LIMIT_CM, mapped, self._cat_limit(180.0)))

    def _cat_clear_to_move(self) -> bool:
        """Whether driving is safe with respect to the cat: either no cat is
        known to be near (seeking), or it was seen just now, ahead and not too
        close. A stale sighting gets one fresh look first; a cat that may be
        beside or behind the car, where nothing can see it, stops the move."""
        if self.track is None:
            return True
        age = self.body.now() - self.track.seen_at
        if self.track.distance is not None and self.track.distance > CAT_FAR_CM and age <= CAT_FAR_FRESH_SECONDS:
            return True
        if age > FRESH_SECONDS:
            self._look(90)
        track = self.track
        return (
            track is not None
            and self.body.now() - track.seen_at <= FRESH_SECONDS
            and abs(track.bearing) <= CAT_AHEAD_DEG
            and (track.distance is None or track.distance >= MIN_CAT_DISTANCE_CM)
        )

    def _safe_forward(self, wanted: float, speed: float = 1.0) -> float:
        if not self._cat_clear_to_move():
            return 0.0
        distance = min(wanted, self._forward_room())
        if distance < 3.0:
            return 0.0
        self._check(self.body.move(distance, 0.0, speed))
        return distance

    def _safe_back(self, wanted: float, came_forward: float = 0.0) -> float:
        if not self._cat_clear_to_move():
            return 0.0
        # Straight back along a path just driven forward is known to be clear.
        distance = min(wanted, max(self._back_room(), came_forward))
        if distance < 3.0:
            return 0.0
        self._check(self.body.move(-distance, 0.0))
        return distance

    def _slide(self, forward: float, sideways: float) -> bool:
        """A sideways or diagonal move inside a play move, checked against
        the cat first. Room was measured by the move itself."""
        if not self._cat_clear_to_move():
            return False
        self._check(self.body.move(forward, sideways))
        return True

    # -- playing ------------------------------------------------------------

    def _play_bout(self) -> None:
        bout_started = self.body.now()
        while self.body.now() < getattr(self, "_bout_ends", bout_started):
            reason = self._should_end()
            if reason:
                return
            if self._lost():
                self._say("searching", "the cat slipped out of view; looking nearby")
                break
            self._play_beat()
        else:
            self.stats["rests"] += 1
            self._say("resting", "taking a breather: backing off to wait")
        self.stats["play_seconds"] = float(self.stats["play_seconds"]) + self.body.now() - bout_started

    def _play_beat(self) -> None:
        look = self._look(self.head)
        if look.cat is None:
            look = self._look(90)
        if look.cat is None:
            return
        # Keep the cat in view: the head follows, the body turns when the head is far off.
        if abs(look.cat.bearing_deg) > HEAD_TRACK_LIMIT_DEG:
            self._face_cat()
        self._range_to_cat()
        self.mood = min(1.0, max(0.0, self.mood + self.rng.gauss(0.0, 0.08)))
        distance = self.track.distance if self.track else None
        if distance is not None and distance > PLAY_FAR_CM + 15.0 and not look.person_ahead:
            if self._close_in(distance):
                return
        if distance is not None and distance < PLAY_NEAR_CM - 10.0:
            backed = self._safe_back(min(BACK_LIMIT_CM, PLAY_NEAR_CM - distance))
            if backed > 0:
                self._say("playing", f"the cat is very close ({distance:.0f} cm): easing back {backed:.0f} cm")
                return
        move = self._choose(person_ahead=look.person_ahead)
        before = self._cat_world()
        self.performed = move
        description = MOVES[move](self)
        # A move with no room falls back to another: credit the one that ran.
        move = self.performed
        self.stats["moves"] += 1
        kinds = self.stats["moves_by_kind"]
        kinds[move] = kinds.get(move, 0) + 1
        self.recent.append(move)
        self._say("playing", description)
        reaction = self._reaction(before)
        if reaction is not None:
            self.memory.reward(move, reaction)

    def _close_in(self, distance: float) -> bool:
        """The cat drifted off: close back in, as a stalk or a dash, or round
        whatever is in the way. False if there is no way to get closer."""
        if self.track is not None and abs(self.track.bearing) > 8.0:
            self._face_cat()
        gap = distance - PLAY_TARGET_CM
        stealthy = self.mood < 0.4
        if self._forward_room() >= min(gap, 15.0):
            went = self._safe_forward(min(SEEK_LEG_CM, gap), speed=0.6 if stealthy else 1.0)
            if went > 0:
                self._say(
                    "playing",
                    f"creeping after the cat ({distance:.0f} cm away, {went:.0f} cm)"
                    if stealthy
                    else f"dashing after the cat ({distance:.0f} cm away, {went:.0f} cm)",
                )
                return True
        if self.last_cat_position is None:
            return False
        # Something is in the way: go round it with the map, to a spot short of the cat.
        pose = self.body.pose()
        cat = np.array(self.last_cat_position)
        offset = cat - np.array(pose[:2])
        length = float(np.linalg.norm(offset))
        if length < 1.0:
            return False
        spot = cat - offset / length * PLAY_TARGET_CM
        if not self._drive_route((float(spot[0]), float(spot[1])), legs=3):
            return False
        self._look(90)
        if self.track is not None and abs(self.track.bearing) > 8.0:
            self._face_cat()
        self._say("playing", "going round the obstacle to get back to the cat")
        return True

    def _drive_route(self, goal: tuple[float, float], legs: int) -> bool:
        """Follow the map's route toward a point for a few legs; True if it moved."""
        moved = False
        for _ in range(legs):
            pose = self.body.pose()
            if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < 20.0:
                break
            leg = self.body.grid().route(pose, goal)
            if leg is None:
                break
            bearing, length = leg
            if abs(bearing) > 8.0:
                self._check(self.body.turn(bearing, precise=False))
            # A short first step (out of a wall's margin) still shows the way.
            went = self._safe_forward(min(SEEK_LEG_CM, max(length, MIN_LEG_CM)))
            if went <= 0:
                break
            moved = True
        return moved

    def _cat_limit(self, heading_change: float) -> float:
        """How far the car may move in a direction (relative to its heading)
        before closing to within MIN_CAT_DISTANCE_CM of the cat."""
        track = self.track
        if track is None or track.distance is None or self.body.now() - track.seen_at > 5.0:
            return math.inf
        off = abs((track.bearing - heading_change + 180.0) % 360.0 - 180.0)
        if off >= 90.0:
            return math.inf
        # Moving at an angle closes the gap by cos(off) per cm.
        return max(0.0, (track.distance - MIN_CAT_DISTANCE_CM) / max(math.cos(math.radians(off)), 0.2))

    def _choose(self, person_ahead: bool = False) -> str:
        """Pick the next move: what suits the distance and mood, what the cat
        has liked, what has not been done lately, and a dash of chance."""
        distance = self.track.distance if self.track and self.track.distance is not None else PLAY_TARGET_CM
        scores = {}
        for name, spec in MOVE_SPECS.items():
            if person_ahead and spec["toward"]:
                continue
            fit = spec["fit"](distance) + spec["energy"] * (self.mood - 0.5)
            mean, count = self.memory.score(name)
            learned = mean + 0.35 / math.sqrt(count + 1)  # optimism for little-tried moves
            repeats = sum(0.6 ** age for age, recent in enumerate(reversed(self.recent)) if recent == name)
            scores[name] = fit + learned - 0.8 * repeats
        temperature = 0.25
        top = max(scores.values())
        weights = {name: math.exp((score - top) / temperature) for name, score in scores.items()}
        pick = self.rng.random() * sum(weights.values())
        for name, weight in weights.items():
            pick -= weight
            if pick <= 0:
                return name
        return next(iter(weights))

    def _cat_world(self) -> np.ndarray | None:
        """The cat's position in the odometry frame (cm), to see whether it moved."""
        if self.track is None or self.track.distance is None or self.body.now() - self.track.seen_at > 2.0:
            return None
        odometry = self.body.odometry()
        heading = math.radians(odometry[2] + self.track.bearing)
        return np.array([odometry[0] - math.sin(heading) * self.track.distance, odometry[1] + math.cos(heading) * self.track.distance])

    def _reaction(self, before: np.ndarray | None) -> float | None:
        """How much the cat itself moved after a move (the car's own motion
        taken out): 0 none, 1 a lot."""
        if before is None:
            return None
        self.body.wait(self.rng.uniform(0.3, 0.7))
        look = self._look(self.head)
        if look.cat is None and self.head != 90:
            look = self._look(90)
        if look.cat is None:
            # It left: maybe a big reaction, maybe boredom. A weak signal.
            return 0.3
        after = self._cat_world()
        if after is None:
            return None
        return float(min(1.0, np.linalg.norm(after - before) / 40.0))

    # -- resting ------------------------------------------------------------

    def _rest(self) -> None:
        backed = self._safe_back(self.rng.uniform(15.0, BACK_LIMIT_CM))
        self._look(90)
        if self.track is not None and abs(self.track.bearing) > 10.0:
            self._face_cat()
        until = self.body.now() + self.rng.uniform(*REST_SECONDS)
        self._say("resting", f"backed off {backed:.0f} cm; waiting, watching the cat")
        while self.body.now() < until:
            pause = min(until - self.body.now(), self.rng.uniform(*REST_FIDGET_SECONDS))
            self._check(self.body.wait(pause))
            if self.body.now() >= until or self._should_end():
                break
            fidget = self.rng.choice(("wiggle", "head", "head", "inch"))
            if fidget == "wiggle":
                swing = self.rng.uniform(6.0, 12.0)
                self._check(self.body.turn(swing, precise=False))
                self._check(self.body.turn(-swing, precise=False))
                self._say(action="a tiny wiggle (still resting)")
            elif fidget == "inch":
                self._safe_forward(self.rng.uniform(3.0, 6.0), speed=0.6)
                self._say(action="inching forward a little (still resting)")
            else:
                pan = 90 + self.rng.choice((-1, 1)) * self.rng.randint(10, 25)
                self._look(pan)
                self._look(90)
                self._say(action="glancing around (still resting)")
            if self.track is not None and self.track.distance is not None and self.track.distance < MIN_CAT_DISTANCE_CM + 10:
                self._safe_back(10.0)
                self._say(action="the cat came close: easing back")
        self._look(90)
        if self._lost():
            self._say("searching", "rest over, but where did the cat go?")
            return
        self._bout_ends = self.body.now() + self.rng.uniform(*BOUT_SECONDS)
        self._say("playing", "rest over: back to bothering the cat!")

    # -- searching nearby ---------------------------------------------------

    def _search_nearby(self) -> None:
        """Look around where the cat was last seen before a full hunt."""
        for pan in (90, 60, 120, 30, 150):
            if self._look(pan).cat is not None:
                self._say("approaching", "there it is again!")
                return
        for _ in range(2):
            self._check(self.body.turn(self.rng.choice((-1, 1)) * 90.0, precise=False))
            for pan in (60, 90, 120):
                if self._look(pan).cat is not None:
                    self._say("approaching", "spotted the cat again!")
                    return
        self.track = None
        self._say("seeking", "the cat moved on: hunting for it again")
        self._check(self._observe())
        if self.belief is not None and self.last_cat_position is not None:
            self.belief.boost(self.last_cat_position)


class _Stopped(Exception):
    pass


# -- moves ---------------------------------------------------------------------
# Each takes the engine, makes one lively, safe move with fresh random
# parameters, and returns what it did in words.


def _side(engine: HuntEngine) -> int:
    return engine.rng.choice((-1, 1))


def _side_name(side: int) -> str:
    return "right" if side > 0 else "left"


def move_pounce(engine: HuntEngine) -> str:
    engine.performed = "pounce"
    wanted = engine.rng.uniform(10.0, 30.0) * (0.6 + 0.6 * engine.mood)
    went = engine._safe_forward(wanted)
    if went <= 0:
        return move_wiggle(engine)
    engine.body.wait(engine.rng.uniform(0.15, 0.6))
    engine._safe_back(went * engine.rng.uniform(0.6, 1.0), came_forward=went)
    return f"fake pounce! ({went:.0f} cm in and back)"


def move_charge_retreat(engine: HuntEngine) -> str:
    engine.performed = "charge_retreat"
    went = engine._safe_forward(engine.rng.uniform(20.0, 40.0))
    engine.body.wait(engine.rng.uniform(0.2, 0.5))
    back = engine._safe_back(went + engine.rng.uniform(0.0, 10.0), came_forward=went)
    return f"charge and retreat ({went:.0f} cm in, {back:.0f} cm out)"


def move_zigzag(engine: HuntEngine) -> str:
    engine.performed = "zigzag"
    toward = engine.rng.random() < 0.6
    zigs = engine.rng.randint(2, 4)
    width = engine.rng.uniform(6.0, 14.0)
    room_left, room_right = engine._side_room(-1), engine._side_room(1)
    width = min(width, room_left, room_right)
    engine.head = 90
    if width < 4.0:
        return move_wiggle(engine)
    step = engine.rng.uniform(5.0, 9.0)
    if toward:
        step = min(step, engine._forward_room() / zigs)
    else:
        step = min(step, engine._back_room() / zigs)
    side = _side(engine)
    offset = 0.0  # sideways from the line it started on
    done = 0
    for index in range(zigs):
        target = side * width * 0.5 * (-1) ** index
        if not engine._slide(step if toward else -step, target - offset):
            break
        offset = target
        done += 1
    # Back onto the line it started on.
    if abs(offset) > 0.5:
        engine._slide(0.0, -offset)
    if done == 0:
        return move_wiggle(engine)
    zigs = done
    return f"zig-zagging {'toward' if toward else 'away from'} the cat ({zigs} zigs)"


def move_sidestep(engine: HuntEngine) -> str:
    engine.performed = "sidestep"
    side = _side(engine)
    room = engine._side_room(side)
    engine.head = 90
    distance = min(engine.rng.uniform(10.0, 25.0), room)
    if distance < 5.0:
        side = -side
        distance = min(engine.rng.uniform(10.0, 25.0), engine._side_room(side))
        engine.head = 90
        if distance < 5.0:
            return move_wiggle(engine)
    if not engine._slide(0.0, side * distance):
        return move_wiggle(engine)
    engine.body.wait(engine.rng.uniform(0.2, 0.8))
    if engine.rng.random() < 0.6 and engine._slide(0.0, -side * distance):
        return f"sidestepping {_side_name(side)} and back ({distance:.0f} cm)"
    return f"sidestepping {_side_name(side)} ({distance:.0f} cm)"


def move_shimmy(engine: HuntEngine) -> str:
    engine.performed = "shimmy"
    room = min(engine._side_room(-1), engine._side_room(1))
    engine.head = 90
    amount = min(engine.rng.uniform(3.0, 6.0), room)
    if amount < 2.0:
        return move_wiggle(engine)
    count = engine.rng.randint(3, 6)
    side = _side(engine)
    offset = 0.0
    shakes = 0
    for index in range(count):
        step = side * amount * (1 if index % 2 == 0 else -1)
        if not engine._slide(0.0, step):
            break
        offset += step
        shakes += 1
    if abs(offset) > 0.5:
        engine._slide(0.0, -offset)
    if shakes == 0:
        return move_wiggle(engine)
    return f"shimmying side to side ({shakes} shakes)"


def move_wiggle(engine: HuntEngine) -> str:
    engine.performed = "wiggle"
    swings = engine.rng.randint(2, 5)
    angle = engine.rng.uniform(8.0, 20.0) * (0.7 + 0.6 * engine.mood)
    side = _side(engine)
    for index in range(swings):
        engine._check(engine.body.turn(side * angle * (1 if index == 0 else 2) * (-1) ** index, precise=False))
    engine._check(engine.body.turn(side * angle * (1 if swings % 2 else -1) * -1, precise=False))
    return f"wiggling ({swings} swings)"


def move_spin(engine: HuntEngine) -> str:
    engine.performed = "spin"
    side = _side(engine)
    for _ in range(4):
        engine._check(engine.body.turn(side * 90.0, precise=False))
    engine._look(90)
    if engine.track is not None and abs(engine.track.bearing) > 10.0:
        engine._face_cat()
    return f"spinning a full circle {'left' if side > 0 else 'right'}!"


def move_orbit(engine: HuntEngine) -> str:
    """Slide sideways while turning to keep facing the cat: an arc around it."""
    engine.performed = "orbit"
    distance = engine.track.distance if engine.track and engine.track.distance else PLAY_TARGET_CM
    side = _side(engine)
    room = engine._side_room(side)
    engine.head = 90
    seconds = min(engine.rng.uniform(0.8, 1.8), room * MOVE_SECONDS_PER_CM * 1.5)
    if seconds < 0.4 or distance < PLAY_NEAR_CM or not engine._cat_clear_to_move():
        return move_sidestep(engine)
    strafe = 0.7 * side
    # Sliding right leaves the cat to the left: turn counterclockwise at the
    # rate that keeps it ahead, sideways speed / distance (about 60 cm/s at
    # full strafe; full steering turns about 384 degrees a second).
    rate = 0.7 * 60.0 / distance * (180 / math.pi)
    steering = side * min(0.5, rate / 384.0)
    engine._check(engine.body.arc(strafe, steering, seconds))
    engine._look(90)
    if engine.track is not None and abs(engine.track.bearing) > 10.0:
        engine._face_cat()
    return f"circling {_side_name(side)} around the cat"


def move_stalk(engine: HuntEngine) -> str:
    engine.performed = "stalk"
    steps = engine.rng.randint(2, 4)
    moved = 0.0
    for _ in range(steps):
        moved += engine._safe_forward(engine.rng.uniform(3.0, 6.0), speed=0.6)
        engine.body.wait(engine.rng.uniform(0.4, 1.0))
    if moved <= 0:
        return move_freeze(engine)
    return f"stalking slowly ({moved:.0f} cm, freezing between creeps)"


def move_freeze(engine: HuntEngine) -> str:
    engine.performed = "freeze"
    seconds = engine.rng.uniform(1.0, 3.5)
    engine.body.wait(seconds)
    return f"freezing... staring at the cat ({seconds:.1f} s)"


def move_peekaboo(engine: HuntEngine) -> str:
    engine.performed = "peekaboo"
    side = _side(engine)
    angle = engine.rng.uniform(50.0, 90.0)
    engine._check(engine.body.turn(side * angle, precise=False))
    engine.body.wait(engine.rng.uniform(0.6, 1.8))
    engine._check(engine.body.turn(-side * angle, precise=False))
    return "peekaboo: looked away, then whipped back round"


def move_head_bob(engine: HuntEngine) -> str:
    engine.performed = "head_bob"
    bobs = engine.rng.randint(2, 4)
    for _ in range(bobs):
        offset = engine.rng.randint(10, 25) * _side(engine)
        engine._look(90 + offset)
    engine._look(90)
    return f"bobbing the head about ({bobs} looks)"


def move_back_off_tease(engine: HuntEngine) -> str:
    engine.performed = "back_off_tease"
    back = engine._safe_back(engine.rng.uniform(10.0, 20.0))
    engine.body.wait(engine.rng.uniform(0.5, 1.5))
    went = engine._safe_forward(back * engine.rng.uniform(0.5, 0.9) + 2.0)
    return f"backing off {back:.0f} cm... then sneaking {went:.0f} cm back in"


def move_combo(engine: HuntEngine) -> str:
    """Two quick moves run together into something new."""
    names = [name for name in ("wiggle", "shimmy", "pounce", "sidestep", "head_bob", "peekaboo") if name in MOVES]
    first, second = engine.rng.sample(names, 2)
    one = MOVES[first](engine)
    two = MOVES[second](engine)
    engine.performed = "combo"
    return f"combo: {one}, then {two}"


MOVES: dict[str, Callable[[HuntEngine], str]] = {
    "pounce": move_pounce,
    "charge_retreat": move_charge_retreat,
    "zigzag": move_zigzag,
    "sidestep": move_sidestep,
    "shimmy": move_shimmy,
    "wiggle": move_wiggle,
    "spin": move_spin,
    "orbit": move_orbit,
    "stalk": move_stalk,
    "freeze": move_freeze,
    "peekaboo": move_peekaboo,
    "head_bob": move_head_bob,
    "back_off_tease": move_back_off_tease,
    "combo": move_combo,
}


def _band(near: float, far: float) -> Callable[[float], float]:
    """Suits distances between near and far, falling off outside."""

    def fit(distance: float) -> float:
        if near <= distance <= far:
            return 0.5
        gap = near - distance if distance < near else distance - far
        return 0.5 - gap / 40.0

    return fit


# How each move suits the distance, the mood (energy), and whether it drives at the cat.
MOVE_SPECS: dict[str, dict] = {
    "pounce": {"fit": _band(55.0, 110.0), "energy": 0.6, "toward": True},
    "charge_retreat": {"fit": _band(70.0, 130.0), "energy": 0.7, "toward": True},
    "zigzag": {"fit": _band(50.0, 120.0), "energy": 0.5, "toward": True},
    "sidestep": {"fit": _band(35.0, 110.0), "energy": 0.2, "toward": False},
    "shimmy": {"fit": _band(35.0, 100.0), "energy": 0.3, "toward": False},
    "wiggle": {"fit": _band(30.0, 130.0), "energy": 0.1, "toward": False},
    "spin": {"fit": _band(45.0, 130.0), "energy": 0.6, "toward": False},
    "orbit": {"fit": _band(55.0, 110.0), "energy": 0.3, "toward": False},
    "stalk": {"fit": _band(60.0, 140.0), "energy": -0.4, "toward": True},
    "freeze": {"fit": _band(30.0, 140.0), "energy": -0.6, "toward": False},
    "peekaboo": {"fit": _band(40.0, 120.0), "energy": 0.0, "toward": False},
    "head_bob": {"fit": _band(30.0, 140.0), "energy": -0.3, "toward": False},
    "back_off_tease": {"fit": _band(30.0, 80.0), "energy": 0.2, "toward": False},
    "combo": {"fit": _band(45.0, 110.0), "energy": 0.4, "toward": True},
}
