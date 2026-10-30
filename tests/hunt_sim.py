"""A simulated house, car and cat for the hunt.

The floor plan is explore_sim's two rooms and corridor. The car's motion is
exact (odometry is the truth), so what is tested is the hunting and playing.
The camera sees the cat only inside its field of view, within 4 m and with
nothing in the way; the sonar echoes off it like a post. The cat wanders now
and then and reacts to the car's moves (to all of them, or only to the kinds
it likes), and never sits closer than 25 cm to a wall.
"""

import math
import random

import numpy as np

from explore_sim import FloorPlan, NO_ECHO_CM, ROOM_A, CORRIDOR, ROOM_B
from robocar.exploration import CoverageGrid
from robocar.hunt import CAMERA_HFOV_DEG, DEFAULT_SIZE_SCALE, PERSON_AHEAD_DEG, CatView, Look, pan_bearing
from robocar.surfaces import SONAR_BEAM_WIDTH_DEGREES, direction, sonar_origin

CAT_RADIUS_CM = 10.0
TOUCH_CM = 20.0
CAR_RADIUS_CM = 15.0


def build_map(floor: FloorPlan) -> CoverageGrid:
    """A coverage grid as a mapping run would leave it: sweeps all over the house."""
    grid = CoverageGrid()
    for box in (ROOM_A, CORRIDOR, ROOM_B):
        for x in np.arange(box[0] + 45, box[2] - 30, 70):
            for y in np.arange(box[1] + 45, box[3] - 30, 70):
                for heading in (0.0, 90.0, 180.0, 270.0):
                    pose = (float(x), float(y), heading)
                    grid.integrate(floor.scan(pose), pose, SONAR_BEAM_WIDTH_DEGREES)
    return grid


class SimCat:
    def __init__(self, floor: FloorPlan, position, rng: random.Random, likes=None, react=0.5, wander=0.05):
        self.floor = floor
        self.x, self.y = position
        self.rng = rng
        self.likes = likes  # None: reacts to anything; else a set of motion kinds
        self.react = react
        self.wander = wander
        self.moves = 0

    car = (0.0, 0.0)  # kept up to date by the body

    def hop(self, scale: float) -> None:
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            distance = self.rng.uniform(0.4, 1.0) * scale
            x, y = self.x + math.cos(angle) * distance, self.y + math.sin(angle) * distance
            # A startled cat jumps about, but not onto the car.
            if (
                self.floor.clearance((x, y)) >= 25.0
                and self._inside((x, y))
                and math.hypot(x - self.car[0], y - self.car[1]) >= 35.0
            ):
                self.x, self.y = x, y
                self.moves += 1
                return

    def _inside(self, point) -> bool:
        return any(box[0] < point[0] < box[2] and box[1] < point[1] < box[3] for box in (ROOM_A, CORRIDOR, ROOM_B))

    def reacts_to(self, kind: str, car_distance: float) -> None:
        if car_distance > 160:
            return
        if self.likes is not None and kind not in self.likes:
            return
        if self.rng.random() < self.react:
            self.hop(35.0)

    def idle(self, seconds: float) -> None:
        if self.rng.random() < 1 - (1 - self.wander) ** seconds:
            self.hop(25.0)


class SimHuntBody:
    def __init__(self, floor: FloorPlan, grid: CoverageGrid, pose, cat: SimCat, rng: random.Random, landmarks=()):
        self.floor = floor
        self._grid = grid
        self.truth = tuple(float(v) for v in pose)
        self.cat = cat
        self.rng = rng
        self._landmarks = list(landmarks)
        self.t = 0.0
        self.pan = 90
        self.bumps = 0
        self.touches = 0
        self.closest_cat = math.inf
        self.statuses: list[dict] = []
        self.person = None  # (x, y) of a person standing in the room, if any
        self.moves_toward_person = 0
        self.observations = 0
        self.looks = 0
        self.stop_at = math.inf
        self._swept_at = None
        self.sweep_looks_used = 0

    # -- time -----------------------------------------------------------------
    def now(self) -> float:
        return self.t

    def _spend(self, seconds: float) -> None:
        self.t += seconds
        self.cat.car = self.truth[:2]
        self.cat.idle(seconds)

    def wait(self, seconds: float) -> bool:
        self._spend(max(0.0, seconds))
        return not self.stopped()

    def stopped(self) -> bool:
        return self.t >= self.stop_at

    def battery_ok(self) -> bool:
        return True

    # -- sensing --------------------------------------------------------------
    def _cat_distance(self) -> float:
        return math.hypot(self.cat.x - self.truth[0], self.cat.y - self.truth[1])

    def _blocked(self, start, end) -> bool:
        a, b = np.array(start, float), np.array(end, float)
        length = float(np.linalg.norm(b - a))
        for fraction in np.linspace(0.05, 0.95, max(2, int(length / 5))):
            if self.floor.clearance(a + (b - a) * fraction) < 1.0:
                return True
        return False

    def look(self, pan: int) -> Look:
        self.pan = pan
        self.looks += 1
        self._spend(0.25)
        return self._view(pan)

    def _view(self, pan: int) -> Look:
        camera = sonar_origin(self.truth)
        heading = self.truth[2] + pan_bearing(pan)
        offset = np.array([self.cat.x, self.cat.y]) - camera
        distance = float(np.linalg.norm(offset))
        bearing = math.degrees(math.atan2(-offset[0], offset[1]))
        relative = (bearing - heading + 180.0) % 360.0 - 180.0
        cat = None
        if abs(relative) <= CAMERA_HFOV_DEG / 2 and distance <= 400 and not self._blocked(camera, (self.cat.x, self.cat.y)):
            box_height = 1.1 * DEFAULT_SIZE_SCALE / max(distance, 1.0)
            car_bearing = pan_bearing(pan) + relative + self.rng.gauss(0, 1.0)
            cat = CatView(car_bearing, box_height, 0.85, (300, 200, int(box_height * 1.3), int(box_height)), 640)
        person_ahead = False
        if self.person is not None:
            offset = np.array(self.person) - camera
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            person_ahead = abs((bearing - self.truth[2] + 180) % 360 - 180) <= PERSON_AHEAD_DEG and np.linalg.norm(offset) < 150
        return Look(cat, person_ahead)

    def ping(self, pan: int) -> float | None:
        self.pan = pan
        self._spend(0.1)
        angle = pan_bearing(pan)
        wall = self.floor.sonar.distance(self.truth, angle)
        best = NO_ECHO_CM if wall is None else wall
        origin = sonar_origin(self.truth)
        offset = np.array([self.cat.x, self.cat.y]) - origin
        bearing = math.degrees(math.atan2(-offset[0], offset[1]))
        if abs((bearing - self.truth[2] - angle + 180) % 360 - 180) <= SONAR_BEAM_WIDTH_DEGREES / 2:
            best = min(best, float(np.linalg.norm(offset)) - CAT_RADIUS_CM)
        if self.person is not None:
            offset = np.array(self.person) - origin
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            if abs((bearing - self.truth[2] - angle + 180) % 360 - 180) <= SONAR_BEAM_WIDTH_DEGREES / 2:
                best = min(best, float(np.linalg.norm(offset)) - 15.0)
        return best

    def observe(self) -> None:
        self.observations += 1
        self._spend(10.0)
        self._grid.integrate(self.floor.scan(self.truth), self.truth, SONAR_BEAM_WIDTH_DEGREES)
        self._swept_at = (self.truth, self.t)

    def sweep_looks(self, pans) -> dict:
        """What the sweep's pictures showed, while the car stands where it took them."""
        swept, self._swept_at = self._swept_at, None
        if swept is None or swept[0] != self.truth:
            return {}
        self.sweep_looks_used += 1
        return {pan: self._view(pan) for pan in pans}

    def pose(self):
        return self.truth

    def odometry(self):
        return self.truth

    def grid(self) -> CoverageGrid:
        return self._grid

    def landmarks(self):
        return self._landmarks

    # -- moving ---------------------------------------------------------------
    def _collide(self, point) -> str | None:
        if self.floor.clearance(point) < CAR_RADIUS_CM:
            return "wall"
        # Touching the cat means the car moving into it (it may walk up close on its own).
        now = math.hypot(self.cat.x - point[0], self.cat.y - point[1])
        if now < TOUCH_CM and now < self._cat_distance() - 0.5:
            return "cat"
        return None

    def _track_cat(self) -> None:
        self.closest_cat = min(self.closest_cat, self._cat_distance())

    def turn(self, degrees: float, precise: bool) -> bool:
        self.truth = (self.truth[0], self.truth[1], self.truth[2] + degrees)
        self._spend(abs(degrees) / 150.0 + (0.8 if precise else 0.2))
        self.cat.car = self.truth[:2]
        self.cat.reacts_to("turn", self._cat_distance())
        return not self.stopped()

    def move(self, forward_cm: float, sideways_cm: float, speed: float = 1.0) -> bool:
        heading = direction(self.truth[2])
        right = np.array([heading[1], -heading[0]])
        step_vector = heading * forward_cm + right * sideways_cm
        length = float(np.linalg.norm(step_vector))
        if self.person is not None and forward_cm > 0:
            offset = np.array(self.person) - np.array(self.truth[:2])
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            if abs((bearing - self.truth[2] + 180) % 360 - 180) <= 25 and np.linalg.norm(offset) < 150:
                self.moves_toward_person += 1
        position = np.array(self.truth[:2], float)
        for fraction in np.linspace(0, 1, max(2, int(length / 2) + 1))[1:]:
            candidate = np.array(self.truth[:2]) + step_vector * fraction
            hit = self._collide(candidate)
            if hit == "wall":
                self.bumps += 1
                break
            if hit == "cat":
                self.touches += 1
                break
            position = candidate
        self.truth = (float(position[0]), float(position[1]), self.truth[2])
        self._spend(length / (50.0 * speed) + 0.25)
        self._track_cat()
        kind = "strafe" if abs(sideways_cm) > abs(forward_cm) else "forward" if forward_cm > 0 else "back"
        self.cat.car = self.truth[:2]
        self.cat.reacts_to(kind, self._cat_distance())
        return not self.stopped()

    def arc(self, strafe: float, steering: float, seconds: float) -> bool:
        steps = max(1, int(seconds / 0.05))
        for _ in range(steps):
            heading = direction(self.truth[2])
            right = np.array([heading[1], -heading[0]])
            candidate = np.array(self.truth[:2]) + right * strafe * 60.0 * 0.05
            hit = self._collide(candidate)
            if hit == "wall":
                self.bumps += 1
                break
            if hit == "cat":
                self.touches += 1
                break
            self.truth = (float(candidate[0]), float(candidate[1]), self.truth[2] + steering * 384.0 * 0.05)
            self._track_cat()
        self._spend(seconds + 0.2)
        self.cat.car = self.truth[:2]
        self.cat.reacts_to("strafe", self._cat_distance())
        return not self.stopped()

    def report(self, status: dict) -> None:
        self.statuses.append(status)
