"""A simulated house, car and cat for the hunt.

The floor plan is explore_sim's two rooms and corridor. The car's motion is
exact (odometry is the truth), so what is tested is the hunting and playing.
The camera sees the cat only inside its field of view, within 4 m and with
nothing in the way; the sonar echoes off it like a post. The cat wanders now
and then and reacts to the car's moves (to all of them, or only to the kinds
it likes), and never sits closer than 25 cm to a wall.

Like the real car's camera feed, the view is always current: `sighting()`
is what the camera shows now, and with interrupts on, a cat coming into
view ends a turn, move or sweep at once (checked every tenth of a second).
Steady motion set with `command()` happens while time passes in `wait()`.
"""

import math
import random

import numpy as np

from explore_sim import FloorPlan, NO_ECHO_CM, ROOM_A, CORRIDOR, ROOM_B
from robocar.exploration import CoverageGrid
from robocar.drive import (
    FORWARD_CM_PER_SECOND_AT_FULL_DUTY,
    MOTOR_DEADBAND_FRACTION,
    STRAFE_CM_PER_SECOND_AT_FULL_DUTY,
    TURN_DEGREES_PER_SECOND_AT_FULL_DUTY,
)
from robocar.hunt import CAMERA_HFOV_DEG, DEFAULT_SIZE_SCALE, PERSON_AHEAD_DEG, CatView, Look, Sighting, pan_bearing
from robocar.surfaces import SONAR_BEAM_WIDTH_DEGREES, direction, sonar_origin

CAT_RADIUS_CM = 10.0
TOUCH_CM = 20.0
CAR_RADIUS_CM = 15.0
FEED_CHECK_SECONDS = 0.1
HEAD_SECONDS_PER_DEGREE = 0.002


def speed_fraction(command: float) -> float:
    effective = max(0.0, abs(command) - MOTOR_DEADBAND_FRACTION) / (1 - MOTOR_DEADBAND_FRACTION)
    return math.copysign(effective, command)


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
    def __init__(
        self, floor: FloorPlan, grid: CoverageGrid, pose, cat: SimCat, rng: random.Random, landmarks=(), camera_fixed=True, lidar=False
    ):
        self.floor = floor
        # A lidar on the head beside the sonar: precise ranges on a 2 degree beam.
        self.lidar = lidar
        self.precise_ranges = 0
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
        # Like the real car: the camera fixed facing forward (or on the head).
        self.camera_fixed = camera_fixed
        self.interrupts = False
        self._spotted = None
        self.interrupted = 0  # actions a cat sighting cut short
        self.first_visible_at = None  # when the cat first came into the camera's view
        self.command_now = (0.0, 0.0, 0.0)
        self.commanded_seconds = 0.0  # time spent driving on steady commands
        self._react_clock = 0.0

    # -- time -----------------------------------------------------------------
    def now(self) -> float:
        return self.t

    def _spend(self, seconds: float) -> None:
        self.t += seconds
        self.cat.car = self.truth[:2]
        self.cat.idle(seconds)

    def wait(self, seconds: float) -> bool:
        remaining = max(0.0, seconds)
        while remaining > 1e-9:
            dt = min(0.05, remaining)
            remaining -= dt
            if self.command_now != (0.0, 0.0, 0.0):
                self._drive_for(dt)
            self._spend(dt)
            if self.command_now != (0.0, 0.0, 0.0):
                self._feed()
        return not self.stopped()

    def command(self, throttle: float, strafe: float, steering: float) -> None:
        self.command_now = (float(throttle), float(strafe), float(steering))

    def halt(self) -> None:
        self.command_now = (0.0, 0.0, 0.0)

    def _drive_for(self, dt: float) -> None:
        throttle, strafe, steering = self.command_now
        self.commanded_seconds += dt
        heading = direction(self.truth[2])
        right = np.array([heading[1], -heading[0]])
        velocity = (
            heading * FORWARD_CM_PER_SECOND_AT_FULL_DUTY * speed_fraction(throttle)
            + right * STRAFE_CM_PER_SECOND_AT_FULL_DUTY * speed_fraction(strafe)
        )
        candidate = np.array(self.truth[:2]) + velocity * dt
        if self.person is not None and throttle > 0:
            offset = np.array(self.person) - np.array(self.truth[:2])
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            if abs((bearing - self.truth[2] + 180) % 360 - 180) <= 25 and np.linalg.norm(offset) < 150:
                self.moves_toward_person += 1
        hit = self._collide(candidate)
        if hit == "wall":
            self.bumps += 1
            self.command_now = (0.0, 0.0, 0.0)
            return
        if hit == "cat":
            self.touches += 1
            self.command_now = (0.0, 0.0, 0.0)
            return
        turn = TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * speed_fraction(steering) * dt
        self.truth = (float(candidate[0]), float(candidate[1]), self.truth[2] + turn)
        self._track_cat()
        self.cat.car = self.truth[:2]
        # The cat notices a car driving about now and then.
        self._react_clock += dt
        if self._react_clock >= 1.0:
            self._react_clock = 0.0
            kind = "turn" if abs(throttle) < 0.05 and abs(strafe) < 0.05 else "forward" if throttle > 0 else "strafe"
            self.cat.reacts_to(kind, self._cat_distance())

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
        self._point(pan)
        self.looks += 1
        self._spend(0.07)  # the next frame
        return self._view(self._camera_pan())

    def camera_pans(self) -> bool:
        return not self.camera_fixed

    def _camera_pan(self) -> int:
        return 90 if self.camera_fixed else self.pan

    def _point(self, pan: int) -> None:
        if pan != self.pan:
            self._spend(0.05 + HEAD_SECONDS_PER_DEGREE * abs(pan - self.pan))
            self.pan = pan

    def sighting(self) -> Sighting:
        pan = self._camera_pan()
        return Sighting(self.t, pan, self.truth, self._view(pan), self.command_now != (0.0, 0.0, 0.0))

    def interrupt_on_cat(self, enabled: bool) -> None:
        self.interrupts = enabled
        self._spotted = None

    def spotted(self):
        return self._spotted

    def clear_spotted(self) -> None:
        self._spotted = None

    def _feed(self) -> bool:
        """What the continuous feed sees now: True if a cat sighting should
        interrupt the current action."""
        pan = self._camera_pan()
        look = self._view(pan)
        if look.cat is not None:
            if self.first_visible_at is None:
                self.first_visible_at = self.t
            self._spotted = Sighting(self.t, pan, self.truth, look, True)
            if self.interrupts:
                self.interrupted += 1
                return True
        return False

    def knows_where_it_is(self) -> bool:
        return True

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

    def ping(self, pan: int, precise: bool = False) -> float | None:
        self._point(pan)
        self._spend(0.04 if self.lidar else 0.1)
        return self._range(pan, precise)

    def range_ahead(self, precise: bool = False) -> float | None:
        self._point(90)
        return self._range(90, precise)

    def ranges_precisely(self) -> bool:
        return self.lidar

    def _range(self, pan: int, precise: bool = False) -> float | None:
        if not self.lidar:
            return self._sonar_range(pan)
        lidar = self._lidar_range(pan)
        if precise:
            self.precise_ranges += 1
            return lidar
        sonar = self._sonar_range(pan)
        return min(value for value in (lidar, sonar) if value is not None) if lidar is not None or sonar is not None else None

    def _lidar_range(self, pan: int) -> float | None:
        angle = pan_bearing(pan)
        best = self.floor.sonar.lidar_distance(self.truth, angle)
        origin = sonar_origin(self.truth)
        for thing, radius in ((np.array([self.cat.x, self.cat.y]), CAT_RADIUS_CM), (None if self.person is None else np.array(self.person), 15.0)):
            if thing is None:
                continue
            offset = thing - origin
            distance = float(np.linalg.norm(offset))
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            half = math.degrees(math.atan2(radius, max(distance, 1.0))) + 1.0
            if abs((bearing - self.truth[2] - angle + 180) % 360 - 180) <= half:
                best = distance - radius if best is None else min(best, distance - radius)
        return None if best is None else round(best + self.rng.gauss(0.0, 1.5), 1)

    def _sonar_range(self, pan: int) -> float | None:
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
        """A sonar sweep, the head passing 33 to 150; the feed watches throughout."""
        self.observations += 1
        for pan in range(33, 151, 9):
            self.pan = pan
            self._spend(10.0 / 14)
            if self._feed():
                self.pan = 90
                return
        self.pan = 90
        self._grid.integrate(self.floor.scan(self.truth), self.truth, SONAR_BEAM_WIDTH_DEGREES)

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
        step = math.copysign(min(abs(degrees), 15.0), degrees)
        turned = 0.0
        while abs(turned) < abs(degrees) - 1e-9:
            piece = math.copysign(min(abs(step), abs(degrees) - abs(turned)), degrees)
            self.truth = (self.truth[0], self.truth[1], self.truth[2] + piece)
            turned += piece
            self._spend(abs(piece) / 150.0)
            if self._feed():
                break
        self._spend(0.8 if precise else 0.2)
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
        start = np.array(self.truth[:2], float)
        since_check = 0.0
        for fraction in np.linspace(0, 1, max(2, int(length / 2) + 1))[1:]:
            candidate = start + step_vector * fraction
            hit = self._collide(candidate)
            if hit == "wall":
                self.bumps += 1
                break
            if hit == "cat":
                self.touches += 1
                break
            since_check += float(np.linalg.norm(candidate - position))
            position = candidate
            if since_check >= 50.0 * speed * FEED_CHECK_SECONDS:
                since_check = 0.0
                self.truth = (float(position[0]), float(position[1]), self.truth[2])
                if self._feed():
                    break
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
            if _ % 2 and self._feed():
                break
        self._spend(seconds + 0.2)
        self.cat.car = self.truth[:2]
        self.cat.reacts_to("strafe", self._cat_distance())
        return not self.stopped()

    def report(self, status: dict) -> None:
        self.statuses.append(status)
