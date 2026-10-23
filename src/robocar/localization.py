"""Pose estimation between mapping stops.

Each stop's pose is estimated in three layers:

1. Wheel odometry, integrated from the actual ramped motor output, predicts
   how far the car moved and turned (mecanum wheels slip, so it is uncertain).
2. Camera views from the previous and current stop measure the yaw change
   precisely and the direction of travel without scale. Disagreements with
   odometry are gated: agreeing views overrule slipping wheels, while a lone
   disagreeing view is ignored.
3. The new ultrasonic sweep is registered against walls and features that are
   already in the surface map. This regularized point-to-line fit supplies the
   metric correction that monocular vision cannot. It is accepted only when it
   is statistically consistent with the prediction.

Poses are (x_cm, y_cm, heading_deg) with headings counterclockwise. Covariances
are 3x3 in (cm, cm, radians).
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from robocar.surfaces import Echo, Surface, SurfaceMap, pose_rotation

Pose = tuple[float, float, float]
REGISTRATION_ITERATIONS = 12
REGISTRATION_MIN_MATCHES = 4
REGISTRATION_MAX_RMS_CM = 6.0
# 99% chi-square bound for three degrees of freedom.
REGISTRATION_GATE_CHI2 = 11.3
VISUAL_YAW_SIGMA_DEGREES = 1.5
VISUAL_SINGLE_YAW_SIGMA_DEGREES = 4.0
VISUAL_BEARING_SIGMA_DEGREES = 8.0
VISUAL_OVERRIDE_MAX_DEGREES = 30.0


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def compose(pose: Sequence[float], delta: Sequence[float]) -> Pose:
    offset = pose_rotation(pose[2]) @ np.array([delta[0], delta[1]])
    return (
        float(pose[0] + offset[0]),
        float(pose[1] + offset[1]),
        wrap_degrees(float(pose[2]) + float(delta[2])),
    )


def relative(start: Sequence[float], end: Sequence[float]) -> Pose:
    offset = pose_rotation(start[2]).T @ np.array([end[0] - start[0], end[1] - start[1]])
    return (float(offset[0]), float(offset[1]), wrap_degrees(float(end[2]) - float(start[2])))


def pose_sigma(covariance: np.ndarray) -> tuple[float, float]:
    """Return the worst-axis position sigma in cm and the heading sigma in degrees."""
    position = float(np.sqrt(max(np.linalg.eigvalsh(covariance[:2, :2]).max(), 0.0)))
    return position, math.degrees(math.sqrt(max(float(covariance[2, 2]), 0.0)))


@dataclass(frozen=True)
class VisualMotion:
    yaw_deg: float
    travel_bearing_deg: float | None
    estimates: int
    spread_deg: float


@dataclass(frozen=True)
class MotionPrediction:
    delta: Pose
    covariance: np.ndarray
    odometry_delta: Pose
    visual_used: bool
    visual_rejected: bool
    bearing_used: bool


def predict_motion(odometry_delta: Sequence[float], visual: VisualMotion | None) -> MotionPrediction:
    dx, dy, odometry_yaw = (float(value) for value in odometry_delta)
    distance = math.hypot(dx, dy)
    along_sigma = 1.0 + 0.2 * distance
    lateral_sigma = 1.0 + 0.12 * distance
    odometry_yaw_sigma = 1.0 + 0.25 * abs(odometry_yaw) + 0.03 * distance

    yaw, yaw_sigma = odometry_yaw, odometry_yaw_sigma
    visual_used = visual_rejected = bearing_used = False
    if visual is not None:
        visual_sigma = (
            VISUAL_YAW_SIGMA_DEGREES + visual.spread_deg / 2
            if visual.estimates >= 2
            else VISUAL_SINGLE_YAW_SIGMA_DEGREES
        )
        disagreement = abs(wrap_degrees(visual.yaw_deg - odometry_yaw))
        if disagreement <= 3 * math.hypot(visual_sigma, odometry_yaw_sigma):
            weight_odometry, weight_visual = odometry_yaw_sigma**-2, visual_sigma**-2
            yaw = (odometry_yaw * weight_odometry + (odometry_yaw + wrap_degrees(visual.yaw_deg - odometry_yaw)) * weight_visual) / (
                weight_odometry + weight_visual
            )
            yaw_sigma = (weight_odometry + weight_visual) ** -0.5
            visual_used = True
        elif visual.estimates >= 2 and visual.spread_deg <= 3.0 and disagreement <= VISUAL_OVERRIDE_MAX_DEGREES:
            # Several camera estimates agree with each other but not with the
            # wheels: the wheels slipped or the car was moved by hand. Far
            # larger disagreements are camera mistakes, not slip.
            yaw, yaw_sigma = visual.yaw_deg, visual_sigma
            visual_used = True
        else:
            visual_rejected = True

    # Odometry integrated along its own heading; bend the path by half of the
    # heading correction, then point it along the camera's travel direction.
    travel = pose_rotation((yaw - odometry_yaw) / 2) @ np.array([dx, dy])
    if distance > 5.0 and visual is not None and visual.travel_bearing_deg is not None and not visual_rejected:
        odometry_bearing = math.degrees(math.atan2(-travel[0], travel[1]))
        difference = wrap_degrees(visual.travel_bearing_deg - odometry_bearing)
        if abs(difference) <= 30.0:
            odometry_bearing_sigma = math.degrees(math.atan2(lateral_sigma, distance))
            weight_odometry = odometry_bearing_sigma**-2
            weight_visual = VISUAL_BEARING_SIGMA_DEGREES**-2
            fused_bearing = odometry_bearing + difference * weight_visual / (weight_odometry + weight_visual)
            radians = math.radians(fused_bearing)
            travel = distance * np.array([-math.sin(radians), math.cos(radians)])
            lateral_sigma = distance * math.radians((weight_odometry + weight_visual) ** -0.5) + 0.5
            bearing_used = True

    if distance > 1e-6:
        unit = travel / max(float(np.linalg.norm(travel)), 1e-9)
    else:
        unit = np.array([0.0, 1.0])
    side = np.array([unit[1], -unit[0]])
    position_covariance = along_sigma**2 * np.outer(unit, unit) + lateral_sigma**2 * np.outer(side, side)
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = position_covariance
    covariance[2, 2] = math.radians(yaw_sigma) ** 2
    return MotionPrediction(
        (float(travel[0]), float(travel[1]), wrap_degrees(yaw)),
        covariance,
        (dx, dy, odometry_yaw),
        visual_used,
        visual_rejected,
        bearing_used,
    )


def propagate(pose: Sequence[float], covariance: np.ndarray, prediction: MotionPrediction) -> tuple[Pose, np.ndarray]:
    heading = math.radians(pose[2])
    dx, dy = prediction.delta[0], prediction.delta[1]
    pose_jacobian = np.eye(3)
    pose_jacobian[0, 2] = -math.sin(heading) * dx - math.cos(heading) * dy
    pose_jacobian[1, 2] = math.cos(heading) * dx - math.sin(heading) * dy
    delta_jacobian = np.eye(3)
    delta_jacobian[:2, :2] = pose_rotation(pose[2])
    new_covariance = (
        pose_jacobian @ covariance @ pose_jacobian.T
        + delta_jacobian @ prediction.covariance @ delta_jacobian.T
    )
    return compose(pose, prediction.delta), new_covariance


@dataclass(frozen=True)
class RegistrationResult:
    pose: Pose
    covariance: np.ndarray
    accepted: bool
    matches: int
    rms_cm: float
    correction: Pose
    reason: str

    def summary(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "matches": self.matches,
            "rms_cm": round(self.rms_cm, 2),
            "correction": [round(value, 2) for value in self.correction],
            "reason": self.reason,
        }


def register_scan(
    echoes: Sequence[Echo],
    prior_pose: Sequence[float],
    prior_covariance: np.ndarray,
    surface_map: SurfaceMap,
) -> RegistrationResult:
    prior = (float(prior_pose[0]), float(prior_pose[1]), float(prior_pose[2]))
    walls = [wall for wall in surface_map.walls() if wall.confirmed or len(wall.points) >= 5]
    features = surface_map.features(confirmed_only=True)

    def rejected(reason: str, matches: int = 0, rms: float = 0.0) -> RegistrationResult:
        return RegistrationResult(prior, prior_covariance, False, matches, rms, (0.0, 0.0, 0.0), reason)

    if len(echoes) < REGISTRATION_MIN_MATCHES:
        return rejected("too few echoes")
    if not walls and not features:
        return rejected("no mapped surfaces yet")

    prior_state = np.array([prior[0], prior[1], math.radians(prior[2])])
    covariance = prior_covariance + np.diag([0.25, 0.25, math.radians(0.25) ** 2])
    gate_position_sigma = math.sqrt(max(float(np.linalg.eigvalsh(covariance[:2, :2]).max()), 0.0))
    gate_heading_sigma = math.sqrt(max(float(covariance[2, 2]), 0.0))

    def solve(prior_information: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
        return _fit_to_surfaces(
            echoes, prior_state, prior_information, walls, features, gate_position_sigma, gate_heading_sigma
        )

    # First let the echoes speak almost on their own, then test whether what
    # they say is consistent with the motion estimate (an innovation test).
    weak_information = np.linalg.inv(covariance + np.diag([30.0**2, 30.0**2, math.radians(20) ** 2]))
    measured_state, measured_information, matches, rms = solve(weak_information)
    if matches < REGISTRATION_MIN_MATCHES:
        return rejected("too few echoes matched mapped surfaces", matches, rms)
    if rms > REGISTRATION_MAX_RMS_CM:
        return rejected("matched echoes fit poorly", matches, rms)
    innovation = measured_state - prior_state
    innovation[2] = math.atan2(math.sin(innovation[2]), math.cos(innovation[2]))
    innovation_covariance = covariance + np.linalg.inv(measured_information)
    if float(innovation @ np.linalg.solve(innovation_covariance, innovation)) > REGISTRATION_GATE_CHI2:
        return rejected("sonar disagrees with the motion estimate", matches, rms)

    state, information, matches, rms = solve(np.linalg.inv(covariance))
    difference = state - prior_state
    difference[2] = math.atan2(math.sin(difference[2]), math.cos(difference[2]))
    return RegistrationResult(
        (float(state[0]), float(state[1]), wrap_degrees(math.degrees(state[2]))),
        np.linalg.inv(information),
        True,
        matches,
        rms,
        (float(difference[0]), float(difference[1]), math.degrees(float(difference[2]))),
        "registered against mapped surfaces",
    )


def _fit_to_surfaces(
    echoes: Sequence[Echo],
    prior_state: np.ndarray,
    prior_information: np.ndarray,
    walls: Sequence[Surface],
    features: Sequence[Surface],
    position_sigma: float,
    heading_sigma: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Regularized Gauss-Newton fit of echoes to walls (point-to-line) and features."""
    state = prior_state.copy()
    local = np.array([[echo.x_cm, echo.y_cm] for echo in echoes])
    sigmas = np.array([echo.sigma_cm for echo in echoes])
    reach = np.linalg.norm(local, axis=1)
    information = prior_information
    matches = 0
    residual_squares: list[float] = []
    for iteration in range(REGISTRATION_ITERATIONS):
        cosine, sine = math.cos(state[2]), math.sin(state[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        rotation_derivative = np.array([[-sine, -cosine], [cosine, -sine]])
        world = local @ rotation.T + state[:2]
        derivatives = local @ rotation_derivative.T
        shrink = 1.0 - iteration / REGISTRATION_ITERATIONS
        information = prior_information.copy()
        gradient = prior_information @ (state - prior_state)
        matches = 0
        residual_squares = []
        for index, point in enumerate(world):
            gate = 6.0 + shrink * (3 * position_sigma + 3 * reach[index] * heading_sigma)
            best: tuple[float, np.ndarray, np.ndarray, float] | None = None
            for wall in walls:
                along = wall.along(point)
                if not wall.near_piece(along, 10.0):
                    continue
                residual = wall.offset(point)
                if abs(residual) > gate:
                    continue
                sigma = math.hypot(sigmas[index], wall.rms_cm + 1.0)
                score = abs(residual) / sigma
                if best is None or score < best[0]:
                    jacobian = np.array([[wall.normal[0], wall.normal[1], float(wall.normal @ derivatives[index])]])
                    best = (score, jacobian, np.array([residual]), sigma)
            for feature in features:
                difference = point - feature.centroid
                distance = float(np.linalg.norm(difference))
                if distance > gate:
                    continue
                sigma = math.hypot(sigmas[index], feature.rms_cm + 2.0)
                score = distance / sigma
                if best is None or score < best[0]:
                    jacobian = np.column_stack([np.eye(2), derivatives[index]])
                    best = (score, jacobian, difference, sigma)
            if best is None:
                continue
            score, jacobian, residual, sigma = best
            huber = 1.0 if score <= 2.0 else 2.0 / score
            weight = huber / sigma**2
            information = information + weight * jacobian.T @ jacobian
            gradient = gradient + weight * jacobian.T @ residual
            matches += 1
            residual_squares.extend(float(value) ** 2 for value in residual)
        step = -np.linalg.solve(information, gradient)
        state = state + step
        if np.linalg.norm(step[:2]) < 0.05 and abs(step[2]) < math.radians(0.02):
            break
    rms = math.sqrt(sum(residual_squares) / len(residual_squares)) if residual_squares else 0.0
    return state, information, matches, rms
