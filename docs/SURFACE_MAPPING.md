# Surface mapping design

RoboCar turns its swept ultrasonic ranges into a map of **surfaces** — wall
segments joined into contours, plus point features such as corners, edges and
furniture legs — and refines that map every time it stops and sweeps again.
Wheel odometry, the camera and the sonar each estimate how the car moved;
the estimates are fused so that each one covers the others' blind spots.

Code: `src/robocar/surfaces.py` (geometry and map),
`src/robocar/localization.py` (motion fusion and scan registration),
`src/robocar/mapping.py` (the per-stop pipeline), and odometry in
`src/robocar/drive.py`.

## Conventions

- Robot frame: +y forward, +x right. The sonar sits `SONAR_FORWARD_OFFSET_CM`
  ahead of the turning centre on the pan axis.
- Headings and sweep angles are degrees **counterclockwise** (positive = left).
  This matches the Freenove motor mix (positive steering turns left), the
  servo (`angle_deg = pan − 90`, so pan 150 looks left) and the camera
  yaw estimate. A direction is `(−sin θ, cos θ)`.
- A run's world frame is the car's pose at its first stop, `(0, 0, 0)`.
- The old dashboard drew `x = sin θ`, which mirrored the map left/right. The
  new renderer uses the convention above.

## What happens at each stop

1. **Camera views.** Three frames at the current pan/tilt. Their texture is
   scored (ORB keypoints × image coverage).
2. **Odometry.** `DriveController` integrates the actual ramped wheel duties
   through the inverse mecanum mix, at 100 Hz, into `(x, y, heading)`. The
   difference since the last stop is the motion guess, scaled by a learned
   per-run correction.
3. **Decide whether side pictures are needed.** If the forward view is bland,
   failed to measure motion, or disagrees with the wheels (or the previous
   stop needed side views), the sweep also grabs frames at pan 54° and 126°.
   The head is already passing through those angles, so this costs only the
   snapshots.
4. **Sweep.** 21 pan positions, 6° apart (4× the old five), three pings per
   angle, median kept. Adjacent beams overlap, so surfaces show up as
   continuous range profiles instead of isolated points.
5. **Motion fusion** (`predict_motion`):
   - Odometry gets an uncertainty that grows with distance and turn (mecanum
     wheels slip, especially when turning).
   - Camera yaw from the essential matrix is precise; the camera's travel
     direction (unscaled) corrects sideways slip.
   - Gate: if camera and wheels disagree by more than 3σ, several agreeing
     camera estimates overrule the wheels (slip, or the car was moved by hand);
     a lone disagreeing view is discarded.
   - The result is propagated into the pose and its 3×3 covariance.
6. **Sonar registration** (`register_scan`): the new echoes are fitted to the
   walls (point-to-line) and features (point-to-point) already in the map with
   a regularized, Huber-weighted Gauss–Newton solve. First the echoes are
   solved almost on their own and compared with the prediction using both
   covariances (a χ² innovation test at 99%). Only if they agree is the final
   pose fused. Directions the surfaces cannot see (along a single wall) stay
   with the prediction and keep their uncertainty.
7. **Learn the odometry scale.** When registration pins the distance travelled
   (or the camera pins the turn), the per-run translation/rotation scale is
   nudged toward the measured ratio, clamped to 0.5–2.0.
8. **Integrate the sweep into the map** (below), and save the observation with
   its pose and a JSON record of every estimate (`estimate_json`).

## From echoes to surfaces

**Beam correction.** The sensor reports the nearest reflector anywhere in a
~15° cone.

- *Regions of constant depth (RCDs).* Adjacent readings within ~1% of the local
  minimum range are one reflector. It sits at the RCD's centre bearing, where
  the surface faces the sensor. RCDs are grown outward from the nearest
  reading so slopes never pair up into false plateaus.
- *Oblique readings.* Along a wall, the range rises away from the normal, and
  the echo comes from the cone edge nearest the normal. Shifting those
  bearings half a beam toward falling range puts the points back on the wall
  instead of on a bowed arc (≈9 cm bow → <1 cm in tests).
- *Range peaks* between two nearer readings are concave corners inside the
  beam and are dropped. *Range jumps* split the sweep into separate runs
  (different objects).
- Each echo carries an uncertainty from range noise and bearing uncertainty;
  pose uncertainty is added when it is placed in the world.

**Surfaces.** Each run is split into straight pieces (recursive split, then
merge), and pieces with ≥3 echoes become wall candidates; the rest are point
echoes. A candidate fuses with an existing wall when it is parallel (≤15°),
close (offset tolerance grows with range and pose uncertainty) and overlapping
or within 20 cm along it. Walls keep their supporting points and are refitted
by weighted total least squares with outlier rejection, so every sweep refines
position, angle and extent. Afterwards the map consolidates:

- collinear overlapping walls merge;
- features lying on a wall are absorbed into it;
- nearby features merge;
- three or more collinear features are promoted to a wall.

**Contours.** Wall ends are joined at corners, using the walls' intersection
when they meet at an angle. Smooth walls stop echoing near a corner (specular
reflection), so a hidden corner up to 45 cm beyond both ends is filled in.
Two walls share at most one corner.

**Negative evidence.** A beam that travelled well past a stored wall it hit
nearly head-on, or past a feature inside its core, counts a miss against it.
Surfaces with at least two misses and as many misses as sightings are retired,
so moved boxes, opened doors and people fade out.

**Confidence.** A surface is *confirmed* after two sweeps with
confidence ≥ 0.5 (`sightings / (sightings + 2·misses + 1)`). Only confirmed
surfaces (or walls with ≥5 points) anchor registration.

## Persistence and API

- Each observation stores the raw sweep, the refined pose and the estimate
  record. On start-up the newest run's map is rebuilt by replaying its sweeps,
  so nothing derived needs migrating when the algorithms improve.
- `GET /api/mapping/surfaces` returns surfaces, features, contours, recent
  echoes, the path and the pose uncertainty. `/api/status` reports which
  sources fixed the latest pose.

## Calibrate on the car

Simulation tests validate the geometry. These constants decide how well it
transfers to the real car:

| Constant | File | How to measure |
| --- | --- | --- |
| `SONAR_FORWARD_OFFSET_CM` | surfaces.py | Turning centre to the pan axis. |
| `SONAR_BEAM_WIDTH_DEGREES` | surfaces.py | Sweep a flat wall at 1 m; the RCD width is the beam width. |
| `FORWARD_CM_PER_SECOND_AT_FULL_DUTY` | drive.py | Drive 1 s forward; measure. |
| `TURN_DEGREES_PER_SECOND_AT_FULL_DUTY` | drive.py | Rotate 2 s; measure. The camera also reports turns. |
| `STRAFE_CM_PER_SECOND_AT_FULL_DUTY`, `MOTOR_DEADBAND_FRACTION` | drive.py | Strafe 1 s; lowest duty that moves the car. |
| `CAMERA_HORIZONTAL_FOV_DEGREES` | mapping.py | IMX477 lens field of view. |

Also check the sign convention once: put a box on the car's left and confirm
the dashboard draws it on the left.

## Known limits and next steps

- A sweep now takes about 6–7 s, so the two-minute explore limit fits fewer
  bursts.
- Close to a wall (≲30 cm) the whole sweep collapses into one RCD. Registration
  declines and the pose relies on wheels and camera for that stop.
- Soft furnishings absorb sound, and very oblique smooth surfaces never echo.
  Expect gaps rather than false walls.
- Runs are separate frames. Next steps: relocalize a new run against the
  previous map (global registration search), close loops with a pose graph,
  and let exploration aim for unexplained contour ends (frontiers).
