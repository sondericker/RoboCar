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
4. **Sweep.** 40 pan positions, 3° apart (−57° to +60°), three pings per
   angle, median kept. The head first parks at pan 30 and then only steps
   upward, so every reading (and side picture) is reached from the same side
   and servo backlash cannot shift one reading against another. Adjacent beams
   overlap heavily, so surfaces show up as continuous range profiles.
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

**Beam correction.** The sensor reports the nearest reflector anywhere in its
cone. The datasheet says 15°, but on real walls it is wider; your first real
run measured about 28°. Each map starts at 25° and re-measures it (below).

- *Regions of constant depth (RCDs).* Adjacent readings within ~1% of a local
  minimum range face the sensor. They are grown outward from the nearest
  reading, and a group with a nearer neighbour is only a gentle slope, so it
  is split back up. Each reading in an RCD is laid on the surface: readings
  within half a beam of the centre mark the normal foot, and readings beyond
  that are moved half a beam toward the centre.
- *Cut-off RCDs.* When an RCD runs off the edge of the sweep, or into
  something nearer, its middle is not the normal. The normal lies at least
  half a beam beyond the edge that was seen.
- *Oblique readings.* Along a wall, the range rises away from the normal, and
  the echo comes from the cone edge nearest the normal. Shifting those
  bearings half a beam toward falling range puts the points back on the wall
  instead of on a bowed arc (≈9 cm bow → <1 cm in tests).
- *Occluding edges.* Where the range jumps farther (an opening, a doorway, the
  end of a box), the near surface keeps answering until its edge leaves the
  cone. So the edge lies half a beam back from the jump, and echoes beyond it
  are pulled back onto the edge instead of bleeding into the opening.
- *Range peaks* between nearer readings, of any width, are concave corners
  inside the beam and are dropped. *Range jumps* split the sweep into
  separate runs (different objects).
- Each echo carries an uncertainty from range noise and bearing uncertainty,
  plus the width of surface it vouches for (range × sweep step). Pose
  uncertainty is added when the echo is placed in the world.

**Beam width self-calibration.** A flat wall reads its perpendicular distance
across a flat top exactly one beam wide, then d / cos(angle past the beam
edge). Fitting that profile to every run that shows the flat top *and* rising
flanks on both sides measures the beam. With only one flank, beam width and
wall direction can't be separated, so those runs are skipped. After six good
fits, the map uses their median (clamped to 10–40°).

**Surfaces as fine segments.** Each run is split into straight pieces
(recursive split, then merge), and pieces with ≥3 echoes become wall
candidates; the rest are point echoes. A candidate fuses with an existing wall
when it is parallel (≤15°), close (offset tolerance grows with range and pose
uncertainty) and within a doorway's width (60 cm) along it. Walls keep their
supporting points and are refitted by weighted total least squares with
outlier rejection, so every sweep refines the line.

A wall is only drawn, matched and joined where its points are continuous. It
breaks into separate segments wherever neighbouring points are farther apart
than max(10 cm, 1.5 × their combined footprints). A doorway in a wall is
therefore one line model with two segments and a gap. A lone point beyond a
gap steadies the line but isn't drawn. Single new points join a wall only right
next to an existing segment, so a wall can't creep across an opening sweep by
sweep. Afterwards the map consolidates:

- collinear walls up to a doorway apart share one line (their gaps remain);
- features lying on a wall are absorbed into it;
- nearby features merge;
- three or more collinear features are promoted to a wall.

**Contours.** Segment ends of different walls are joined at corners, using the
walls' intersection when they meet at an angle. Smooth walls stop echoing near
a corner (specular reflection), so a hidden corner up to 45 cm beyond both
ends is filled in. But the map remembers the centre lines of recent beams (the
space sound actually crossed), and no join is made across a beam that passed
through. That keeps passages at corners open. Two segments share at most one
corner, and confirmed walls claim corners before fragments seen once.

**Negative evidence.** A beam that travelled well past a stored wall it hit
within 30° of head-on removes the wall points inside its core after two such
misses. This opens gaps where openings are confirmed later. It also counts a
miss against the wall, or against a feature inside the core. Surfaces with at
least two misses and as many misses as sightings are retired, so moved boxes,
opened doors and people fade out.

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
| `SONAR_BEAM_WIDTH_DEGREES` | surfaces.py | Starting value only; each map re-measures it from flat walls (`beam_width_deg` in `/api/mapping/surfaces`). |
| `FORWARD_CM_PER_SECOND_AT_FULL_DUTY` | drive.py | Drive 1 s forward; measure. |
| `TURN_DEGREES_PER_SECOND_AT_FULL_DUTY` | drive.py | Rotate 2 s; measure. The camera also reports turns. |
| `STRAFE_CM_PER_SECOND_AT_FULL_DUTY`, `MOTOR_DEADBAND_FRACTION` | drive.py | Strafe 1 s; lowest duty that moves the car. |
| `CAMERA_HORIZONTAL_FOV_DEGREES` | mapping.py | IMX477 lens field of view. |

Also check the sign convention once: put a box on the car's left and confirm
the dashboard draws it on the left.

## Known limits and next steps

- A sweep takes about 11–12 s, so the two-minute explore limit fits fewer
  bursts.
- An opening narrower than the beam's footprint at that range (about 0.44 ×
  range for a 25° beam) can't be seen through from that spot. Every reading
  hits a jamb, and it looks like a wall until sweeps from elsewhere carve it
  out.
- Close to a wall (≲30 cm) the whole sweep collapses into one RCD. Registration
  declines and the pose relies on wheels and camera for that stop.
- Soft furnishings absorb sound, and very oblique smooth surfaces never echo.
  Expect gaps rather than false walls.
- Runs are separate frames. Next steps: relocalize a new run against the
  previous map (global registration search), close loops with a pose graph,
  and let exploration aim for unexplained contour ends (frontiers).
