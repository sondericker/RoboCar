# RoboCar

RoboCar is a Raspberry Pi 4WD car application scaffold for a VNC-accessible
camera dashboard, object detection, manual driving, and later autonomous
navigation.

## Quick start

Start RoboCar with:

```bash
./start-robocar.sh
```

The script activates `/home/sondericker/robotics-venv` and launches the app.
Press Ctrl+C in the terminal to stop it.

If the virtual environment does not exist yet, create it and install RoboCar:

```bash
python3 -m venv --system-site-packages /home/sondericker/robotics-venv
/home/sondericker/robotics-venv/bin/pip install -e /home/sondericker/Documents/Python/RoboCarv1
cp .env.example .env
```

Open `http://127.0.0.1:5000` in the Raspberry Pi's desktop browser (including
through VNC). The app binds to loopback by default. Without hardware enabled,
drive commands are simulated and cannot move the car. To stop it, press
Ctrl+C in the terminal. Camera processing starts paused; press **Start camera
processing** when you want the live feed and detections.

The dashboard loads a small COCO-pretrained YOLO11 ONNX model by default and
draws labeled boxes, including the `person` and `cat` classes. On first launch
the application downloads the official `yolo11n.onnx` asset to
`~/.cache/robocar/`. Set `ROBOCAR_MODEL_PATH` to another YOLO ONNX model to
change it. Inference uses ONNX Runtime CPU providers.

## Raspberry Pi setup

The confirmed hardware profile is the Freenove FNK0043 Rev. 2/B kit, mecanum
wheels, and Raspberry Pi IMX477 camera. Picamera2 is the selected camera
backend; verify the sensor is detected with `rpicam-hello --list-cameras`.
Install the Pi-specific
system packages there; `requirements-hardware.txt` documents the Python-side
extras. The `picamera2` package is normally supplied by Raspberry Pi OS and
should be installed with its OS package manager, not from pip. If using a
virtual environment with Picamera2, create it with `--system-site-packages`.

Copy `.env.example` to `.env` and set:

```dotenv
ROBOCAR_CAMERA_BACKEND=picamera2
ROBOCAR_HARDWARE_ENABLED=true
FREENOVE_SERVER_PATH=/path/to/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
ROBOCAR_MODEL_PATH=~/.cache/robocar/yolo11n.onnx
ROBOCAR_LABELS_PATH=
```

Keep the car's wheels off the ground while first testing physical controls.
The motor adapter is disabled unless `ROBOCAR_HARDWARE_ENABLED=true`. Dashboard
controls are a convenience, not a safety-rated emergency stop. Do not expose
the unauthenticated control server to an untrusted network; its default
loopback binding is intentional.

If hardware startup reports that the PCA9685 at I²C address `0x40` did not
respond, make sure the car battery is connected so the kit board is powered,
then check its I²C connection. On this kit the motor controller communicates
over I²C (not SPI). With the car powered, run `i2cdetect -y 1`; the board
should appear at address `40`. The YOLO/ONNX schema messages printed during
model loading are separate from this I²C error; the I²C error is what prevents
hardware mode from starting.

The drive pad sends separate forward/reverse, lateral strafe, and rotation
commands for mecanum motion. Start at low duty and verify the direction of each
axis with the wheels raised before driving on the floor. Motor duty ramps up
and down over 0.25 seconds for normal drive commands and stops. The watchdog
and explicit mapping-stop action still cut motor output immediately.

Use **Stop camera processing** on the dashboard to pause frame capture and
object detection. The camera is released while paused, and the video stream
waits without sending frames. Press **Start camera processing** to resume.

Pan/tilt sliders are also on the dashboard. The head starts at pan 90° and
tilt 100°, and **Center head** returns it there. The initial software travel
is limited to 30°–150° to avoid the servo end stops; calibrate the safe range
on the actual camera head before widening it.

When physical hardware is enabled, the dashboard also displays the live
ultrasonic range in centimeters. The sensor samples in the background and
reports missing echoes or read errors instead of presenting an old value as
current. With hardware disabled, the UI identifies the range finder as
disabled; it does not simulate a distance. The reading is informational and
does not currently interlock or stop drive commands.

## Mapping survey and attended auto explore

**Start mapping survey** records checkpoints while you drive manually.
**Start attended auto explore** enables bounded autonomous motion only when
physical hardware and the ultrasonic sensor are enabled, the car is stopped,
and the operator confirms the room is clear and attended. The car scans while
stopped, moves in short bursts, then stops and scans again. Autonomous motion
uses full normalized motor output and the default 4095/4095 motor duty cap.

After each sweep it plans its next move with **frontier exploration**:

- **Coverage grid:** every sweep marks the space the sound crossed as free
  and the places it echoed as occupied, in a 5 cm grid shared with earlier
  sessions. The edges between known-free and unknown space are the
  *frontiers*: the parts of the room not mapped yet. Spots beside and behind
  the car, which the forward sonar can never see, don't count.
- **Where to go:** it plans a path on the grid, keeping a car's radius from
  obstacles, to the best reachable frontier. Short paths and big frontiers
  score well, and left turns cost more, so it works round a room clockwise.
  It sticks with a frontier until another is clearly better. The dashboard
  marks the target with a crosshair.
- **Moving:** turns toward the target are 25° when it's near straight ahead,
  and 50° when it's far round (past 135° it keeps turning right instead of
  reversing). Forward steps are 25 cm when calibrated (otherwise 0.25-second
  bursts), and only while the corridor ahead (±12°, roughly the car's width)
  has at least 60 cm clearance. If the sonar blocks the way the grid wanted,
  it picks a different frontier.
- **Camera-measured turns:** before turning, the camera photographs a strip
  every 10° across the pan on the turn side (0–60°). The car then turns
  closed-loop on wheel odometry. Afterwards its new forward view is matched
  against every strip picture, and at least two must agree on how far it
  really turned. If the turn is off by more than 3° and the measurement
  roughly agrees with the wheels, the car makes up to two corrective turns
  (at most 20° each). The measured turn also recalibrates odometry and fixes
  the pose for the next sweep. Without trustworthy images it relies on
  odometry.
- **Finished:** with no reachable frontier left, it looks round once more
  and then stops with "explored".
- **Back up:** only when it can't turn safely (something within 25 cm), or
  after a lot of turning without finding a way forward, does it back up once.
  After backing up it turns rather than driving straight back in, and if it
  still can't move it stops. The rear is not sensed, so keep the area behind
  the car clear and expect this recovery maneuver may bump into something.

If the sensor returns no readings at all, it stops with a sensor fault. The
dashboard shows the reason for the current move. Each explore session is
limited to four minutes and 60 moves. **STOP** cancels the session and stops
the drive output. A one-second watchdog stops stale commands, while
autonomous moves refresh the command until they end. Exploration never
starts at boot.

**Pictures:** during a mapping session the camera stays open, so auto-exposure
settles between stops. Before each picture it waits until two quick frames in
a row match (the car and head have stopped moving), up to 2 seconds. Each
stop's picture is checked for brightness, clipping, sharpness and usable
detail. If it isn't decent, it is retaken: a blurred one at the same exposure
after a pause, otherwise with adjusted exposure (compensation from −1 to
+4 EV, then manual exposures up to 50 ms with gain), up to five tries. The best
picture is kept, and the setting that worked is where the next stop starts.
The **What the camera sees** panel shows:

- the latest stop picture with its detected features and exposure;
- stop-to-stop matching (green lines are matches kept, red rejected) with the
  measured camera turn;
- the latest turn check against its strip picture;
- the latest recognized place against the stored picture.

**Calibrate turns & steps** (attended) measures how far the car really turns
and moves. Face it toward a flat wall about 1–1.5 m away with room to turn.
It photographs a strip across the pan, then turns 15°, 30° and 45° left and
right, adjusting each command until the camera says it landed within 2°. It
then uses the camera to return to its starting heading and makes 10, 20 and
30 cm steps toward the wall, each followed by the same burst in reverse. It
adjusts the burst length until the sonar says each step is right (within
7%). It refuses to start with anything within 25 cm, and stops stepping
before the wall is closer than 45 cm. Results are saved beside the map
database (not cleared by reset), shown on the dashboard, and used for every
turn and step afterwards; each session keeps refining them.

Use **Reset mapping database** to permanently delete all saved observations.
The dashboard asks for confirmation, and reset is unavailable while mapping is
active. The motor duty cap and watchdog can be changed with
`ROBOCAR_MAX_MOTOR_DUTY` (maximum 4095) and `ROBOCAR_COMMAND_TIMEOUT`.
Existing systemd installations must update `/etc/robocar/robocar.env` and
restart the service for changed values to take effect.

Stay beside the car and be ready to cut battery power. Use only a clear,
single-level room without people, pets, stairs, drop-offs, or fragile objects.
The software stop and forward-facing ultrasonic scanner are not safety-rated
protective devices; the sensor cannot see behind the car or guarantee detection
of every surface. Test with wheels raised first, then at low speed in a cleared
area. Keep unauthenticated dashboard access restricted to a trusted network.

At each stop RoboCar sweeps the ultrasonic sensor across 40 directions, 3°
apart (three pings each, always stepping the same way so servo backlash
cancels). It turns the sweep into **surfaces**: fine wall segments joined at
corners into contours, plus point features such as edges and legs. Openings
and passages stay open: echoes are pulled back to the edges they came from,
walls break wherever their points leave a gap, and contours never close
across space a beam passed through. The beam-width correction, measured from
the walls themselves, straightens walls that a raw plot would bow. Each new
sweep refines, extends, merges or retires the surfaces it sees again.

To place each sweep, RoboCar estimates how the car moved:

- **Wheel odometry** integrates the actual ramped motor output.
- **The camera** compares views with the previous stop to measure the turn
  precisely and the direction of travel. When the forward view is bland or
  disagrees with the wheels, it also takes side pictures during the sweep.
- **The sonar** fits the new sweep against surfaces already mapped, which
  supplies the metric correction the camera cannot.

Each source is weighted by its uncertainty, and disagreements are tested
before they are trusted. Odometry scale is relearned during each run. The
dashboard draws confirmed surfaces solid, surfaces seen once dashed, raw
echoes as dots, and the car's path with its pose uncertainty. See
[docs/SURFACE_MAPPING.md](docs/SURFACE_MAPPING.md) for the design and the
constants to calibrate on the car.

This is still an approximate sketch, not a validated or safety-rated map.
Ultrasonic sensors miss soft and very oblique surfaces, monocular images have
no absolute scale, and wheel slip varies. Do not use it for autonomous
navigation or collision avoidance; exploration still steers from the raw
sweep. A sweep takes about 11–12 seconds, so a two-minute explore session fits
fewer bursts than before.

**Remembering places:** every stop's picture features, sweep and pose are kept
in the database across sessions. At each stop RoboCar checks whether it has
seen the place before. It matches picture features, verifies the camera
geometry, then confirms with a sonar fit against that place's remembered
surfaces. Returning to a place within a session closes a loop: a pose graph
spreads the accumulated drift back along the path, and matches that
contradict everything else are thrown out. Recognizing a place from an
earlier session moves the new session into the old one's frame, so the map
becomes one persistent house map built from every joined session, and
exploration prefers ground no session has covered yet. The dashboard draws
earlier sessions faintly and recognized places as purple dashed links. The
whole graph is rebuilt from the database when RoboCar restarts. Sessions that
never see a remembered place stay in their own frame.

By default, survey data is stored at
`~/.local/share/robocar/mapping.sqlite3` (or under `$XDG_DATA_HOME` when set).
Set `ROBOCAR_MAP_PATH` to use another database location.
Mapping stops and reports an error rather than starting a new checkpoint when
free space drops below 256 MB. The database file is restricted to the Pi
account; survey images are available through the same unauthenticated dashboard
API, so keep the service on a trusted network.

For a useful persistent room-scale occupancy map, add a calibrated 2D lidar
and use a localization/SLAM stack (with wheel odometry or an IMU if available).
With the current sensors, use the display as an approximate survey record only.

## Start at boot and connect from your Mac

After installing RoboCar into the Raspberry Pi user's Python environment, run
the installer once from a terminal on the Pi:

```bash
cd /home/sondericker/Documents/Python/RoboCarv1
./scripts/install-service.sh
```

It asks whether to enable physical motor and pan/tilt outputs (default: off),
then uses `sudo` to install and enable a systemd service. Enter your normal
account's sudo password if prompted. Camera processing is disabled at boot.
After setup, the dashboard starts automatically whenever the Pi boots and
listens on the LAN without authentication. Find the Pi address using
`hostname -I`, then visit `http://<car-ip>:5000` from the MacBook. Anyone on a
network that can reach this port can view the camera and control the car. Keep
it only on a trusted home network; do not forward the HTTP port through your
router or use it on an untrusted Wi-Fi network.

Check service logs with `sudo journalctl -u robocar -f`. Stop the running
application with:

```bash
./scripts/stop-service.sh
```

This stops RoboCar now but leaves automatic startup enabled. To stop it and
prevent it from starting at the next boot, run `sudo systemctl disable --now
robocar`.

## Kit review

The supplied Freenove tree contains 43 Python files: a Raspberry Pi server
with motor/PCA9685, servo, ultrasonic, infrared, ADC, LEDs, and Picamera2
modules; a PyQt5 desktop client/server UI; a TCP command/video protocol; and
the bundled `rpi-ws281x` library. Its setup scripts install legacy/board
specific dependencies and edit Raspberry Pi boot configuration. RoboCar does
not run those scripts or modify the kit source. It integrates the kit's
`Ordinary_Car` driver only when physical hardware is explicitly enabled.

The kit README identifies its source as Creative Commons
Attribution-NonCommercial-ShareAlike 3.0. RoboCar's adapter imports the kit
from the provided local source path rather than copying its implementation.
Review the upstream license and attribution requirements before redistributing
any kit code or derivatives.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for architecture, phased
milestones, safety limits, and decisions to verify on the actual car.
