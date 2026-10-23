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

Pan/tilt sliders are also on the dashboard. Their initial software travel is
limited to 30°–150° to avoid the servo end stops; calibrate the safe range on
the actual camera head before widening it.

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

After each sweep it plans its next move:

- **Forward:** a 0.25-second burst when the corridor ahead (±12°, roughly the
  car's width) has at least 60 cm clearance. Motor duty ramps up over that
  interval and down for 0.25 seconds after.
- **Look around:** after three forward bursts in a row, it turns toward the
  best side opening (at least 80 cm clear across a car-wide corridor). Turns
  are closed-loop on wheel odometry, so it turns by that opening's actual
  angle, up to 75°, not by a fixed nudge. Openings that lead away from where
  the car has already been score higher.
- **Search:** when the way ahead is blocked or gives no echo (often a smooth
  wall at a steep angle) and no side opening is clear, it turns in place 45°
  at a time, always the same way, up to four times, as long as nothing in
  the sweep is closer than 25 cm.
- **Back up:** only when that search finds nothing does it back up once for
  0.25 seconds; if still boxed in, it stops. The rear is not sensed, so keep
  the area behind the car clear and expect this recovery maneuver may bump
  into something.

If the sensor returns no readings at all, it stops with a sensor fault. The
dashboard shows the reason for the current move. Each explore session is
limited to two minutes and 30 moves. **STOP** cancels the session and stops
the drive output. A one-second watchdog stops stale commands, while
autonomous moves refresh the command until they end. Exploration never
starts at boot.

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
fewer bursts than before. Separate sessions start in separate local coordinate
frames and are not merged. The newest session's map is rebuilt from its saved
sweeps when RoboCar restarts.

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
