import logging
import math

from flask import Flask, Response, jsonify, render_template, request

from robocar.camera import CameraService
from robocar.config import Config
from robocar.detector import ObjectDetector
from robocar.drive import (
    DriveController,
    DriveInhibitedError,
    FreenoveDriver,
    FreenovePanTilt,
    PanTiltInhibitedError,
    PanTiltController,
    SimulatedDriver,
    SimulatedPanTilt,
)
from robocar.mapping import MappingController, MappingStore
from robocar.range_sensor import FreenoveRangeSensor, RangeSensorController


logger = logging.getLogger(__name__)


def _open_hardware(name: str, opener):
    """Open a piece of the car's hardware; on failure log why and return the reason.

    A car board that is switched off or unplugged must not take the whole
    dashboard (and the camera with it) down: that part is reported as
    unavailable instead."""
    try:
        return opener(), None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("%s unavailable: %s", name, reason)
        return None, reason


def create_app(config: Config | None = None) -> Flask:
    config = config or Config.from_env()
    drive_fault = pan_tilt_fault = range_fault = None
    if config.hardware_enabled:
        driver, drive_fault = _open_hardware("Motor board", lambda: FreenoveDriver(config.freenove_server_path))
        pan_tilt_driver, pan_tilt_fault = _open_hardware(
            "Pan/tilt servos", lambda: FreenovePanTilt(config.freenove_server_path)
        )
        range_sensor_driver, range_fault = _open_hardware(
            "Ultrasonic sensor", lambda: FreenoveRangeSensor(config.freenove_server_path)
        )
    else:
        driver, pan_tilt_driver, range_sensor_driver = SimulatedDriver(), SimulatedPanTilt(), None
    drive = DriveController(
        driver or SimulatedDriver(),
        config.max_motor_duty,
        config.command_timeout,
        config.hardware_enabled,
        fault=drive_fault,
    )
    pan_tilt = PanTiltController(pan_tilt_driver or SimulatedPanTilt(), config.hardware_enabled, fault=pan_tilt_fault)
    range_sensor = RangeSensorController(range_sensor_driver, config.hardware_enabled, fault=range_fault)
    detector = ObjectDetector(
        config.model_path,
        config.labels_path,
        config.confidence,
        config.nms_threshold,
        hailo_model=config.hailo_model_path,
    )
    camera = CameraService(
        config.camera_backend,
        config.camera_index,
        config.frame_width,
        config.frame_height,
        detector,
        config.camera_enabled,
    )
    mapping = MappingController(
        camera=camera,
        drive=drive,
        pan_tilt=pan_tilt,
        range_sensor=range_sensor,
        store=MappingStore(config.mapping_database),
        detector=detector,
    )

    app = Flask(__name__)
    app.extensions["robocar_camera"] = camera
    app.extensions["robocar_drive"] = drive
    app.extensions["robocar_pan_tilt"] = pan_tilt
    app.extensions["robocar_range_sensor"] = range_sensor
    app.extensions["robocar_mapping"] = mapping

    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/video.mjpg")
    def video():
        return Response(
            camera.stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "camera": camera.status(),
                "drive": drive.status(),
                "range": range_sensor.status(),
                "mapping": mapping.status(),
            }
        )

    @app.post("/api/mapping")
    def mapping_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        try:
            if payload["enabled"]:
                mapping.start()
            else:
                mapping.stop()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.post("/api/mapping/explore")
    def autonomous_mapping_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        if payload["enabled"] and not isinstance(payload.get("attended"), bool):
            return jsonify({"error": "attended must be a boolean"}), 400
        try:
            if payload["enabled"]:
                mapping.start_autonomous(payload["attended"])
            else:
                mapping.stop()
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.get("/api/calibration")
    def calibration_status():
        return jsonify({"calibration": mapping.calibration()})

    @app.post("/api/calibration")
    def calibration_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        if payload["enabled"] and not isinstance(payload.get("attended"), bool):
            return jsonify({"error": "attended must be a boolean"}), 400
        try:
            if payload["enabled"]:
                mapping.start_calibration(payload["attended"])
            else:
                mapping.stop()
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.get("/api/mapping/observations")
    def mapping_observations():
        return jsonify({"observations": mapping.observations()})

    @app.get("/api/mapping/coverage.png")
    def mapping_coverage():
        image = mapping.coverage_png()
        if image is None:
            return jsonify({"error": "Nothing swept yet"}), 404
        return Response(image, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/mapping/vision")
    def mapping_vision():
        return jsonify({"panels": mapping.vision_panels()})

    @app.get("/api/mapping/vision/<name>.jpg")
    def mapping_vision_image(name: str):
        image = mapping.vision_image(name)
        if image is None:
            return jsonify({"error": "No such picture yet"}), 404
        return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/mapping/surfaces")
    def mapping_surfaces():
        return jsonify(mapping.surface_map())

    @app.post("/api/mapping/reset")
    def mapping_reset():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("confirm") is not True:
            return jsonify({"error": "confirm must be true"}), 400
        try:
            mapping.reset()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.get("/api/mapping/observations/<int:observation_id>/image")
    def mapping_observation_image(observation_id: int):
        image = mapping.image(observation_id)
        if image is None:
            return jsonify({"error": "Observation not found"}), 404
        return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/camera")
    def camera_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        enabled = camera.set_enabled(payload["enabled"])
        return jsonify({"enabled": enabled})

    @app.post("/api/drive")
    def drive_command():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        throttle = payload.get("throttle")
        strafe = payload.get("strafe")
        steering = payload.get("steering")
        if (
            isinstance(throttle, bool)
            or isinstance(strafe, bool)
            or isinstance(steering, bool)
            or not isinstance(throttle, (int, float))
            or not isinstance(strafe, (int, float))
            or not isinstance(steering, (int, float))
        ):
            return jsonify({"error": "throttle, strafe, and steering must be numbers"}), 400
        try:
            throttle_value, strafe_value, steering_value = (
                float(throttle),
                float(strafe),
                float(steering),
            )
        except (OverflowError, ValueError):
            return jsonify({"error": "throttle, strafe, and steering must be finite numbers"}), 400
        if not all(math.isfinite(value) for value in (throttle_value, strafe_value, steering_value)):
            return jsonify({"error": "throttle, strafe, and steering must be finite numbers"}), 400
        try:
            command = drive.set_command(throttle_value, strafe_value, steering_value)
        except DriveInhibitedError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"throttle": command[0], "strafe": command[1], "steering": command[2]})

    @app.post("/api/stop")
    def stop():
        if mapping.status()["enabled"]:
            mapping.stop()
        drive.stop()
        return jsonify({"stopped": True})

    @app.get("/api/pan-tilt")
    def pan_tilt_status():
        return jsonify(pan_tilt.status())

    @app.post("/api/pan-tilt")
    def pan_tilt_command():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        pan, tilt = payload.get("pan"), payload.get("tilt")
        if (
            isinstance(pan, bool)
            or not isinstance(pan, int)
            or isinstance(tilt, bool)
            or not isinstance(tilt, int)
            or not 30 <= pan <= 150
            or not 30 <= tilt <= 150
        ):
            return jsonify({"error": "pan and tilt must be integers from 30 to 150"}), 400
        try:
            angles = pan_tilt.set_angles(pan, tilt)
        except PanTiltInhibitedError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"pan": angles[0], "tilt": angles[1]})

    return app
