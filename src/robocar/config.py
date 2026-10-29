from dataclasses import dataclass
import os
from pathlib import Path

from robocar.detector import default_hailo_model


def _default_mapping_database() -> Path:
    data_home = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "robocar" / "mapping.sqlite3"


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 5000
    camera_backend: str = "picamera2"
    camera_index: int = 0
    camera_enabled: bool = False
    frame_width: int = 640
    frame_height: int = 480
    model_path: Path | None = None
    labels_path: Path | None = None
    # Object detection on the AI HAT+ (Hailo); None runs it on the CPU only.
    hailo_model_path: Path | None = None
    confidence: float = 0.35
    nms_threshold: float = 0.45
    hardware_enabled: bool = False
    freenove_server_path: Path | None = None
    max_motor_duty: int = 4095
    command_timeout: float = 1.0
    mapping_database: Path = _default_mapping_database()

    @classmethod
    def from_env(cls) -> "Config":
        backend = os.getenv("ROBOCAR_CAMERA_BACKEND", "picamera2").strip().lower()
        if backend not in {"opencv", "picamera2"}:
            raise ValueError("ROBOCAR_CAMERA_BACKEND must be 'opencv' or 'picamera2'")

        confidence = float(os.getenv("ROBOCAR_CONFIDENCE", "0.35"))
        nms_threshold = float(os.getenv("ROBOCAR_NMS_THRESHOLD", "0.45"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("ROBOCAR_CONFIDENCE must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("ROBOCAR_NMS_THRESHOLD must be between 0 and 1")

        default_model = str(Path.home() / ".cache" / "robocar" / "yolo11n.onnx")
        model = os.getenv("ROBOCAR_MODEL_PATH", default_model).strip()
        # auto: the AI HAT+ when present, the CPU model otherwise (and as a
        # fallback); hailo or cpu: only that; off: no object detection.
        detector = os.getenv("ROBOCAR_DETECTOR", "auto").strip().lower()
        if detector not in {"auto", "hailo", "cpu", "off"}:
            raise ValueError("ROBOCAR_DETECTOR must be 'auto', 'hailo', 'cpu' or 'off'")
        hailo_model = os.getenv("ROBOCAR_HAILO_MODEL", "").strip()
        hailo_path = Path(hailo_model).expanduser() if hailo_model else default_hailo_model()
        if detector in {"cpu", "off"}:
            hailo_path = None
        if detector in {"hailo", "off"}:
            model = ""
        labels = os.getenv("ROBOCAR_LABELS_PATH", "").strip()
        kit_path = os.getenv("FREENOVE_SERVER_PATH", "").strip()
        mapping_database = os.getenv("ROBOCAR_MAP_PATH", "").strip()
        max_duty = int(os.getenv("ROBOCAR_MAX_MOTOR_DUTY", "4095"))
        timeout = float(os.getenv("ROBOCAR_COMMAND_TIMEOUT", "1.0"))
        if not 0 < max_duty <= 4095:
            raise ValueError("ROBOCAR_MAX_MOTOR_DUTY must be between 1 and 4095")
        if timeout <= 0:
            raise ValueError("ROBOCAR_COMMAND_TIMEOUT must be greater than 0")

        return cls(
            host=os.getenv("ROBOCAR_HOST", "127.0.0.1"),
            port=int(os.getenv("ROBOCAR_PORT", "5000")),
            camera_backend=backend,
            camera_index=int(os.getenv("ROBOCAR_CAMERA_INDEX", "0")),
            camera_enabled=os.getenv("ROBOCAR_CAMERA_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            frame_width=int(os.getenv("ROBOCAR_FRAME_WIDTH", "640")),
            frame_height=int(os.getenv("ROBOCAR_FRAME_HEIGHT", "480")),
            model_path=Path(model).expanduser() if model else None,
            labels_path=Path(labels).expanduser() if labels else None,
            hailo_model_path=hailo_path,
            confidence=confidence,
            nms_threshold=nms_threshold,
            hardware_enabled=os.getenv("ROBOCAR_HARDWARE_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            freenove_server_path=Path(kit_path) if kit_path else None,
            max_motor_duty=max_duty,
            command_timeout=timeout,
            mapping_database=Path(mapping_database).expanduser()
            if mapping_database
            else _default_mapping_database(),
        )
