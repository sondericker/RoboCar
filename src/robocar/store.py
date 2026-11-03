"""Where mapping keeps what it saw: one SQLite database beside the other
saved files. Each stop (picture, sweep, pose and how it was estimated), the
place pictures for recognizing stops later (from the head camera, and from
the forward camera for the hunt), and the links between recognized places.
The file is private to the user and the database is checked for free disk
space before each stop is written."""

import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from robocar.places import Keyframe
from robocar.ranging import with_current_angles

# Mapping pauses rather than fill the disk.
MINIMUM_FREE_BYTES = 256 * 1024 * 1024

class MappingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False, timeout=10)
        if str(path) != ":memory:":
            os.chmod(path, 0o600)
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                image_jpeg BLOB NOT NULL,
                scan_json TEXT NOT NULL,
                run_id TEXT,
                mode TEXT NOT NULL DEFAULT 'survey',
                pose_x_cm REAL,
                pose_y_cm REAL,
                heading_deg REAL,
                estimate_json TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS keyframes (
                observation_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                pan_offset_deg REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                points BLOB NOT NULL,
                descriptors BLOB NOT NULL
            )
            """
        )
        # Place pictures from the forward (hunting) camera, kept apart from
        # the head camera's: pictures from different cameras do not match.
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_keyframes (
                observation_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                pan_offset_deg REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                points BLOB NOT NULL,
                descriptors BLOB NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS place_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                dx_cm REAL NOT NULL,
                dy_cm REAL NOT NULL,
                dheading_deg REAL NOT NULL,
                sigma_json TEXT NOT NULL,
                kind TEXT NOT NULL,
                inliers INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(observations)").fetchall()
        }
        for column, definition in (
            ("run_id", "TEXT"),
            ("mode", "TEXT NOT NULL DEFAULT 'survey'"),
            ("pose_x_cm", "REAL"),
            ("pose_y_cm", "REAL"),
            ("heading_deg", "REAL"),
            ("estimate_json", "TEXT"),
        ):
            if column not in columns:
                self._connection.execute(
                    f"ALTER TABLE observations ADD COLUMN {column} {definition}"
                )
        self._connection.commit()

    def save(
        self,
        image_jpeg: bytes,
        scan: list[dict[str, object]],
        run_id: str,
        mode: str,
        pose: tuple[float, float, float] | None,
        estimate: dict[str, object] | None = None,
    ) -> int:
        self.ensure_capacity()
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO observations (
                    captured_at, image_jpeg, scan_json, run_id, mode,
                    pose_x_cm, pose_y_cm, heading_deg, estimate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    captured_at,
                    image_jpeg,
                    json.dumps(scan, separators=(",", ":")),
                    run_id,
                    mode,
                    pose[0] if pose else None,
                    pose[1] if pose else None,
                    pose[2] if pose else None,
                    json.dumps(estimate, separators=(",", ":")) if estimate is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    @property
    def path(self) -> Path:
        return self._path

    def ensure_capacity(self) -> None:
        if str(self._path) != ":memory:":
            free_bytes = shutil.disk_usage(self._path.parent).free
            if free_bytes < MINIMUM_FREE_BYTES:
                raise OSError(
                    f"Mapping paused: less than {MINIMUM_FREE_BYTES // (1024 * 1024)} MB "
                    "of free space remains"
                )

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM observations").fetchone()
        return int(row[0])

    def latest(self) -> tuple[int, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, captured_at FROM observations ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return (int(row[0]), str(row[1])) if row else None

    def latest_run_id(self) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id FROM observations WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return str(row[0]) if row else None

    def recent(self, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row(row) for row in reversed(rows)]

    def run_observations(self, run_id: str) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: Sequence[object]) -> dict[str, object]:
        return {
            "id": int(row[0]),
            "captured_at": row[1],
            "scan": with_current_angles(json.loads(row[2])),
            "run_id": row[3],
            "mode": row[4],
            "pose_x_cm": row[5],
            "pose_y_cm": row[6],
            "heading_deg": row[7],
            "estimate": json.loads(row[8]) if row[8] else None,
        }

    def image(self, observation_id: int) -> bytes | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT image_jpeg FROM observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        return bytes(row[0]) if row else None

    def place_observations(self) -> list[dict[str, object]]:
        """Every stop of every run, without images, oldest first."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations ORDER BY id
                """
            ).fetchall()
        return [self._row(row) for row in rows]

    def save_keyframe(self, keyframe: Keyframe, table: str = "keyframes") -> None:
        if table not in ("keyframes", "forward_keyframes"):
            raise ValueError(f"Unknown keyframe table {table}")
        with self._lock, self._connection:
            self._connection.execute(
                f"""
                INSERT OR REPLACE INTO {table} (
                    observation_id, run_id, pan_offset_deg, width, height, points, descriptors
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    keyframe.observation_id,
                    keyframe.run_id,
                    keyframe.pan_offset_deg,
                    keyframe.width,
                    keyframe.height,
                    keyframe.points.astype(np.float32).tobytes(),
                    keyframe.descriptors.astype(np.uint8).tobytes(),
                ),
            )

    def save_forward_keyframe(self, keyframe: Keyframe) -> None:
        self.save_keyframe(keyframe, "forward_keyframes")

    def forward_keyframes(self) -> list[Keyframe]:
        return self.keyframes("forward_keyframes")

    def keyframes(self, table: str = "keyframes") -> list[Keyframe]:
        if table not in ("keyframes", "forward_keyframes"):
            raise ValueError(f"Unknown keyframe table {table}")
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT observation_id, run_id, pan_offset_deg, width, height, points, descriptors
                FROM {table} ORDER BY observation_id
                """
            ).fetchall()
        return [
            Keyframe(
                int(row[0]),
                str(row[1]),
                float(row[2]),
                int(row[3]),
                int(row[4]),
                np.frombuffer(row[5], dtype=np.float32).reshape(-1, 2),
                np.frombuffer(row[6], dtype=np.uint8).reshape(-1, 32),
            )
            for row in rows
        ]

    def save_link(
        self,
        source: int,
        target: int,
        delta: tuple[float, float, float],
        sigma: Sequence[float],
        kind: str,
        inliers: int,
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO place_links (
                    source, target, dx_cm, dy_cm, dheading_deg, sigma_json, kind, inliers, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    target,
                    delta[0],
                    delta[1],
                    delta[2],
                    json.dumps([round(float(value), 3) for value in sigma]),
                    kind,
                    inliers,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def delete_link(self, link_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM place_links WHERE id = ?", (link_id,))

    def links(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, source, target, dx_cm, dy_cm, dheading_deg, sigma_json, kind, inliers
                FROM place_links ORDER BY id
                """
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "source": int(row[1]),
                "target": int(row[2]),
                "delta": (float(row[3]), float(row[4]), float(row[5])),
                "sigma": json.loads(row[6]),
                "kind": row[7],
                "inliers": int(row[8]),
            }
            for row in rows
        ]

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM observations")
            self._connection.execute("DELETE FROM keyframes")
            self._connection.execute("DELETE FROM forward_keyframes")
            self._connection.execute("DELETE FROM place_links")
            self._connection.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('observations', 'place_links')"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


