import unittest
from pathlib import Path
import time
from unittest.mock import patch

from robocar.config import Config
from robocar.web import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Config(mapping_database=Path(":memory:")))
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["robocar_mapping"].close()
        self.app.extensions["robocar_drive"].close()
        self.app.extensions["robocar_pan_tilt"].close()
        self.app.extensions["robocar_range_sensor"].close()
        self.app.extensions["robocar_camera"].close()

    def test_dashboard_and_status_are_available(self):
        dashboard = self.client.get("/").get_data(as_text=True)
        self.assertIn("Start camera processing", dashboard)
        self.assertNotIn('<img id="camera-feed" src=', dashboard)
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["drive"]["mode"], "simulation")
        self.assertFalse(status["drive"]["enabled"])
        self.assertEqual(status["drive"]["max_duty"], 4095)
        self.assertEqual(status["drive"]["watchdog_timeout_seconds"], 1.0)
        self.assertFalse(status["camera"]["enabled"])
        self.assertEqual(status["range"]["mode"], "disabled")
        self.assertIsNone(status["range"]["distance_cm"])
        self.assertEqual(status["mapping"]["observation_count"], 0)
        self.assertIn("Ultrasonic range", dashboard)
        self.assertIn("Mapping &amp; exploration", dashboard)
        self.assertIn("Start mapping &amp; auto explore", dashboard)
        self.assertEqual(dashboard.count('id="mapping-toggle"'), 1)
        self.assertIn('id="reset-mapping"', dashboard)
        self.assertIn('class="mapping-section"', dashboard)
        self.assertIn("min-height: 540px", dashboard)
        self.assertNotIn('id="map-history"', dashboard)
        self.assertNotIn('id="explore-toggle"', dashboard)
        self.assertNotIn("Each run is limited", dashboard)
        self.assertNotIn('id="explore-confirmation"', dashboard)

    def test_camera_processing_can_be_paused_and_resumed(self):
        response = self.client.post("/api/camera", json={"enabled": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"enabled": False})
        self.assertFalse(self.client.get("/api/status").get_json()["camera"]["enabled"])
        response = self.client.post("/api/camera", json={"enabled": True})
        self.assertEqual(response.get_json(), {"enabled": True})
        self.assertTrue(self.client.get("/api/status").get_json()["camera"]["enabled"])
        self.assertEqual(self.client.post("/api/camera", json={"enabled": "false"}).status_code, 400)

    def test_drive_endpoint_validates_payload_and_simulates(self):
        self.assertEqual(self.client.post("/api/drive", json={"throttle": "fast", "strafe": 0, "steering": 0}).status_code, 400)
        self.assertEqual(self.client.post("/api/drive", json={"throttle": float("nan"), "strafe": 0, "steering": 0}).status_code, 400)
        self.assertEqual(self.client.post("/api/drive", json={"throttle": 10**1000, "strafe": 0, "steering": 0}).status_code, 400)
        response = self.client.post("/api/drive", json={"throttle": 0.5, "strafe": -0.25, "steering": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"throttle": 0.5, "strafe": -0.25, "steering": 0.0})
        self.assertEqual(self.client.post("/api/stop").get_json(), {"stopped": True})

    def test_manual_outputs_are_locked_during_mapping_scans(self):
        drive = self.app.extensions["robocar_drive"]
        pan_tilt = self.app.extensions["robocar_pan_tilt"]
        self.assertTrue(drive.set_inhibited(True))
        pan_tilt.set_inhibited(True)
        try:
            self.assertEqual(
                self.client.post(
                    "/api/drive",
                    json={"throttle": 1, "strafe": 0, "steering": 0},
                ).status_code,
                409,
            )
            self.assertEqual(
                self.client.post("/api/pan-tilt", json={"pan": 45, "tilt": 90}).status_code,
                409,
            )
        finally:
            drive.set_inhibited(False)
            pan_tilt.set_inhibited(False)

    def test_pan_tilt_controls_and_validates_angles(self):
        self.assertEqual(self.client.get("/api/pan-tilt").get_json(), {
            "enabled": False,
            "mode": "simulation",
            "fault": None,
            "pan": 90,
            "tilt": 100,
        })
        response = self.client.post("/api/pan-tilt", json={"pan": 45, "tilt": 125})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"pan": 45, "tilt": 125})
        self.assertEqual(self.client.post("/api/pan-tilt", json={"pan": 151, "tilt": 90}).status_code, 400)

    def test_mapping_records_a_stationary_camera_observation(self):
        self.assertEqual(self.client.post("/api/mapping", json={"enabled": "yes"}).status_code, 400)
        camera = self.app.extensions["robocar_camera"]
        with patch.object(camera, "capture_snapshot", return_value=b"jpeg-data"):
            response = self.client.post("/api/mapping", json={"enabled": True})
            self.assertEqual(response.status_code, 200)
            deadline = time.monotonic() + 2
            while (
                self.client.get("/api/status").get_json()["mapping"]["observation_count"] == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            observations = self.client.get("/api/mapping/observations").get_json()["observations"]
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]["scan"], [])
            image = self.client.get(
                f"/api/mapping/observations/{observations[0]['id']}/image"
            )
            self.assertEqual(image.data, b"jpeg-data")
            self.assertEqual(self.client.post("/api/mapping", json={"enabled": False}).status_code, 200)

    def test_camera_matching_pictures_are_listed_and_served(self):
        self.assertEqual(self.client.get("/api/mapping/vision").get_json(), {"panels": []})
        self.assertEqual(self.client.get("/api/mapping/vision/stop.jpg").status_code, 404)

    def test_surface_map_endpoint_reports_contours_and_path(self):
        data = self.client.get("/api/mapping/surfaces").get_json()
        for key in ("surfaces", "features", "contours", "echoes", "path", "scan_count", "pose_sigma_cm"):
            self.assertIn(key, data)

    def test_autonomous_exploration_requires_explicit_attendance_and_hardware(self):
        self.assertEqual(
            self.client.post("/api/mapping/explore", json={"enabled": True}).status_code,
            400,
        )
        response = self.client.post(
            "/api/mapping/explore",
            json={"enabled": True, "attended": True},
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("hardware mode", response.get_json()["error"])

    def test_mapping_database_reset_requires_confirmation_and_clears_history(self):
        mapping = self.app.extensions["robocar_mapping"]
        store = mapping._store
        store.save(b"jpeg", [], "test-run", "survey", None)
        self.assertEqual(self.client.post("/api/mapping/reset", json={}).status_code, 400)

        response = self.client.post("/api/mapping/reset", json={"confirm": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["observation_count"], 0)
        self.assertEqual(self.client.get("/api/mapping/observations").get_json()["observations"], [])

    def test_mapping_database_cannot_be_reset_during_active_mapping(self):
        mapping = self.app.extensions["robocar_mapping"]
        mapping.start()
        try:
            response = self.client.post("/api/mapping/reset", json={"confirm": True})
            self.assertEqual(response.status_code, 409)
        finally:
            mapping.stop()

    def test_remote_interface_is_available_without_authentication(self):
        app = create_app(Config(host="0.0.0.0", mapping_database=Path(":memory:")))
        client = app.test_client()
        try:
            self.assertEqual(client.get("/api/status").status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)
        finally:
            app.extensions["robocar_mapping"].close()
            app.extensions["robocar_drive"].close()
            app.extensions["robocar_pan_tilt"].close()
            app.extensions["robocar_range_sensor"].close()
            app.extensions["robocar_camera"].close()



class MissingCarBoardTests(unittest.TestCase):
    """With the car's board switched off or unplugged the dashboard must still run."""

    def test_the_service_starts_and_reports_the_motor_board_unavailable(self):
        app = create_app(
            Config(mapping_database=Path(":memory:"), hardware_enabled=True, freenove_server_path=Path("/no/such/kit"))
        )
        try:
            client = app.test_client()
            status = client.get("/api/status").get_json()
            self.assertEqual(status["drive"]["mode"], "unavailable")
            self.assertIn("not found", status["drive"]["fault"])
            self.assertEqual(status["range"]["mode"], "unavailable")
            refused = client.post("/api/drive", json={"throttle": 1, "strafe": 0, "steering": 0})
            self.assertEqual(refused.status_code, 409)
            self.assertIn("Motor board unavailable", refused.get_json()["error"])
            self.assertEqual(client.post("/api/stop").status_code, 200)
            self.assertEqual(client.post("/api/pan-tilt", json={"pan": 60, "tilt": 100}).status_code, 409)
            explore = client.post("/api/mapping/explore", json={"enabled": True, "attended": True})
            self.assertEqual(explore.status_code, 409)
            self.assertIn("Motor board unavailable", explore.get_json()["error"])
        finally:
            for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera"):
                app.extensions[name].close()


if __name__ == "__main__":
    unittest.main()
