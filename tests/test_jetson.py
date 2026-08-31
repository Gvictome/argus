"""
Jetson Orin transition: platform detection, CSI capture, multi-camera.

None of this needs a Jetson to test -- the board is detected from files
and the capture path is a pipeline string, both of which can be faked.
"""

import pytest

from src.camera import platform_detect
from src.camera.platform_detect import Board, detect_board
from src.camera.backends import (
    JetsonCSIBackend,
    OpenCVBackend,
    Picamera2Backend,
    build_backend,
    build_csi_pipeline,
    rotation_to_flip_method,
)
from src.camera import CameraConfig, CameraRegistry, CameraService, build_registry_from_settings


class TestBoardDetection:
    """
    Regression: IS_RASPBERRY_PI was platform.machine().startswith('aarch'),
    which is TRUE on a Jetson Orin. The camera service would try picamera2
    on a Jetson, fail, and report a hardware error with a software cause.
    """

    def test_jetson_detected_from_tegra_release(self, tmp_path, monkeypatch):
        tegra = tmp_path / "nv_tegra_release"
        tegra.write_text("# R36 (release), REVISION: 3.0")
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tegra)
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", tmp_path / "absent")

        assert detect_board() is Board.JETSON

    def test_jetson_detected_from_device_tree(self, tmp_path, monkeypatch):
        model = tmp_path / "model"
        model.write_bytes(b"NVIDIA Jetson Orin Nano Developer Kit\x00")
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tmp_path / "absent")
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", model)

        assert detect_board() is Board.JETSON

    def test_raspberry_pi_still_detected(self, tmp_path, monkeypatch):
        model = tmp_path / "model"
        model.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\x00")
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tmp_path / "absent")
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", model)

        assert detect_board() is Board.RASPBERRY_PI

    def test_a_jetson_is_not_reported_as_a_pi(self, tmp_path, monkeypatch):
        """The actual bug, stated directly."""
        model = tmp_path / "model"
        model.write_bytes(b"NVIDIA Jetson Orin Nano\x00")
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tmp_path / "absent")
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", model)

        assert detect_board() is not Board.RASPBERRY_PI

    def test_laptop_is_generic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tmp_path / "absent")
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", tmp_path / "absent")

        assert detect_board() is Board.GENERIC

    def test_unreadable_device_tree_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(platform_detect, "_TEGRA_RELEASE", tmp_path / "absent")
        monkeypatch.setattr(platform_detect, "_DEVICE_TREE_MODEL", tmp_path)  # a dir

        assert detect_board() is Board.GENERIC


class TestBackendSelection:
    def test_jetson_gets_the_csi_backend(self):
        backend = build_backend(Board.JETSON, 1, CameraConfig())

        assert isinstance(backend, JetsonCSIBackend)
        assert backend.sensor_id == 1

    def test_pi_gets_picamera2(self):
        assert isinstance(build_backend(Board.RASPBERRY_PI, 0, CameraConfig()),
                          Picamera2Backend)

    def test_generic_gets_opencv(self):
        assert isinstance(build_backend(Board.GENERIC, 0, CameraConfig()), OpenCVBackend)


class TestCSIPipeline:
    """
    The pipeline string is the whole Jetson capture path. A typo here
    fails at runtime on hardware we cannot test against, so its parts are
    asserted directly.
    """

    def test_sensor_id_selects_the_connector(self):
        """Multi-camera on the Orin is exactly this parameter."""
        assert "sensor-id=0" in build_csi_pipeline(sensor_id=0)
        assert "sensor-id=1" in build_csi_pipeline(sensor_id=1)

    def test_uses_nvarguscamerasrc(self):
        """A plain V4L2 source cannot see a CSI camera on a Jetson."""
        assert build_csi_pipeline().startswith("nvarguscamerasrc")

    def test_keeps_frames_in_nvmm_and_converts_in_hardware(self):
        pipeline = build_csi_pipeline()

        assert "memory:NVMM" in pipeline
        assert "nvvidconv" in pipeline

    def test_outputs_bgr_for_opencv(self):
        assert "format=(string)BGR " in build_csi_pipeline() + " "

    def test_sink_drops_stale_frames(self):
        """
        Without drop=true a slow consumer builds an unbounded backlog and
        latency grows forever. A late security frame is worthless.
        """
        pipeline = build_csi_pipeline()

        assert "drop=true" in pipeline
        assert "max-buffers=1" in pipeline

    def test_resolution_and_framerate_are_applied(self):
        pipeline = build_csi_pipeline(capture_size=(1280, 720), framerate=60)

        assert "width=(int)1280" in pipeline
        assert "height=(int)720" in pipeline
        assert "framerate=(fraction)60/1" in pipeline

    @pytest.mark.parametrize("rotation,expected", [(0, 0), (90, 1), (180, 2), (270, 3)])
    def test_rotation_maps_to_flip_method(self, rotation, expected):
        assert rotation_to_flip_method(rotation) == expected

    def test_flips_map_to_mirror_methods(self):
        assert rotation_to_flip_method(0, hflip=True) == 4
        assert rotation_to_flip_method(0, vflip=True) == 2


class TestCameraRegistry:
    def test_registry_holds_multiple_cameras(self):
        reg = CameraRegistry()
        reg.add(CameraConfig(sensor_id=0, name="front"))
        reg.add(CameraConfig(sensor_id=1, name="back"))

        assert reg.names() == ["front", "back"]
        assert reg.get("back").config.sensor_id == 1

    def test_primary_is_the_lowest_sensor_id(self):
        """Not insertion order: configuration ordering must not decide."""
        reg = CameraRegistry()
        reg.add(CameraConfig(sensor_id=1, name="back"))
        reg.add(CameraConfig(sensor_id=0, name="front"))

        assert reg.primary().config.name == "front"

    def test_unknown_camera_is_none(self):
        assert CameraRegistry().get("nope") is None

    def test_shutdown_all_survives_a_failing_camera(self):
        """One wedged sensor must not strand the others open."""
        reg = CameraRegistry()
        good = reg.add(CameraConfig(sensor_id=0, name="good"))
        bad = reg.add(CameraConfig(sensor_id=1, name="bad"))

        def boom():
            raise RuntimeError("device is wedged")
        bad.shutdown = boom
        calls = []
        good.shutdown = lambda: calls.append(1)

        reg.shutdown_all()

        assert calls == [1]


class _Settings:
    CAMERA_SENSORS = ""
    CAMERA_INDEX = 0
    CAMERA_RESOLUTION = (1920, 1080)
    CAMERA_FPS = 30
    CAMERA_ROTATION = 0


class TestRegistryFromSettings:
    def test_parses_two_named_sensors(self):
        s = _Settings()
        s.CAMERA_SENSORS = "0:front,1:back"

        reg = build_registry_from_settings(s)

        assert reg.names() == ["front", "back"]
        assert [c.config.sensor_id for c in reg.all()] == [0, 1]

    def test_parses_bare_ids(self):
        s = _Settings()
        s.CAMERA_SENSORS = "0,1"

        assert build_registry_from_settings(s).names() == ["camera0", "camera1"]

    def test_empty_falls_back_to_a_single_camera(self):
        """Existing single-camera installs must not need reconfiguring."""
        reg = build_registry_from_settings(_Settings())

        assert len(reg.all()) == 1
        assert reg.primary().config.sensor_id == 0

    def test_malformed_entries_are_skipped_not_fatal(self):
        s = _Settings()
        s.CAMERA_SENSORS = "0:front,notanumber:x,1:back"

        assert build_registry_from_settings(s).names() == ["front", "back"]


class TestCameraState:
    def test_reports_stopped_before_initialize(self):
        assert CameraService().get_status()["status"] == "stopped"

    def test_status_carries_sensor_identity(self):
        cam = CameraService(CameraConfig(sensor_id=1, name="back"))
        status = cam.get_status()

        assert status["sensor_id"] == 1
        assert status["name"] == "back"
        assert "board" in status
