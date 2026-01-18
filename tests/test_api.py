"""
API endpoint tests
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns online status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["service"] == "THE EYE"

    def test_status_endpoint(self, client):
        """Test status endpoint returns system info"""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "services" in data
        assert "version" in data


class TestCameraEndpoints:
    """Test camera-related endpoints"""

    def test_camera_status(self, client):
        """Test camera status endpoint"""
        response = client.get("/api/camera/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "resolution" in data

    def test_camera_snapshot_not_initialized(self, client):
        """Test snapshot returns 503 when camera not initialized"""
        response = client.get("/api/camera/snapshot")
        assert response.status_code == 503


class TestDetectionEndpoints:
    """Test detection-related endpoints"""

    def test_detection_status(self, client):
        """Test detection status endpoint"""
        response = client.get("/api/detection/status")
        assert response.status_code == 200
        data = response.json()
        assert "motion_detection" in data
        assert "face_recognition" in data

    def test_list_faces(self, client):
        """Test listing faces returns empty list initially"""
        response = client.get("/api/faces")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0


class TestAutomationEndpoints:
    """Test automation-related endpoints"""

    def test_list_devices(self, client):
        """Test listing devices returns empty list initially"""
        response = client.get("/api/devices")
        assert response.status_code == 200
        assert response.json() == []

    def test_device_not_found(self, client):
        """Test getting non-existent device returns 404"""
        response = client.get("/api/devices/nonexistent")
        assert response.status_code == 404

    def test_list_automations(self, client):
        """Test listing automations returns empty list initially"""
        response = client.get("/api/automations")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0


class TestSecurityEndpoints:
    """Test security-related endpoints"""

    def test_logout(self, client):
        """Test logout endpoint"""
        response = client.post("/api/auth/logout")
        assert response.status_code == 200

    def test_audit_logs(self, client):
        """Test getting audit logs"""
        response = client.get("/api/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
