"""
API endpoint tests for EcoMetrics Backend.
"""
import pytest
import json
import os
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from config import Config


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def cleanup_sessions():
    """Clean up test sessions after tests."""
    yield
    # Cleanup code runs after tests
    sessions_dir = Config.SESSIONS_DIR
    calculated_dir = Config.CALCULATED_DIR

    # Remove test sessions
    for item in sessions_dir.glob("sess_*"):
        if item.is_dir():
            shutil.rmtree(item)

    for item in calculated_dir.glob("sess_*"):
        if item.is_dir():
            shutil.rmtree(item)

    # Reset registry
    registry_file = Config.SESSION_REGISTRY
    if registry_file.exists():
        with open(registry_file, "w") as f:
            json.dump({"sessions": []}, f)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct response."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "ecometrics-backend"
        assert "version" in data


class TestSourcesEndpoint:
    """Tests for sources endpoints."""

    def test_list_sources(self, client):
        """Test listing source files."""
        response = client.get("/api/sources")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_preview_source_not_found(self, client):
        """Test preview for non-existent source."""
        response = client.get("/api/sources/nonexistent_source/preview")

        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False
        assert data["error"]["code"] == "SOURCE_NOT_FOUND"


class TestSessionsEndpoint:
    """Tests for session management endpoints."""

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/sessions")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "sessions" in data

    def test_create_session(self, client, cleanup_sessions):
        """Test creating a new session."""
        response = client.post(
            "/api/sessions",
            json={
                "name": "Test Session",
                "description": "A test session"
            }
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["success"] is True
        assert "sessionId" in data
        assert data["name"] == "Test Session"
        assert data["sessionId"].startswith("sess_")

    def test_create_session_missing_name(self, client):
        """Test creating session without name fails."""
        response = client.post("/api/sessions", json={})

        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert data["error"]["code"] == "MISSING_PARAMETER"

    def test_get_session(self, client, cleanup_sessions):
        """Test getting session details."""
        # First create a session
        create_response = client.post(
            "/api/sessions",
            json={"name": "Test Session"}
        )
        session_id = create_response.get_json()["sessionId"]

        # Then get it
        response = client.get(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["session"]["sessionId"] == session_id

    def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = client.get("/api/sessions/sess_nonexistent")

        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False
        assert data["error"]["code"] == "SESSION_NOT_FOUND"

    def test_delete_session(self, client, cleanup_sessions):
        """Test deleting a session."""
        # First create a session
        create_response = client.post(
            "/api/sessions",
            json={"name": "Test Session to Delete"}
        )
        session_id = create_response.get_json()["sessionId"]

        # Then delete it
        response = client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["sessionId"] == session_id

        # Verify it's gone
        get_response = client.get(f"/api/sessions/{session_id}")
        assert get_response.status_code == 404


class TestConfigEndpoint:
    """Tests for configuration endpoints."""

    def test_save_config_session_not_found(self, client):
        """Test saving config to non-existent session."""
        response = client.post(
            "/api/sessions/sess_nonexistent/sources/test_source/config",
            json={
                "columnMappings": {
                    "ACCT_ID": {"sourceColumn": "EQ_EQUIP_NO"}
                }
            }
        )

        assert response.status_code == 404
        data = response.get_json()
        assert data["error"]["code"] == "SESSION_NOT_FOUND"

    def test_get_config_not_found(self, client, cleanup_sessions):
        """Test getting non-existent config."""
        # Create a session first
        create_response = client.post(
            "/api/sessions",
            json={"name": "Test Session"}
        )
        session_id = create_response.get_json()["sessionId"]

        # Try to get config that doesn't exist
        response = client.get(
            f"/api/sessions/{session_id}/sources/nonexistent/config"
        )

        assert response.status_code == 404
        data = response.get_json()
        assert data["error"]["code"] == "CONFIG_NOT_FOUND"


class TestCalculateEndpoint:
    """Tests for calculation endpoints."""

    def test_calculate_session_not_found(self, client):
        """Test calculating for non-existent session."""
        response = client.post(
            "/api/sessions/sess_nonexistent/calculate",
            json={"sources": ["test_source"]}
        )

        assert response.status_code == 404
        data = response.get_json()
        assert data["error"]["code"] == "SESSION_NOT_FOUND"

    def test_calculate_no_sources(self, client, cleanup_sessions):
        """Test calculating with no configured sources."""
        # Create a session
        create_response = client.post(
            "/api/sessions",
            json={"name": "Test Session"}
        )
        session_id = create_response.get_json()["sessionId"]

        # Try to calculate without any sources
        response = client.post(
            f"/api/sessions/{session_id}/calculate",
            json={}
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False


class TestAnalyticsEndpoint:
    """Tests for analytics endpoints."""

    def test_summary_session_not_found(self, client):
        """Test getting summary for non-existent session."""
        response = client.get("/api/sessions/sess_nonexistent/analytics/summary")

        assert response.status_code == 404
        data = response.get_json()
        assert data["error"]["code"] == "SESSION_NOT_FOUND"

    def test_chart_data_session_not_found(self, client):
        """Test getting chart data for non-existent session."""
        response = client.post(
            "/api/sessions/sess_nonexistent/analytics/chart-data",
            json={
                "chartType": "pie",
                "groupBy": "Subtype"
            }
        )

        assert response.status_code == 404
        data = response.get_json()
        assert data["error"]["code"] == "SESSION_NOT_FOUND"


class TestErrorResponses:
    """Tests for error response format."""

    def test_error_response_format(self, client):
        """Test that error responses follow the standard format."""
        response = client.get("/api/sessions/sess_nonexistent")

        assert response.status_code == 404
        data = response.get_json()

        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "timestamp" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
