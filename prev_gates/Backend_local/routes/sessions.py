"""
Session management endpoints.
"""
from datetime import datetime
from flask import Blueprint, request, jsonify

from services.storage import StorageService


sessions_bp = Blueprint("sessions", __name__)


def make_error_response(code: str, message: str, details: dict = None, status_code: int = 400):
    """Create a standardized error response."""
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }
    if details:
        response["error"]["details"] = details
    return jsonify(response), status_code


@sessions_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all sessions."""
    try:
        registry = StorageService.load_session_registry()
        sessions = []

        for sess_info in registry.get("sessions", []):
            # Get full session metadata
            session = StorageService.get_session(sess_info["sessionId"])
            if session:
                # Calculate total emissions if available
                total_mtco2e = 0
                if session.get("calculations"):
                    latest_calc = session["calculations"][-1]
                    total_mtco2e = latest_calc.get("totalMtCO2e", 0)

                sessions.append({
                    "sessionId": session["sessionId"],
                    "name": session["name"],
                    "createdAt": session["createdAt"],
                    "lastModified": session["lastModified"],
                    "status": session["status"],
                    "configuredSources": len(session.get("sources", [])),
                    "hasCalculations": len(session.get("calculations", [])) > 0,
                    "totalMtCO2e": total_mtco2e,
                })

        return jsonify({
            "success": True,
            "sessions": sessions
        })

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)


@sessions_bp.route("/sessions", methods=["POST"])
def create_session():
    """Create a new processing session."""
    data = request.get_json()

    if not data:
        return make_error_response(
            "MISSING_PARAMETER",
            "Request body is required"
        )

    name = data.get("name")
    if not name:
        return make_error_response(
            "MISSING_PARAMETER",
            "Session name is required"
        )

    description = data.get("description", "")

    try:
        session = StorageService.create_session(name, description)

        return jsonify({
            "success": True,
            "sessionId": session["sessionId"],
            "name": session["name"],
            "createdAt": session["createdAt"],
        }), 201

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)


@sessions_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session details."""
    try:
        session = StorageService.get_session(session_id)

        if session is None:
            return make_error_response(
                "SESSION_NOT_FOUND",
                f"Session not found: {session_id}",
                status_code=404
            )

        return jsonify({
            "success": True,
            "session": session
        })

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)


@sessions_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session and all associated data."""
    try:
        session = StorageService.get_session(session_id)

        if session is None:
            return make_error_response(
                "SESSION_NOT_FOUND",
                f"Session not found: {session_id}",
                status_code=404
            )

        deleted_files = StorageService.delete_session(session_id)

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "deletedFiles": deleted_files,
        })

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)
