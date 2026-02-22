"""
Configuration endpoints for column mappings.
"""
from datetime import datetime
from flask import Blueprint, request, jsonify

from services.storage import StorageService
from services.validation import ValidationService


config_bp = Blueprint("config", __name__)


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


@config_bp.route("/sessions/<session_id>/sources/<source_id>/config", methods=["POST"])
def save_config(session_id: str, source_id: str):
    """Save column mapping configuration for a data source."""
    data = request.get_json()

    if not data:
        return make_error_response(
            "MISSING_PARAMETER",
            "Request body is required"
        )

    # Check session exists
    session = StorageService.get_session(session_id)
    if session is None:
        return make_error_response(
            "SESSION_NOT_FOUND",
            f"Session not found: {session_id}",
            status_code=404
        )

    # Validate configuration
    validation_result = ValidationService.validate_config(data, source_id)

    if not validation_result["valid"]:
        return make_error_response(
            "VALIDATION_ERROR",
            "Configuration validation failed",
            details={"errors": validation_result["errors"]},
            status_code=400
        )

    try:
        # Save configuration
        config_version = StorageService.save_source_config(session_id, source_id, data)

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "sourceId": source_id,
            "configVersion": config_version,
            "validation": {
                "status": "valid" if not validation_result["warnings"] else "valid_with_warnings",
                "warnings": validation_result["warnings"],
                "errors": [],
            },
            "savedAt": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)


@config_bp.route("/sessions/<session_id>/sources/<source_id>/config", methods=["GET"])
def get_config(session_id: str, source_id: str):
    """Retrieve saved configuration for a source."""
    # Check session exists
    session = StorageService.get_session(session_id)
    if session is None:
        return make_error_response(
            "SESSION_NOT_FOUND",
            f"Session not found: {session_id}",
            status_code=404
        )

    try:
        config = StorageService.get_source_config(session_id, source_id)

        if config is None:
            return make_error_response(
                "CONFIG_NOT_FOUND",
                f"No configuration found for source: {source_id}",
                status_code=404
            )

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "sourceId": source_id,
            "config": {
                "columnMappings": config.get("columnMappings"),
                "filters": config.get("filters", []),
                "efidLookup": config.get("efidLookup", {}),
                "sourceSheet": config.get("sourceSheet"),
            },
            "configVersion": config.get("configVersion", 1),
            "lastModified": config.get("lastModified"),
        })

    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)
