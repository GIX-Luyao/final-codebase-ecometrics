"""
Calculation endpoints.
"""
from datetime import datetime
from flask import Blueprint, request, jsonify

from config import Config
from services.storage import StorageService
from services.validation import ValidationService
from services.calculation_engine import CalculationEngine


calculate_bp = Blueprint("calculate", __name__)


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


@calculate_bp.route("/sessions/<session_id>/calculate", methods=["POST"])
def calculate(session_id: str):
    """Execute emissions calculations for configured sources."""
    data = request.get_json() or {}

    # Check session exists
    session = StorageService.get_session(session_id)
    if session is None:
        return make_error_response(
            "SESSION_NOT_FOUND",
            f"Session not found: {session_id}",
            status_code=404
        )

    # Get sources to calculate
    sources = data.get("sources", [])
    if not sources:
        # Calculate all configured sources
        sources = [s["sourceId"] for s in session.get("sources", [])]

    if not sources:
        return make_error_response(
            "MISSING_PARAMETER",
            "No sources specified and no configured sources found"
        )

    # Validate calculation request
    validation_result = ValidationService.validate_calculation_request(session_id, sources)
    if not validation_result["valid"]:
        return make_error_response(
            "VALIDATION_ERROR",
            "Calculation validation failed",
            details={"errors": validation_result["errors"]},
            status_code=400
        )

    gwp_version = data.get("gwpVersion", Config.DEFAULT_GWP_VERSION)

    try:
        # Run calculations
        engine = CalculationEngine()
        result = engine.calculate_for_session(session_id, sources, gwp_version)

        return jsonify({
            "success": True,
            "sessionId": session_id,
            **result
        })

    except ValueError as e:
        error_msg = str(e)
        error_code = "CALCULATION_ERROR"

        # Determine more specific error code
        if "Missing emission factors" in error_msg:
            error_code = "MISSING_EMISSION_FACTOR"
        elif "Missing GWP" in error_msg:
            error_code = "MISSING_GWP"

        return make_error_response(
            error_code,
            error_msg,
            status_code=422
        )

    except FileNotFoundError as e:
        return make_error_response(
            "SOURCE_NOT_FOUND",
            str(e),
            status_code=404
        )

    except Exception as e:
        return make_error_response(
            "CALCULATION_ERROR",
            str(e),
            status_code=500
        )
