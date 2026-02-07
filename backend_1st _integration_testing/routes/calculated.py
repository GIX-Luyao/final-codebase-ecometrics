"""Interface 3: GET /api/calculated-data - Return calculated emissions data."""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from services.calculation import (
    run_calculation_pipeline,
    prepare_chart_data,
    add_formula_references
)

calculated_bp = Blueprint('calculated', __name__)
logger = logging.getLogger('ecometrics')


@calculated_bp.route('/api/calculated-data', methods=['GET'])
def get_calculated_data():
    """
    Return calculated emissions data in chart-ready format.

    Query Parameters:
        sessionId (required): Session identifier

    Returns:
        JSON with success, sessionId, calculationTimestamp, summary, charts, rawData
    """
    session_id = request.args.get('sessionId')

    if not session_id:
        logger.error("[no-session] CALC - FAILED - Missing 'sessionId' parameter")
        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_PARAMETER",
                "message": "Query parameter 'sessionId' is required"
            }
        }), 400

    try:
        logger.info(f"[{session_id}] CALC - START - Running calculation pipeline")

        # Run the calculation pipeline
        df_calculated = run_calculation_pipeline(session_id)

        # Prepare chart data
        charts = prepare_chart_data(df_calculated)

        # Add formula references to raw data
        raw_data = add_formula_references(df_calculated)

        # Calculate summary
        total_mtco2e = df_calculated['mtCO2e_calc'].sum()
        record_count = len(df_calculated)

        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"[{session_id}] CALC - COMPLETE - {record_count} records, total mtCO2e: {total_mtco2e:.5f}")

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "calculationTimestamp": timestamp,
            "summary": {
                "totalMtCO2e": round(total_mtco2e, 5),
                "recordCount": record_count
            },
            "charts": charts,
            "rawData": raw_data
        })

    except FileNotFoundError as e:
        logger.error(f"[{session_id}] CALC - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "FILE_NOT_FOUND",
                "message": str(e)
            }
        }), 404

    except Exception as e:
        logger.error(f"[{session_id}] CALC - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        }), 500
