"""Interface 2: POST /api/data/update - Receive and validate table edits."""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from services.storage import (
    read_raw_file,
    read_session_data,
    save_session_data,
    append_changes_log,
    generate_session_id
)
from services.validation import validate_changes_batch

data_update_bp = Blueprint('data_update', __name__)
logger = logging.getLogger('ecometrics')


def apply_changes(df, changes: list):
    """
    Apply all validated changes to DataFrame.

    Args:
        df: pandas DataFrame to modify
        changes: List of change objects

    Returns:
        Modified DataFrame
    """
    for change in changes:
        if change['changeType'] == 'column_rename':
            df = df.rename(columns={
                change['oldName']: change['newName']
            })
        elif change['changeType'] == 'cell_edit':
            df.loc[change['rowIndex'], change['column']] = change['newValue']

    return df


@data_update_bp.route('/api/data/update', methods=['POST'])
def update_data():
    """
    Receive and validate table edits from frontend, save to session.

    Request Body:
        sessionId (optional): Session identifier (auto-generated if not provided)
        dataType (required): "fleetfuel" or "efid"
        changes (required): List of change objects

    Returns:
        JSON with success, sessionId, changesApplied, validation, storage
    """
    data = request.get_json()

    if not data:
        logger.error("[no-session] EDIT - FAILED - No JSON body provided")
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Request body must be JSON"
            }
        }), 400

    # Extract and validate required fields
    session_id = data.get('sessionId') or generate_session_id()
    data_type = data.get('dataType')
    changes = data.get('changes')

    if not data_type:
        logger.error(f"[{session_id}] EDIT - FAILED - Missing 'dataType'")
        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_PARAMETER",
                "message": "'dataType' is required"
            }
        }), 400

    if data_type not in ('fleetfuel', 'efid'):
        logger.error(f"[{session_id}] EDIT - FAILED - Invalid dataType: {data_type}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_PARAMETER",
                "message": f"Invalid dataType '{data_type}'. Must be 'fleetfuel' or 'efid'"
            }
        }), 400

    if not changes:
        logger.error(f"[{session_id}] EDIT - FAILED - Missing 'changes'")
        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_PARAMETER",
                "message": "'changes' is required"
            }
        }), 400

    # Log received changes
    for change in changes:
        change_type = change.get('changeType', 'unknown')
        if change_type == 'column_rename':
            logger.info(f"[{session_id}] EDIT - RECEIVED - Column rename: {change.get('oldName')} -> {change.get('newName')}")
        elif change_type == 'cell_edit':
            logger.info(f"[{session_id}] EDIT - RECEIVED - Cell edit: row {change.get('rowIndex')}, column {change.get('column')}")
        else:
            logger.info(f"[{session_id}] EDIT - RECEIVED - {change_type}")

    try:
        # Load existing session data or raw data
        df = read_session_data(session_id, data_type)
        if df is None:
            df = read_raw_file(data_type)

        # Validate all changes
        is_valid, errors = validate_changes_batch(changes, df.columns.tolist())

        if not is_valid:
            error_msg = "; ".join(errors)
            logger.error(f"[{session_id}] VALIDATE - FAILED - {error_msg}")
            return jsonify({
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": error_msg
                }
            }), 400

        logger.info(f"[{session_id}] VALIDATE - SUCCESS - {len(changes)} change(s) validated")

        # Apply changes
        df = apply_changes(df, changes)

        # Save to session
        file_path = save_session_data(session_id, data_type, df)

        # Log changes
        timestamped_changes = [
            {**change, "timestamp": datetime.now(timezone.utc).isoformat()}
            for change in changes
        ]
        append_changes_log(session_id, timestamped_changes)

        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"[{session_id}] SAVE - SUCCESS - Data saved to {file_path}")

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "changesApplied": len(changes),
            "validation": {"status": "passed"},
            "storage": {
                "status": "committed",
                "filePath": file_path,
                "timestamp": timestamp
            }
        })

    except FileNotFoundError as e:
        logger.error(f"[{session_id}] EDIT - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "FILE_NOT_FOUND",
                "message": str(e)
            }
        }), 404

    except Exception as e:
        logger.error(f"[{session_id}] EDIT - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        }), 500
