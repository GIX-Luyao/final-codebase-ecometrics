"""Interface 1: GET /api/raw-data - Return raw data files for frontend preview."""

import logging

from flask import Blueprint, request, jsonify

from services.storage import read_raw_file, get_raw_file_name

raw_data_bp = Blueprint('raw_data', __name__)
logger = logging.getLogger('ecometrics')


def infer_column_types(df) -> dict:
    """Map pandas dtypes to simple types for frontend."""
    type_map = {
        'int64': 'integer',
        'int32': 'integer',
        'float64': 'float',
        'float32': 'float',
        'object': 'string',
        'datetime64[ns]': 'date',
        'bool': 'boolean'
    }
    return {col: type_map.get(str(df[col].dtype), 'string') for col in df.columns}


@raw_data_bp.route('/api/raw-data', methods=['GET'])
def get_raw_data():
    """
    Return raw data files for frontend preview.

    Query Parameters:
        type (required): "fleetfuel" or "efid"
        invalid (optional): "true" to get invalid test file for error testing

    Returns:
        JSON with success, dataType, fileName, rowCount, columns, columnTypes, preview
    """
    data_type = request.args.get('type')
    invalid = request.args.get('invalid', '').lower() == 'true'

    # Validate required parameter
    if not data_type:
        logger.error("[no-session] LOAD - FAILED - Missing 'type' parameter")
        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_PARAMETER",
                "message": "Query parameter 'type' is required"
            }
        }), 400

    # Validate data type
    if data_type not in ('fleetfuel', 'efid'):
        logger.error(f"[no-session] LOAD - FAILED - Invalid type: {data_type}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_PARAMETER",
                "message": f"Invalid type '{data_type}'. Must be 'fleetfuel' or 'efid'"
            }
        }), 400

    try:
        # Read the raw file
        df = read_raw_file(data_type, invalid=invalid)
        file_name = get_raw_file_name(data_type, invalid=invalid)

        logger.info(f"[no-session] LOAD - SUCCESS - Loaded {file_name} ({len(df)} rows)")

        return jsonify({
            "success": True,
            "dataType": data_type,
            "fileName": file_name,
            "rowCount": len(df),
            "columns": df.columns.tolist(),
            "columnTypes": infer_column_types(df),
            "preview": df.to_dict(orient='records')
        })

    except FileNotFoundError as e:
        logger.error(f"[no-session] LOAD - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "FILE_NOT_FOUND",
                "message": str(e)
            }
        }), 404

    except Exception as e:
        logger.error(f"[no-session] LOAD - FAILED - {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        }), 500
