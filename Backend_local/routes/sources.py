"""
Source management endpoints.
"""
from datetime import datetime
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np

from services.storage import StorageService


sources_bp = Blueprint("sources", __name__)


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


@sources_bp.route("/sources", methods=["GET"])
def list_sources():
    """List all available source files (consumption and reference)."""
    try:
        consumption = StorageService.list_source_files()
        reference = StorageService.list_reference_files()
        return jsonify({
            "success": True,
            "sources": consumption,
            "reference": reference
        })
    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)


@sources_bp.route("/sources/<source_id>/preview", methods=["GET"])
def preview_source(source_id: str):
    """Preview raw data structure before processing."""
    sheet = request.args.get("sheet")

    try:
        # Get source file path
        file_path = StorageService.get_source_path(source_id)
        if file_path is None:
            return make_error_response(
                "SOURCE_NOT_FOUND",
                f"Source not found: {source_id}",
                status_code=404
            )

        # Load data
        df = StorageService.load_raw_data(source_id, sheet)

        # Build column info
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "nullCount": int(df[col].isna().sum()),
                "uniqueCount": int(df[col].nunique()),
            }

            # Determine type and add type-specific stats
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_info["type"] = "float" if pd.api.types.is_float_dtype(dtype) else "integer"
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_info["min"] = float(non_null.min())
                    col_info["max"] = float(non_null.max())
                    col_info["mean"] = round(float(non_null.mean()), 2)
                col_info["sampleValues"] = [
                    float(v) if pd.notna(v) else None
                    for v in df[col].head(3).tolist()
                ]
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_info["type"] = "datetime"
                col_info["sampleValues"] = [
                    str(v) if pd.notna(v) else None
                    for v in df[col].head(3).tolist()
                ]
            else:
                col_info["type"] = "string"
                col_info["sampleValues"] = [
                    str(v) if pd.notna(v) else None
                    for v in df[col].head(3).tolist()
                ]

            columns.append(col_info)

        # Build preview (first 10 rows)
        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        # Convert any numpy types to native Python types
        for row in preview:
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating)):
                    row[key] = value.item()
                elif isinstance(value, pd.Timestamp):
                    row[key] = value.isoformat()

        return jsonify({
            "success": True,
            "sourceId": source_id,
            "fileName": file_path.name,
            "sheet": sheet,
            "totalRows": len(df),
            "columns": columns,
            "preview": preview,
        })

    except FileNotFoundError:
        return make_error_response(
            "SOURCE_NOT_FOUND",
            f"Source not found: {source_id}",
            status_code=404
        )
    except Exception as e:
        return make_error_response("STORAGE_ERROR", str(e), status_code=500)
