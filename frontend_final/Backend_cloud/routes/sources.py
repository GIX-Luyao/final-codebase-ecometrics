"""
Source management endpoints.
"""
import io
from datetime import datetime
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np

from config import Config
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


def _get_filename(source_path) -> str:
    """Extract just the filename from either a Path object or an S3 key string."""
    if source_path is None:
        return ""
    # Path objects have a .name attribute; S3 keys are strings like "CONSUMPTION/file.xlsx"
    if hasattr(source_path, "name"):
        return source_path.name
    return str(source_path).split("/")[-1]


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
        # Get source file path / key
        source_path = StorageService.get_source_path(source_id)
        if source_path is None:
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
            "fileName": _get_filename(source_path),
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


@sources_bp.route("/files/upload", methods=["POST"])
def upload_file():
    """
    Upload a raw data file to storage (S3 in cloud mode, local dir in local mode).

    Body: multipart/form-data
      - file:   the file bytes
      - folder: "CONSUMPTION" | "REFERENCE"

    Returns: { success, sourceId, name, fileName, folder, sheets }
    """
    if "file" not in request.files:
        return make_error_response("MISSING_PARAMETER", "No file field in request", status_code=400)

    file = request.files["file"]
    if not file.filename:
        return make_error_response("MISSING_PARAMETER", "Empty filename", status_code=400)

    folder = request.form.get("folder", "CONSUMPTION").upper()
    if folder not in ("CONSUMPTION", "REFERENCE"):
        return make_error_response(
            "INVALID_PARAMETER",
            "folder must be 'CONSUMPTION' or 'REFERENCE'",
            status_code=400
        )

    filename = file.filename
    if "." not in filename:
        return make_error_response("INVALID_FILE_TYPE", "File has no extension", status_code=400)

    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in Config.SUPPORTED_EXTENSIONS:
        return make_error_response(
            "INVALID_FILE_TYPE",
            f"Unsupported file type '{ext}'. Supported: {Config.SUPPORTED_EXTENSIONS}",
            status_code=400
        )

    stem = filename.rsplit(".", 1)[0]
    source_id = stem.lower().replace(" ", "_").replace("-", "_")
    if folder == "REFERENCE":
        source_id = "ref_" + source_id

    file_bytes = file.read()

    try:
        if Config.is_s3_mode():
            import boto3
            s3 = boto3.client("s3", region_name=Config.AWS_REGION)
            s3_key = f"{folder}/{filename}"
            s3.put_object(
                Bucket=Config.S3_RAW_BUCKET,
                Key=s3_key,
                Body=file_bytes,
            )
        else:
            # Local mode: write to appropriate local directory
            if folder == "CONSUMPTION":
                dest = Config.CONSUMPTION_DIR / filename
            else:
                dest = Config.REFERENCE_DIR / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file_bytes)

        # Detect sheet names for Excel files
        sheets = None
        if ext in [".xlsx", ".xls"]:
            try:
                xl = pd.ExcelFile(io.BytesIO(file_bytes))
                sheets = xl.sheet_names
            except Exception:
                sheets = []

        return jsonify({
            "success": True,
            "sourceId": source_id,
            "name": stem,
            "fileName": filename,
            "folder": folder,
            "sheets": sheets,
        }), 201

    except Exception as e:
        return make_error_response("STORAGE_ERROR", f"Upload failed: {str(e)}", status_code=500)
