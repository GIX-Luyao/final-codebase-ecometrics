"""
S3-backed storage operations for AWS cloud deployment.
All methods mirror the interface of StorageService in storage.py so that
the rest of the codebase (routes, engines) requires zero changes.
"""
import io
import json
import secrets
from datetime import datetime
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from config import Config


def _s3_client():
    """Return a boto3 S3 client. Uses ECS task IAM role automatically."""
    return boto3.client("s3", region_name=Config.AWS_REGION)


def _get_object_json(bucket: str, key: str) -> Optional[dict]:
    """Read a JSON object from S3. Returns None if the key does not exist."""
    try:
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _put_object_json(bucket: str, key: str, data: dict):
    """Write a dict as JSON to S3."""
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _load_df_from_s3(bucket: str, key: str, sheet_name=None) -> pd.DataFrame:
    """Stream a CSV or Excel file from S3 into a pandas DataFrame."""
    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    if key.endswith(".csv"):
        return pd.read_csv(io.BytesIO(body))
    elif key.endswith(".xlsx") or key.endswith(".xls"):
        return pd.read_excel(io.BytesIO(body), sheet_name=sheet_name if sheet_name else 0)
    else:
        raise ValueError(f"Unsupported file type for S3 key: {key}")


def _get_sheet_names_from_bytes(body: bytes, key: str) -> list:
    """Extract Excel sheet names from raw bytes."""
    try:
        xl = pd.ExcelFile(io.BytesIO(body))
        return xl.sheet_names
    except Exception:
        return []


def _list_files_in_prefix(bucket: str, prefix: str, source_id_prefix: str = "") -> list[dict]:
    """List all supported files under an S3 prefix. Returns list of source dicts."""
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    sources = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]

            # Skip folder markers and temp Excel files
            if not filename or filename.startswith("~$"):
                continue

            # Determine extension
            if "." not in filename:
                continue
            ext = "." + filename.rsplit(".", 1)[-1].lower()
            if ext not in Config.SUPPORTED_EXTENSIONS:
                continue

            stem = filename.rsplit(".", 1)[0]
            raw_source_id = stem.lower().replace(" ", "_").replace("-", "_")
            source_id = source_id_prefix + raw_source_id

            source_info = {
                "sourceId": source_id,
                "name": stem,
                "filePath": f"s3://{bucket}/{key}",
                "s3Key": key,
                "fileSize": obj["Size"],
                "lastModified": obj["LastModified"].isoformat(),
            }

            # Get sheet names for Excel files
            if ext in [".xlsx", ".xls"]:
                try:
                    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    source_info["sheets"] = _get_sheet_names_from_bytes(body, key)
                except Exception:
                    source_info["sheets"] = []
            else:
                source_info["sheets"] = None

            sources.append(source_info)

    return sources


def _find_s3_key(source_id: str, prefix: str, strip_prefix: str = "") -> Optional[str]:
    """Find the S3 key whose filename stem matches source_id."""
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    # Strip any leading prefix from source_id before matching
    match_id = source_id[len(strip_prefix):] if strip_prefix and source_id.startswith(strip_prefix) else source_id

    for page in paginator.paginate(Bucket=Config.S3_RAW_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            if "." not in filename:
                continue
            stem = filename.rsplit(".", 1)[0]
            file_source_id = stem.lower().replace(" ", "_").replace("-", "_")
            if file_source_id == match_id:
                return key

    return None


class S3StorageService:
    """
    S3-backed storage service. Same interface as StorageService in storage.py.
    Used automatically when Config.is_s3_mode() is True.
    """

    # --- ID Generation ---

    @staticmethod
    def generate_session_id() -> str:
        return f"sess_{secrets.token_hex(3)}"

    @staticmethod
    def generate_calculation_id(session_id: str) -> str:
        return f"calc_{session_id.replace('sess_', '')}_{secrets.token_hex(3)}"

    # --- Source File Operations ---

    @staticmethod
    def list_source_files() -> list[dict]:
        """List all consumption files in S3 RAW_BUCKET/CONSUMPTION/."""
        return _list_files_in_prefix(
            Config.S3_RAW_BUCKET,
            Config.S3_CONSUMPTION_PREFIX,
            source_id_prefix=""
        )

    @staticmethod
    def list_reference_files() -> list[dict]:
        """List all reference files in S3 RAW_BUCKET/REFERENCE/."""
        return _list_files_in_prefix(
            Config.S3_RAW_BUCKET,
            Config.S3_REFERENCE_PREFIX,
            source_id_prefix="ref_"
        )

    @staticmethod
    def get_source_path(source_id: str) -> Optional[str]:
        """Return the S3 key string for a given source_id (or None if not found)."""
        if source_id.startswith("ref_"):
            key = _find_s3_key(source_id, Config.S3_REFERENCE_PREFIX, strip_prefix="ref_")
        else:
            key = _find_s3_key(source_id, Config.S3_CONSUMPTION_PREFIX)
        return key  # callers only use this for existence checks + file name

    @staticmethod
    def load_raw_data(source_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load a source file from S3 into a DataFrame."""
        if source_id.startswith("ref_"):
            key = _find_s3_key(source_id, Config.S3_REFERENCE_PREFIX, strip_prefix="ref_")
        else:
            key = _find_s3_key(source_id, Config.S3_CONSUMPTION_PREFIX)

        if key is None:
            raise FileNotFoundError(f"Source not found in S3: {source_id}")

        return _load_df_from_s3(Config.S3_RAW_BUCKET, key, sheet_name)

    # --- Reference Table Operations ---

    @staticmethod
    def load_reference_table(table_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load a named reference table from S3 REFERENCE/ prefix.

        Maps standard names (EFID, GWP, LOB) to expected filenames.
        Users must have uploaded the reference files to S3 REFERENCE/ folder.
        """
        name_to_stem = {
            "EFID": "EFID",
            "GWP": "GWPs",
            "LOB": "LOB_LowOrgList",
        }

        if table_name not in name_to_stem:
            raise ValueError(f"Unknown reference table: {table_name}")

        expected_stem = name_to_stem[table_name]
        s3 = _s3_client()
        paginator = s3.get_paginator("list_objects_v2")

        # Find the file in REFERENCE/ whose stem matches
        for page in paginator.paginate(Bucket=Config.S3_RAW_BUCKET, Prefix=Config.S3_REFERENCE_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if not filename or "." not in filename:
                    continue
                stem = filename.rsplit(".", 1)[0]
                if stem == expected_stem:
                    return _load_df_from_s3(Config.S3_RAW_BUCKET, key, sheet_name)

        raise FileNotFoundError(
            f"Reference table '{table_name}' not found in S3 s3://{Config.S3_RAW_BUCKET}/{Config.S3_REFERENCE_PREFIX}. "
            f"Please upload '{expected_stem}.xlsx' to the REFERENCE folder."
        )

    @staticmethod
    def load_custom_reference_table(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load a custom reference table.

        file_path is a relative path like 'data/raw/REFERENCE/LOB_LowOrgList.xlsx'.
        We extract the filename stem and look it up in S3 REFERENCE/.
        """
        # Extract just the filename stem from the path
        filename = file_path.replace("\\", "/").split("/")[-1]
        if "." in filename:
            stem = filename.rsplit(".", 1)[0]
        else:
            stem = filename

        s3 = _s3_client()
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=Config.S3_RAW_BUCKET, Prefix=Config.S3_REFERENCE_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                key_filename = key.split("/")[-1]
                if not key_filename or "." not in key_filename:
                    continue
                key_stem = key_filename.rsplit(".", 1)[0]
                if key_stem == stem:
                    return _load_df_from_s3(Config.S3_RAW_BUCKET, key, sheet_name)

        raise FileNotFoundError(
            f"Custom reference table '{filename}' not found in S3 "
            f"s3://{Config.S3_RAW_BUCKET}/{Config.S3_REFERENCE_PREFIX}"
        )

    # --- Session Operations ---

    @staticmethod
    def load_session_registry() -> dict:
        """Load the session registry from S3."""
        data = _get_object_json(Config.S3_DATA_BUCKET, Config.S3_REGISTRY_KEY)
        return data if data else {"sessions": []}

    @staticmethod
    def save_session_registry(registry: dict):
        """Save the session registry to S3."""
        _put_object_json(Config.S3_DATA_BUCKET, Config.S3_REGISTRY_KEY, registry)

    @staticmethod
    def create_session(name: str, description: str = "") -> dict:
        """Create a new session and persist its metadata to S3."""
        session_id = S3StorageService.generate_session_id()
        now = datetime.utcnow().isoformat() + "Z"

        session_meta = {
            "sessionId": session_id,
            "name": name,
            "description": description,
            "createdAt": now,
            "lastModified": now,
            "status": "active",
            "sources": [],
            "calculations": [],
        }

        meta_key = f"{Config.S3_SESSIONS_PREFIX}{session_id}/session_meta.json"
        _put_object_json(Config.S3_DATA_BUCKET, meta_key, session_meta)

        registry = S3StorageService.load_session_registry()
        registry["sessions"].append({
            "sessionId": session_id,
            "name": name,
            "createdAt": now,
            "lastModified": now,
            "status": "active",
            "configuredSources": [],
            "hasCalculations": False,
        })
        S3StorageService.save_session_registry(registry)

        return session_meta

    @staticmethod
    def get_session(session_id: str) -> Optional[dict]:
        """Load session metadata from S3."""
        key = f"{Config.S3_SESSIONS_PREFIX}{session_id}/session_meta.json"
        return _get_object_json(Config.S3_DATA_BUCKET, key)

    @staticmethod
    def update_session(session_id: str, updates: dict):
        """Update session metadata in S3."""
        session = S3StorageService.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        session.update(updates)
        session["lastModified"] = datetime.utcnow().isoformat() + "Z"

        key = f"{Config.S3_SESSIONS_PREFIX}{session_id}/session_meta.json"
        _put_object_json(Config.S3_DATA_BUCKET, key, session)

        registry = S3StorageService.load_session_registry()
        for sess in registry["sessions"]:
            if sess["sessionId"] == session_id:
                sess["lastModified"] = session["lastModified"]
                if "configuredSources" in updates:
                    sess["configuredSources"] = updates["configuredSources"]
                if "hasCalculations" in updates:
                    sess["hasCalculations"] = updates["hasCalculations"]
                break
        S3StorageService.save_session_registry(registry)

    @staticmethod
    def delete_session(session_id: str) -> list[str]:
        """Delete all S3 objects for a session. Returns list of deleted keys."""
        deleted = []
        s3 = _s3_client()

        for prefix in [
            f"{Config.S3_SESSIONS_PREFIX}{session_id}/",
            f"{Config.S3_CALCULATED_PREFIX}{session_id}/",
        ]:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=Config.S3_DATA_BUCKET, Prefix=prefix):
                keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if keys:
                    s3.delete_objects(Bucket=Config.S3_DATA_BUCKET, Delete={"Objects": keys})
                    deleted.extend([k["Key"] for k in keys])

        registry = S3StorageService.load_session_registry()
        registry["sessions"] = [s for s in registry["sessions"] if s["sessionId"] != session_id]
        S3StorageService.save_session_registry(registry)

        return deleted

    # --- Config Operations ---

    @staticmethod
    def save_source_config(session_id: str, source_id: str, config: dict) -> int:
        """Save column mapping configuration to S3."""
        session = S3StorageService.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        config_key = f"{Config.S3_SESSIONS_PREFIX}{session_id}/sources/{source_id}/config.json"

        version = 1
        existing = _get_object_json(Config.S3_DATA_BUCKET, config_key)
        if existing:
            version = existing.get("configVersion", 0) + 1

        config["configVersion"] = version
        config["lastModified"] = datetime.utcnow().isoformat() + "Z"

        _put_object_json(Config.S3_DATA_BUCKET, config_key, config)

        # Update session metadata
        source_exists = False
        for source in session.get("sources", []):
            if source["sourceId"] == source_id:
                source["configuredAt"] = config["lastModified"]
                source["configVersion"] = version
                source_exists = True
                break

        if not source_exists:
            session.setdefault("sources", []).append({
                "sourceId": source_id,
                "configuredAt": config["lastModified"],
                "configVersion": version,
                "hasCalculation": False,
            })

        configured_sources = [s["sourceId"] for s in session.get("sources", [])]
        S3StorageService.update_session(session_id, {
            "sources": session["sources"],
            "configuredSources": configured_sources,
        })

        return version

    @staticmethod
    def get_source_config(session_id: str, source_id: str) -> Optional[dict]:
        """Load saved column mapping configuration from S3."""
        key = f"{Config.S3_SESSIONS_PREFIX}{session_id}/sources/{source_id}/config.json"
        return _get_object_json(Config.S3_DATA_BUCKET, key)

    # --- Calculated Data Operations ---

    @staticmethod
    def save_calculated_data(session_id: str, source_id: str, df: pd.DataFrame) -> str:
        """Save calculated emissions DataFrame as CSV to S3."""
        df_save = df.copy()
        if "formulaReference" in df_save.columns:
            df_save = df_save.drop(columns=["formulaReference"])

        key = f"{Config.S3_CALCULATED_PREFIX}{session_id}/{source_id}_emissions.csv"
        buf = io.StringIO()
        df_save.to_csv(buf, index=False)

        _s3_client().put_object(
            Bucket=Config.S3_DATA_BUCKET,
            Key=key,
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

        return f"s3://{Config.S3_DATA_BUCKET}/{key}"

    @staticmethod
    def load_calculated_data(session_id: str, source_id: str) -> Optional[pd.DataFrame]:
        """Load a single calculated emissions CSV from S3."""
        key = f"{Config.S3_CALCULATED_PREFIX}{session_id}/{source_id}_emissions.csv"
        try:
            return _load_df_from_s3(Config.S3_DATA_BUCKET, key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    @staticmethod
    def load_all_calculated_data(session_id: str) -> pd.DataFrame:
        """Load and concatenate all calculated CSVs for a session from S3."""
        s3 = _s3_client()
        prefix = f"{Config.S3_CALCULATED_PREFIX}{session_id}/"
        dfs = []

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=Config.S3_DATA_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_emissions.csv"):
                    df = _load_df_from_s3(Config.S3_DATA_BUCKET, key)
                    source = key.split("/")[-1].replace("_emissions.csv", "")
                    df["_source"] = source
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def save_calculation_log(session_id: str, calculation_id: str, log_data: dict):
        """Save calculation log JSON to S3."""
        key = f"{Config.S3_CALCULATED_PREFIX}{session_id}/calculation_log.json"
        _put_object_json(Config.S3_DATA_BUCKET, key, log_data)
