"""
Configuration management for EcoMetrics Backend.
Supports both local filesystem and AWS S3 storage modes.
"""
import os
from pathlib import Path


class Config:
    """Application configuration."""

    # --- Storage backend ---
    # Set STORAGE_BACKEND=s3 in ECS task environment to enable S3 mode.
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "s3")  # "local" | "s3"

    # --- S3 configuration (only used when STORAGE_BACKEND=s3) ---
    S3_RAW_BUCKET = os.getenv("S3_RAW_BUCKET", "spu-emissions-raw-data")
    S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "spu-emissions-processed-data")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

    # S3 key prefixes
    S3_CONSUMPTION_PREFIX = "CONSUMPTION/"
    S3_REFERENCE_PREFIX = "REFERENCE/"
    S3_SESSIONS_PREFIX = "sessions/"
    S3_CALCULATED_PREFIX = "calculated/"
    S3_REGISTRY_KEY = "sessions/registry.json"

    # --- Local base paths (used when STORAGE_BACKEND=local) ---
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"

    # Data subdirectories
    RAW_DATA_DIR = DATA_DIR / "raw"
    CONSUMPTION_DIR = RAW_DATA_DIR / "CONSUMPTION"
    REFERENCE_DIR = RAW_DATA_DIR / "REFERENCE"
    SESSIONS_DIR = DATA_DIR / "sessions"
    CALCULATED_DIR = DATA_DIR / "calculated"
    LOGS_DIR = BASE_DIR / "logs"

    # Reference file paths (local fallback)
    EFID_FILE = REFERENCE_DIR / "EFID.xlsx"
    GWP_FILE = REFERENCE_DIR / "GWPs.xlsx"
    LOB_FILE = REFERENCE_DIR / "LOB_LowOrgList.xlsx"

    # Session registry (local mode)
    SESSION_REGISTRY = SESSIONS_DIR / "registry.json"

    # Flask config
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 8000))

    # CORS config - set to CloudFront domain in production
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # Logging
    LOG_FILE = LOGS_DIR / "backend.log"
    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(session_id)s] %(component)s - %(action)s - %(status)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Default GWP version
    DEFAULT_GWP_VERSION = "AR5"

    # Required columns for standard schema
    REQUIRED_COLUMNS = ["ACCT_ID", "Year", "Consumption", "Unit", "Subtype"]

    # Supported file extensions
    SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

    @classmethod
    def is_s3_mode(cls) -> bool:
        """Return True when running in S3/cloud mode."""
        return cls.STORAGE_BACKEND.lower() == "s3"

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist (no-op in S3 mode)."""
        if cls.is_s3_mode():
            return
        dirs = [
            cls.CONSUMPTION_DIR,
            cls.REFERENCE_DIR,
            cls.SESSIONS_DIR,
            cls.CALCULATED_DIR,
            cls.LOGS_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_session_dir(cls, session_id: str) -> Path:
        """Get directory for a specific session."""
        return cls.SESSIONS_DIR / session_id

    @classmethod
    def get_session_sources_dir(cls, session_id: str) -> Path:
        """Get sources directory for a specific session."""
        return cls.get_session_dir(session_id) / "sources"

    @classmethod
    def get_calculated_dir(cls, session_id: str) -> Path:
        """Get calculated data directory for a specific session."""
        return cls.CALCULATED_DIR / session_id
