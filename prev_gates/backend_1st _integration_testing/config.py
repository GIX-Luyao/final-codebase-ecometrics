"""Configuration settings for the EcoMetrics backend."""

from pathlib import Path


class Config:
    """Application configuration."""

    # Directory paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    SESSIONS_DIR = DATA_DIR / "sessions"
    CALCULATED_DIR = DATA_DIR / "calculated"
    LOGS_DIR = BASE_DIR / "logs"
    LOG_FILE = LOGS_DIR / "backend.log"

    # Flask settings
    DEBUG = True
    HOST = "localhost"
    PORT = 8000

    # File mappings
    RAW_FILES = {
        "fleetfuel": "FleetFuel_Raw.csv",
        "fleetfuel_invalid": "FleetFuel_Invalid.csv",
        "efid": "EFID_Reference.csv"
    }

    # Logging format
    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
