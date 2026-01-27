"""File storage operations for EcoMetrics backend."""

import json
import uuid
from pathlib import Path

import pandas as pd

from config import Config


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{uuid.uuid4().hex[:8]}"


def read_raw_file(data_type: str, invalid: bool = False) -> pd.DataFrame:
    """
    Read a raw CSV file from the data/raw/ directory.

    Args:
        data_type: Type of data ("fleetfuel" or "efid")
        invalid: If True, load the invalid test file (for fleetfuel only)

    Returns:
        DataFrame with the raw data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If data_type is not recognized
    """
    if invalid and data_type == "fleetfuel":
        file_key = "fleetfuel_invalid"
    else:
        file_key = data_type

    if file_key not in Config.RAW_FILES:
        raise ValueError(f"Unknown data type: {data_type}")

    file_path = Config.RAW_DATA_DIR / Config.RAW_FILES[file_key]

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    return pd.read_csv(file_path)


def get_raw_file_name(data_type: str, invalid: bool = False) -> str:
    """Get the filename for a raw data type."""
    if invalid and data_type == "fleetfuel":
        file_key = "fleetfuel_invalid"
    else:
        file_key = data_type

    return Config.RAW_FILES.get(file_key, "")


def get_session_dir(session_id: str) -> Path:
    """
    Get or create a session directory.

    Args:
        session_id: The session identifier

    Returns:
        Path to the session directory
    """
    session_dir = Config.SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def read_session_data(session_id: str, data_type: str) -> pd.DataFrame | None:
    """
    Read modified data for a session.

    Args:
        session_id: The session identifier
        data_type: Type of data ("fleetfuel" or "efid")

    Returns:
        DataFrame with session data, or None if no session data exists
    """
    session_dir = Config.SESSIONS_DIR / session_id
    file_path = session_dir / f"{data_type}_modified.csv"

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


def save_session_data(session_id: str, data_type: str, df: pd.DataFrame) -> str:
    """
    Save modified DataFrame to session CSV.

    Args:
        session_id: The session identifier
        data_type: Type of data ("fleetfuel" or "efid")
        df: DataFrame to save

    Returns:
        Relative file path where data was saved
    """
    session_dir = get_session_dir(session_id)
    file_path = session_dir / f"{data_type}_modified.csv"

    df.to_csv(file_path, index=False)

    return str(file_path.relative_to(Config.BASE_DIR))


def read_changes_log(session_id: str) -> list:
    """
    Read the changes log for a session.

    Args:
        session_id: The session identifier

    Returns:
        List of changes, or empty list if no log exists
    """
    session_dir = Config.SESSIONS_DIR / session_id
    log_path = session_dir / "changes.json"

    if not log_path.exists():
        return []

    with open(log_path, "r") as f:
        return json.load(f)


def append_changes_log(session_id: str, changes: list) -> None:
    """
    Append changes to the session's changes log.

    Args:
        session_id: The session identifier
        changes: List of change objects to append
    """
    session_dir = get_session_dir(session_id)
    log_path = session_dir / "changes.json"

    existing_changes = read_changes_log(session_id)
    existing_changes.extend(changes)

    with open(log_path, "w") as f:
        json.dump(existing_changes, f, indent=2)


def get_calculated_dir(session_id: str) -> Path:
    """
    Get or create a calculated data directory for a session.

    Args:
        session_id: The session identifier

    Returns:
        Path to the calculated data directory
    """
    calc_dir = Config.CALCULATED_DIR / session_id
    calc_dir.mkdir(parents=True, exist_ok=True)
    return calc_dir


def save_calculated_data(session_id: str, df: pd.DataFrame) -> str:
    """
    Save calculated results to data/calculated/{session_id}/.

    Args:
        session_id: The session identifier
        df: DataFrame with calculated data

    Returns:
        Relative file path where data was saved
    """
    calc_dir = get_calculated_dir(session_id)
    file_path = calc_dir / "fleetfuel_lean.csv"

    df.to_csv(file_path, index=False)

    return str(file_path.relative_to(Config.BASE_DIR))


def read_calculated_data(session_id: str) -> pd.DataFrame | None:
    """
    Read previously calculated data for a session.

    Args:
        session_id: The session identifier

    Returns:
        DataFrame with calculated data, or None if not exists
    """
    calc_dir = Config.CALCULATED_DIR / session_id
    file_path = calc_dir / "fleetfuel_lean.csv"

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)
