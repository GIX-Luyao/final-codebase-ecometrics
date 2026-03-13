"""
File I/O operations for EcoMetrics Backend.
"""
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import Config


class StorageService:
    """Handles all file storage operations."""

    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"sess_{secrets.token_hex(3)}"

    @staticmethod
    def generate_calculation_id(session_id: str) -> str:
        """Generate a unique calculation ID."""
        return f"calc_{session_id.replace('sess_', '')}_{secrets.token_hex(3)}"

    # --- Source File Operations ---

    @staticmethod
    def list_source_files() -> list[dict]:
        """List all available source files in CONSUMPTION directory."""
        sources = []
        consumption_dir = Config.CONSUMPTION_DIR

        if not consumption_dir.exists():
            return sources

        for file_path in consumption_dir.iterdir():
            # Skip temp files (e.g. ~$filename.xlsx)
            if file_path.name.startswith("~$"):
                continue
            if file_path.suffix.lower() in Config.SUPPORTED_EXTENSIONS:
                source_id = file_path.stem.lower().replace(" ", "_").replace("-", "_")
                file_stat = file_path.stat()

                source_info = {
                    "sourceId": source_id,
                    "name": file_path.stem,
                    "filePath": str(file_path.relative_to(Config.BASE_DIR)),
                    "fileSize": file_stat.st_size,
                    "lastModified": datetime.fromtimestamp(file_stat.st_mtime).isoformat() + "Z",
                }

                # Get sheet names for Excel files
                if file_path.suffix.lower() in [".xlsx", ".xls"]:
                    try:
                        xl = pd.ExcelFile(file_path)
                        source_info["sheets"] = xl.sheet_names
                    except Exception:
                        source_info["sheets"] = []
                else:
                    source_info["sheets"] = None

                sources.append(source_info)

        return sources

    @staticmethod
    def list_reference_files() -> list[dict]:
        """List all available reference files in REFERENCE directory."""
        sources = []
        reference_dir = Config.REFERENCE_DIR

        if not reference_dir.exists():
            return sources

        for file_path in reference_dir.iterdir():
            if file_path.name.startswith("~$"):
                continue
            if file_path.suffix.lower() in Config.SUPPORTED_EXTENSIONS:
                source_id = "ref_" + file_path.stem.lower().replace(" ", "_").replace("-", "_")
                file_stat = file_path.stat()

                source_info = {
                    "sourceId": source_id,
                    "name": file_path.stem,
                    "filePath": str(file_path.relative_to(Config.BASE_DIR)),
                    "fileSize": file_stat.st_size,
                    "lastModified": datetime.fromtimestamp(file_stat.st_mtime).isoformat() + "Z",
                }

                if file_path.suffix.lower() in [".xlsx", ".xls"]:
                    try:
                        xl = pd.ExcelFile(file_path)
                        source_info["sheets"] = xl.sheet_names
                    except Exception:
                        source_info["sheets"] = []
                else:
                    source_info["sheets"] = None

                sources.append(source_info)

        return sources

    @staticmethod
    def get_source_path(source_id: str) -> Optional[Path]:
        """Get the file path for a source ID."""
        # Check if it's a reference file (prefixed with ref_)
        if source_id.startswith("ref_"):
            search_id = source_id[4:]  # strip "ref_" prefix
            search_dir = Config.REFERENCE_DIR
        else:
            search_id = source_id
            search_dir = Config.CONSUMPTION_DIR

        if not search_dir.exists():
            return None

        for file_path in search_dir.iterdir():
            file_source_id = file_path.stem.lower().replace(" ", "_").replace("-", "_")
            if file_source_id == search_id and file_path.suffix.lower() in Config.SUPPORTED_EXTENSIONS:
                return file_path

        return None

    @staticmethod
    def load_raw_data(source_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load raw data file from storage."""
        file_path = StorageService.get_source_path(source_id)

        if file_path is None:
            raise FileNotFoundError(f"Source not found: {source_id}")

        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet_name if sheet_name else 0)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return df

    # --- Reference Table Operations ---

    @staticmethod
    def load_reference_table(table_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load a reference table by name."""
        table_paths = {
            "EFID": Config.EFID_FILE,
            "GWP": Config.GWP_FILE,
            "LOB": Config.LOB_FILE,
        }

        if table_name not in table_paths:
            raise ValueError(f"Unknown reference table: {table_name}")

        file_path = table_paths[table_name]

        if not file_path.exists():
            raise FileNotFoundError(f"Reference table not found: {file_path}")

        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet_name if sheet_name else 0)
        else:
            df = pd.read_csv(file_path)

        return df

    @staticmethod
    def load_custom_reference_table(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load a custom reference table from a path."""
        path = Path(file_path)

        # Handle relative paths
        if not path.is_absolute():
            path = Config.BASE_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Reference table not found: {file_path}")

        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
        else:
            df = pd.read_csv(path)

        return df

    # --- Session Operations ---

    @staticmethod
    def load_session_registry() -> dict:
        """Load the session registry."""
        if not Config.SESSION_REGISTRY.exists():
            return {"sessions": []}

        with open(Config.SESSION_REGISTRY, "r") as f:
            return json.load(f)

    @staticmethod
    def save_session_registry(registry: dict):
        """Save the session registry."""
        Config.SESSION_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        with open(Config.SESSION_REGISTRY, "w") as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def create_session(name: str, description: str = "") -> dict:
        """Create a new session."""
        session_id = StorageService.generate_session_id()
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

        # Create session directory
        session_dir = Config.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "sources").mkdir(exist_ok=True)

        # Save session metadata
        with open(session_dir / "session_meta.json", "w") as f:
            json.dump(session_meta, f, indent=2)

        # Update registry
        registry = StorageService.load_session_registry()
        registry["sessions"].append({
            "sessionId": session_id,
            "name": name,
            "createdAt": now,
            "lastModified": now,
            "status": "active",
            "configuredSources": [],
            "hasCalculations": False,
        })
        StorageService.save_session_registry(registry)

        return session_meta

    @staticmethod
    def get_session(session_id: str) -> Optional[dict]:
        """Get session metadata."""
        session_dir = Config.get_session_dir(session_id)
        meta_file = session_dir / "session_meta.json"

        if not meta_file.exists():
            return None

        with open(meta_file, "r") as f:
            return json.load(f)

    @staticmethod
    def update_session(session_id: str, updates: dict):
        """Update session metadata."""
        session = StorageService.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        session.update(updates)
        session["lastModified"] = datetime.utcnow().isoformat() + "Z"

        session_dir = Config.get_session_dir(session_id)
        with open(session_dir / "session_meta.json", "w") as f:
            json.dump(session, f, indent=2)

        # Update registry
        registry = StorageService.load_session_registry()
        for sess in registry["sessions"]:
            if sess["sessionId"] == session_id:
                sess["lastModified"] = session["lastModified"]
                if "configuredSources" in updates:
                    sess["configuredSources"] = updates["configuredSources"]
                if "hasCalculations" in updates:
                    sess["hasCalculations"] = updates["hasCalculations"]
                break
        StorageService.save_session_registry(registry)

    @staticmethod
    def delete_session(session_id: str) -> list[str]:
        """Delete a session and all associated data."""
        deleted_files = []
        session_dir = Config.get_session_dir(session_id)
        calculated_dir = Config.get_calculated_dir(session_id)

        # Delete session files
        if session_dir.exists():
            for root, dirs, files in os.walk(session_dir, topdown=False):
                for file in files:
                    file_path = Path(root) / file
                    deleted_files.append(str(file_path.relative_to(Config.BASE_DIR)))
                    file_path.unlink()
                for d in dirs:
                    (Path(root) / d).rmdir()
            session_dir.rmdir()

        # Delete calculated files
        if calculated_dir.exists():
            for root, dirs, files in os.walk(calculated_dir, topdown=False):
                for file in files:
                    file_path = Path(root) / file
                    deleted_files.append(str(file_path.relative_to(Config.BASE_DIR)))
                    file_path.unlink()
                for d in dirs:
                    (Path(root) / d).rmdir()
            calculated_dir.rmdir()

        # Update registry
        registry = StorageService.load_session_registry()
        registry["sessions"] = [s for s in registry["sessions"] if s["sessionId"] != session_id]
        StorageService.save_session_registry(registry)

        return deleted_files

    # --- Config Operations ---

    @staticmethod
    def save_source_config(session_id: str, source_id: str, config: dict) -> int:
        """Save column mapping configuration for a source."""
        session = StorageService.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        sources_dir = Config.get_session_sources_dir(session_id)
        source_config_dir = sources_dir / source_id
        source_config_dir.mkdir(parents=True, exist_ok=True)

        config_file = source_config_dir / "config.json"

        # Determine version
        version = 1
        if config_file.exists():
            with open(config_file, "r") as f:
                old_config = json.load(f)
                version = old_config.get("configVersion", 0) + 1

        # Add metadata
        config["configVersion"] = version
        config["lastModified"] = datetime.utcnow().isoformat() + "Z"

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

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
        StorageService.update_session(session_id, {
            "sources": session["sources"],
            "configuredSources": configured_sources,
        })

        return version

    @staticmethod
    def get_source_config(session_id: str, source_id: str) -> Optional[dict]:
        """Get saved configuration for a source."""
        config_file = Config.get_session_sources_dir(session_id) / source_id / "config.json"

        if not config_file.exists():
            return None

        with open(config_file, "r") as f:
            return json.load(f)

    # --- Calculated Data Operations ---

    @staticmethod
    def save_calculated_data(session_id: str, source_id: str, df: pd.DataFrame) -> str:
        """Save calculated emissions data to CSV file."""
        calculated_dir = Config.get_calculated_dir(session_id)
        calculated_dir.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable columns (like formulaReference dict)
        df_save = df.copy()
        if "formulaReference" in df_save.columns:
            df_save = df_save.drop(columns=["formulaReference"])

        file_path = calculated_dir / f"{source_id}_emissions.csv"
        df_save.to_csv(file_path, index=False)

        return str(file_path.relative_to(Config.BASE_DIR))

    @staticmethod
    def load_calculated_data(session_id: str, source_id: str) -> Optional[pd.DataFrame]:
        """Load calculated emissions data from CSV file."""
        file_path = Config.get_calculated_dir(session_id) / f"{source_id}_emissions.csv"

        if not file_path.exists():
            return None

        return pd.read_csv(file_path)

    @staticmethod
    def load_all_calculated_data(session_id: str) -> pd.DataFrame:
        """Load all calculated data for a session."""
        calculated_dir = Config.get_calculated_dir(session_id)

        if not calculated_dir.exists():
            return pd.DataFrame()

        dfs = []
        for file_path in calculated_dir.glob("*_emissions.csv"):
            df = pd.read_csv(file_path)
            df["_source"] = file_path.stem.replace("_emissions", "")
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def save_calculation_log(session_id: str, calculation_id: str, log_data: dict):
        """Save calculation log."""
        calculated_dir = Config.get_calculated_dir(session_id)
        calculated_dir.mkdir(parents=True, exist_ok=True)

        with open(calculated_dir / "calculation_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
