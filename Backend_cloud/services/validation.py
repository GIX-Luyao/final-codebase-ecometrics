"""
Configuration and data validation for EcoMetrics Backend.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from .storage import StorageService


class ValidationService:
    """Handles validation of configurations and data."""

    @staticmethod
    def validate_config(config: dict, source_id: str) -> dict:
        """
        Validate a column mapping configuration.

        Returns:
            dict with 'valid', 'errors', and 'warnings' keys
        """
        errors = []
        warnings = []

        # Check required fields
        if "columnMappings" not in config:
            errors.append("Missing required field: 'columnMappings'")
            return {"valid": False, "errors": errors, "warnings": warnings}

        column_mappings = config["columnMappings"]

        # Check required columns are mapped
        required_cols = ["ACCT_ID", "Consumption", "Unit", "Subtype", "EF_ID"]
        for col in required_cols:
            if col not in column_mappings:
                errors.append(f"Missing required column mapping: '{col}'")

        # Year can be either direct or derived
        if "Year" not in column_mappings:
            # Check if Year can be derived from another column
            has_date_derivation = False
            for mapping in column_mappings.values():
                if isinstance(mapping, dict) and mapping.get("extractType") == "year":
                    has_date_derivation = True
                    break
            if not has_date_derivation:
                errors.append("Missing required column mapping: 'Year'")

        # Load source data to validate columns
        try:
            df = StorageService.load_raw_data(source_id, config.get("sourceSheet"))
            source_columns = set(df.columns)
        except FileNotFoundError:
            errors.append(f"Source file not found for: {source_id}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        except Exception as e:
            errors.append(f"Error loading source file: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Validate each mapping
        for target_col, mapping in column_mappings.items():
            if not isinstance(mapping, dict):
                errors.append(f"Invalid mapping format for column '{target_col}'")
                continue

            # Validate direct column mapping
            if "sourceColumn" in mapping:
                source_col = mapping["sourceColumn"]
                if source_col not in source_columns:
                    errors.append(f"Source column '{source_col}' does not exist in raw data")

            # Validate derived mapping
            elif "derivedFrom" in mapping:
                source_col = mapping["derivedFrom"]
                if source_col not in source_columns:
                    errors.append(f"Source column '{source_col}' does not exist in raw data")
                if "extractType" not in mapping:
                    errors.append(f"Derived mapping for '{target_col}' missing 'extractType'")
                elif mapping["extractType"] not in ["year", "month", "day", "quarter", "dayofweek"]:
                    errors.append(f"Invalid extractType '{mapping['extractType']}' for column '{target_col}'")

            # Validate reference table join
            elif "referenceTable" in mapping:
                ref_table_path = mapping["referenceTable"]
                ref_key = mapping.get("refKey")
                source_key = mapping.get("sourceKey")
                column_joined = mapping.get("columnJoined")

                if not ref_key:
                    errors.append(f"Reference mapping for '{target_col}' missing 'refKey'")
                if not source_key:
                    errors.append(f"Reference mapping for '{target_col}' missing 'sourceKey'")
                if not column_joined:
                    errors.append(f"Reference mapping for '{target_col}' missing 'columnJoined'")

                if source_key and source_key not in source_columns:
                    errors.append(f"Source key column '{source_key}' does not exist in raw data")

                # Check reference table exists
                try:
                    ref_df = StorageService.load_custom_reference_table(
                        ref_table_path, mapping.get("refSheet")
                    )
                    if ref_key and ref_key not in ref_df.columns:
                        errors.append(f"Reference key column '{ref_key}' not found in {ref_table_path}")
                    if column_joined and column_joined not in ref_df.columns:
                        errors.append(f"Column to join '{column_joined}' not found in {ref_table_path}")
                except FileNotFoundError:
                    errors.append(f"Reference table not found: {ref_table_path}")
                except Exception as e:
                    errors.append(f"Error loading reference table {ref_table_path}: {str(e)}")

            # Validate static value (always valid if present)
            elif "staticValue" in mapping:
                pass

            # Validate GHG expansion
            elif "ghgType" in mapping:
                ghg_types = mapping["ghgType"]
                if not isinstance(ghg_types, list) or len(ghg_types) == 0:
                    errors.append("GHG mapping must have a non-empty list of 'ghgType'")

            # Validate pattern-based column construction
            elif "pattern" in mapping:
                pattern = mapping["pattern"]
                if not isinstance(pattern, str) or len(pattern) == 0:
                    errors.append(f"Pattern mapping for '{target_col}' must have a non-empty 'pattern' string")
                # Check that referenced columns in pattern exist
                import re
                placeholders = re.findall(r'\{(\w+)\}', pattern)
                # GHG is a special system-generated column (created during GHG expansion)
                system_generated_columns = {"GHG"}
                for placeholder in placeholders:
                    # Skip system-generated columns like GHG
                    if placeholder in system_generated_columns:
                        continue
                    # Placeholder can reference either a source column or a target column (already mapped)
                    if placeholder not in source_columns and placeholder not in column_mappings:
                        warnings.append(f"Pattern placeholder '{{{placeholder}}}' in '{target_col}' - ensure this column exists or is mapped")

            else:
                errors.append(f"Invalid mapping configuration for column: '{target_col}'")

        # Validate filters
        filters = config.get("filters", [])
        valid_operators = ["=", "!=", ">", ">=", "<", "<=", "in", "not_in", "is_null", "not_null"]
        for i, filter_config in enumerate(filters):
            if "column" not in filter_config:
                errors.append(f"Filter {i} missing 'column' field")
            elif filter_config["column"] not in source_columns:
                errors.append(f"Filter column '{filter_config['column']}' does not exist in raw data")

            if "operator" not in filter_config:
                errors.append(f"Filter {i} missing 'operator' field")
            elif filter_config["operator"] not in valid_operators:
                errors.append(f"Invalid filter operator: '{filter_config['operator']}'")

            if filter_config.get("operator") not in ["is_null", "not_null"] and "value" not in filter_config:
                errors.append(f"Filter {i} missing 'value' field")

        # Validate EFID lookup config
        efid_lookup = config.get("efidLookup", {})
        if efid_lookup:
            if "pattern" not in efid_lookup:
                warnings.append("efidLookup missing 'pattern', will use default")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def validate_data_after_mapping(df: pd.DataFrame, column_mappings: dict) -> dict:
        """
        Validate data after mapping is applied.

        Returns:
            dict with validation results and warnings
        """
        warnings = []

        # Check for null values in required columns
        required_cols = ["ACCT_ID", "Year", "Consumption", "Unit", "Subtype"]
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    pct = (null_count / len(df)) * 100
                    if pct > 0:
                        warnings.append(f"Column '{col}' has {pct:.1f}% null values after mapping")

        # Check for negative consumption values
        if "Consumption" in df.columns:
            negative_count = (df["Consumption"] < 0).sum()
            if negative_count > 0:
                warnings.append(f"Found {negative_count} negative consumption values")

        return {
            "status": "valid" if len(warnings) == 0 else "valid_with_warnings",
            "warnings": warnings,
        }

    @staticmethod
    def validate_calculation_request(session_id: str, sources: list[str]) -> dict:
        """
        Validate a calculation request.

        Returns:
            dict with validation results
        """
        errors = []

        # Check session exists
        session = StorageService.get_session(session_id)
        if session is None:
            errors.append(f"Session not found: {session_id}")
            return {"valid": False, "errors": errors}

        # Check each source has a config
        for source_id in sources:
            config = StorageService.get_source_config(session_id, source_id)
            if config is None:
                errors.append(f"No configuration found for source: {source_id}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
