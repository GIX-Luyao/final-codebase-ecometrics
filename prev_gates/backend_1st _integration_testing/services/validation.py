"""Validation logic for EcoMetrics backend."""

import re


def validate_column_name(name: str) -> tuple[bool, str | None]:
    """
    Validate a column name against naming rules.

    Rules:
    - Must be a string
    - Cannot be a pure number (e.g., "123")
    - Must start with a letter or underscore

    Args:
        name: The column name to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if not isinstance(name, str):
        return False, f"Column name must be string, got {type(name).__name__}"

    if not name:
        return False, "Column name cannot be empty"

    if name.isdigit():
        return False, f"Column name '{name}' is invalid: cannot be a pure number"

    if not re.match(r"^[a-zA-Z_]", name):
        return False, f"Column name '{name}' must start with letter or underscore"

    return True, None


def validate_change(change: dict, df_columns: list[str] | None = None) -> tuple[bool, str | None]:
    """
    Validate a single change object.

    Args:
        change: The change object to validate
        df_columns: Optional list of current DataFrame columns for additional validation

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(change, dict):
        return False, "Change must be an object"

    change_type = change.get("changeType")

    if change_type not in ("column_rename", "cell_edit"):
        return False, f"Invalid changeType: {change_type}"

    if change_type == "column_rename":
        if "oldName" not in change or "newName" not in change:
            return False, "column_rename requires 'oldName' and 'newName'"

        # Validate the new column name
        is_valid, error = validate_column_name(change["newName"])
        if not is_valid:
            return False, error

        # Check if oldName exists in DataFrame
        if df_columns is not None and change["oldName"] not in df_columns:
            return False, f"Column '{change['oldName']}' does not exist"

    elif change_type == "cell_edit":
        required_fields = ["rowIndex", "column", "newValue"]
        for field in required_fields:
            if field not in change:
                return False, f"cell_edit requires '{field}'"

        if not isinstance(change["rowIndex"], int) or change["rowIndex"] < 0:
            return False, "rowIndex must be a non-negative integer"

        # Check if column exists in DataFrame
        if df_columns is not None and change["column"] not in df_columns:
            return False, f"Column '{change['column']}' does not exist"

    return True, None


def validate_changes_batch(changes: list, df_columns: list[str] | None = None) -> tuple[bool, list]:
    """
    Validate all changes in a batch.

    Args:
        changes: List of change objects to validate
        df_columns: Optional list of current DataFrame columns

    Returns:
        Tuple of (all_valid, list of errors)
        If all valid, errors list is empty
    """
    if not isinstance(changes, list):
        return False, ["Changes must be a list"]

    if len(changes) == 0:
        return False, ["Changes list cannot be empty"]

    errors = []
    # Track column renames to update df_columns as we validate
    current_columns = list(df_columns) if df_columns else None

    for i, change in enumerate(changes):
        is_valid, error = validate_change(change, current_columns)
        if not is_valid:
            errors.append(f"Change {i}: {error}")
        elif current_columns is not None and change.get("changeType") == "column_rename":
            # Update tracked columns for subsequent validations
            old_name = change["oldName"]
            new_name = change["newName"]
            if old_name in current_columns:
                idx = current_columns.index(old_name)
                current_columns[idx] = new_name

    return len(errors) == 0, errors
