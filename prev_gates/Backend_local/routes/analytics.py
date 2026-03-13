"""
Analytics and visualization endpoints.
"""
from datetime import datetime
from flask import Blueprint, request, jsonify
import pandas as pd

from services.storage import StorageService


analytics_bp = Blueprint("analytics", __name__)


# Color palette for charts
CHART_COLORS = [
    "#FF6384",  # Red
    "#36A2EB",  # Blue
    "#FFCE56",  # Yellow
    "#4BC0C0",  # Teal
    "#9966FF",  # Purple
    "#FF9F40",  # Orange
    "#C9CBCF",  # Gray
    "#7BC225",  # Green
]


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


def apply_filters_to_df(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """Apply filters to a DataFrame."""
    df_filtered = df.copy()

    for filter_config in filters:
        column = filter_config.get("column")
        operator = filter_config.get("operator")
        value = filter_config.get("value")

        if column not in df_filtered.columns:
            continue

        if operator == "=":
            df_filtered = df_filtered[df_filtered[column] == value]
        elif operator == "!=":
            df_filtered = df_filtered[df_filtered[column] != value]
        elif operator == ">":
            df_filtered = df_filtered[df_filtered[column] > value]
        elif operator == ">=":
            df_filtered = df_filtered[df_filtered[column] >= value]
        elif operator == "<":
            df_filtered = df_filtered[df_filtered[column] < value]
        elif operator == "<=":
            df_filtered = df_filtered[df_filtered[column] <= value]
        elif operator == "in":
            df_filtered = df_filtered[df_filtered[column].isin(value)]
        elif operator == "not_in":
            df_filtered = df_filtered[~df_filtered[column].isin(value)]

    return df_filtered


@analytics_bp.route("/sessions/<session_id>/analytics/summary", methods=["GET"])
def get_summary(session_id: str):
    """Get high-level emissions summary."""
    # Check session exists
    session = StorageService.get_session(session_id)
    if session is None:
        return make_error_response(
            "SESSION_NOT_FOUND",
            f"Session not found: {session_id}",
            status_code=404
        )

    try:
        # Load all calculated data
        df = StorageService.load_all_calculated_data(session_id)

        if df.empty:
            return make_error_response(
                "NO_DATA",
                "No calculated data found for this session",
                status_code=404
            )

        # Calculate summary statistics
        total_mtco2e = float(df["mtCO2e_calc"].sum())

        # Group by different dimensions
        by_sector = {}
        if "Sector" in df.columns:
            by_sector = df.groupby("Sector")["mtCO2e_calc"].sum().to_dict()
            by_sector = {k: round(v, 6) for k, v in by_sector.items()}

        by_year = {}
        if "Year" in df.columns:
            by_year = df.groupby("Year")["mtCO2e_calc"].sum().to_dict()
            by_year = {str(int(k)): round(v, 6) for k, v in by_year.items()}

        by_subtype = {}
        if "Subtype" in df.columns:
            by_subtype = df.groupby("Subtype")["mtCO2e_calc"].sum().to_dict()
            by_subtype = {k: round(v, 6) for k, v in by_subtype.items()}

        by_ghg = {}
        if "GHG" in df.columns:
            by_ghg = df.groupby("GHG")["mtCO2e_calc"].sum().to_dict()
            by_ghg = {k: round(v, 6) for k, v in by_ghg.items()}

        # Year range
        year_range = {}
        if "Year" in df.columns:
            years = df["Year"].dropna().astype(int)
            if len(years) > 0:
                year_range = {"min": int(years.min()), "max": int(years.max())}

        # Count unique sources
        total_sources = df["_source"].nunique() if "_source" in df.columns else 1

        return jsonify({
            "success": True,
            "sessionId": session_id,
            "summary": {
                "totalMtCO2e": round(total_mtco2e, 6),
                "totalSources": total_sources,
                "yearRange": year_range,
                "bySector": by_sector,
                "byYear": by_year,
                "bySubtype": by_subtype,
                "byGHG": by_ghg,
            }
        })

    except Exception as e:
        return make_error_response("CALCULATION_ERROR", str(e), status_code=500)


@analytics_bp.route("/sessions/<session_id>/analytics/chart-data", methods=["POST"])
def get_chart_data(session_id: str):
    """Get data formatted for specific chart types."""
    data = request.get_json() or {}

    # Check session exists
    session = StorageService.get_session(session_id)
    if session is None:
        return make_error_response(
            "SESSION_NOT_FOUND",
            f"Session not found: {session_id}",
            status_code=404
        )

    chart_type = data.get("chartType", "pie")
    metric = data.get("metric", "mtCO2e_calc")
    group_by = data.get("groupBy", "Subtype")
    filters = data.get("filters", [])

    try:
        # Load all calculated data
        df = StorageService.load_all_calculated_data(session_id)

        if df.empty:
            return make_error_response(
                "NO_DATA",
                "No calculated data found for this session",
                status_code=404
            )

        # Apply filters
        df = apply_filters_to_df(df, filters)

        if df.empty:
            return make_error_response(
                "NO_DATA",
                "No data matches the specified filters",
                status_code=404
            )

        # Check required columns exist
        if metric not in df.columns:
            return make_error_response(
                "INVALID_PARAMETER",
                f"Metric column '{metric}' not found in data"
            )

        if group_by not in df.columns:
            return make_error_response(
                "INVALID_PARAMETER",
                f"Group by column '{group_by}' not found in data"
            )

        # Generate chart data based on type
        if chart_type == "pie":
            return _generate_pie_chart(df, metric, group_by, filters)
        elif chart_type == "bar":
            return _generate_bar_chart(df, metric, group_by, filters)
        elif chart_type == "line":
            return _generate_line_chart(df, metric, group_by, filters)
        elif chart_type == "stacked_bar":
            stack_by = data.get("stackBy", "GHG")
            return _generate_stacked_bar_chart(df, metric, group_by, stack_by, filters)
        else:
            return make_error_response(
                "INVALID_PARAMETER",
                f"Unsupported chart type: {chart_type}"
            )

    except Exception as e:
        return make_error_response("CALCULATION_ERROR", str(e), status_code=500)


def _generate_pie_chart(df: pd.DataFrame, metric: str, group_by: str, filters: list) -> dict:
    """Generate pie chart data."""
    grouped = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
    total = grouped.sum()

    data = []
    for i, (label, value) in enumerate(grouped.items()):
        data.append({
            "label": str(label),
            "value": round(float(value), 6),
            "percentage": round((float(value) / total) * 100, 2) if total > 0 else 0,
            "color": CHART_COLORS[i % len(CHART_COLORS)],
        })

    # Build title
    filter_desc = ""
    for f in filters:
        if f.get("operator") == "=":
            filter_desc += f" {f['column']}={f['value']}"

    title = f"{metric} by {group_by}"
    if filter_desc:
        title += f" ({filter_desc.strip()})"

    return jsonify({
        "success": True,
        "chartType": "pie",
        "title": title,
        "data": data,
        "total": round(float(total), 6),
    })


def _generate_bar_chart(df: pd.DataFrame, metric: str, group_by: str, filters: list) -> dict:
    """Generate bar chart data."""
    grouped = df.groupby(group_by)[metric].sum().sort_index()

    labels = [str(x) for x in grouped.index.tolist()]
    values = [round(float(v), 6) for v in grouped.values]

    title = f"{metric} by {group_by}"

    return jsonify({
        "success": True,
        "chartType": "bar",
        "title": title,
        "labels": labels,
        "datasets": [
            {
                "label": metric,
                "data": values,
                "backgroundColor": CHART_COLORS[0],
            }
        ],
    })


def _generate_line_chart(df: pd.DataFrame, metric: str, group_by: str, filters: list) -> dict:
    """Generate line chart data (time series)."""
    grouped = df.groupby(group_by)[metric].sum().sort_index()

    labels = [str(x) for x in grouped.index.tolist()]
    values = [round(float(v), 6) for v in grouped.values]

    title = f"{metric} over {group_by}"

    return jsonify({
        "success": True,
        "chartType": "line",
        "title": title,
        "labels": labels,
        "datasets": [
            {
                "label": metric,
                "data": values,
                "borderColor": CHART_COLORS[0],
                "fill": False,
            }
        ],
    })


def _generate_stacked_bar_chart(
    df: pd.DataFrame,
    metric: str,
    group_by: str,
    stack_by: str,
    filters: list
) -> dict:
    """Generate stacked bar chart data."""
    if stack_by not in df.columns:
        # Fall back to regular bar chart
        return _generate_bar_chart(df, metric, group_by, filters)

    pivot = df.pivot_table(
        values=metric,
        index=group_by,
        columns=stack_by,
        aggfunc="sum",
        fill_value=0
    )

    labels = [str(x) for x in pivot.index.tolist()]
    datasets = []

    for i, col in enumerate(pivot.columns):
        datasets.append({
            "label": str(col),
            "data": [round(float(v), 6) for v in pivot[col].values],
            "backgroundColor": CHART_COLORS[i % len(CHART_COLORS)],
        })

    title = f"{metric} by {group_by} (stacked by {stack_by})"

    return jsonify({
        "success": True,
        "chartType": "stacked_bar",
        "title": title,
        "labels": labels,
        "datasets": datasets,
    })
