"""Emissions calculation logic for EcoMetrics backend."""

from datetime import datetime

import pandas as pd

from config import Config
from services.storage import (
    read_raw_file,
    read_session_data,
    save_calculated_data
)


def create_lookup_dicts(gwp_df: pd.DataFrame, ef_df: pd.DataFrame, gwp_version: str = "AR5_GWP") -> tuple[dict, dict]:
    """
    Return GWP and EF lookup dictionaries.

    Args:
        gwp_df: DataFrame with GWP values (columns: GHG_Name, AR5_GWP)
        ef_df: DataFrame with emission factors
        gwp_version: Column name for GWP version to use

    Returns:
        Tuple of (gwp_dict, ef_dict)
    """
    gwp_dict = gwp_df.set_index("GHG_Name")[gwp_version].to_dict()
    ef_dict = (
        ef_df
        .set_index(["Subtype", "GHG", "Sector"])["GHG_MTperUnit"]
        .to_dict()
    )
    return gwp_dict, ef_dict


def preprocess_fleet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the fleet fuel file.

    Handles both original column names and potentially renamed columns.

    Args:
        df: Raw fleet fuel DataFrame

    Returns:
        Preprocessed and aggregated DataFrame
    """
    columns_to_drop = [
        'YEAR', 'MANUFACTURER', 'MODEL', 'EQTYP_EQUIP_TYPE', 'DESCRIPTION',
        'IN_SERVICE_DATE', 'RETIRE_DATE', 'EST_REPLACE_YR',
        'METER_1_LIFE_TOTAL', 'METER_2_LIFE_TOTAL',
        'mtCO2', 'mtCH4', 'mtN2O', 'mtCO2e'
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    # Column mapping with fallbacks for potentially renamed columns
    # Each target column has a list of possible source names (case-insensitive matching)
    column_mappings = {
        'ACCT_ID': ['EQ_EQUIP_NO', 'ACCT_ID'],
        'LowOrg': ['DEPT_DEPT_CODE', 'LowOrg'],
        'Vehicle Location': ['LOC_STATION_LOC', 'Vehicle Location'],
        'Subtype': ['FUEL_TYPE', 'Fuel_Type', 'Subtype'],
        'Consumption': ['QTY_FUEL', 'Consumption'],
        'FillDate': ['FTK_DATE', 'FillDate']
    }

    # Build rename dict only for columns that exist
    rename_dict = {}
    for target, sources in column_mappings.items():
        for source in sources:
            if source in df.columns and source != target:
                rename_dict[source] = target
                break

    df = df.rename(columns=rename_dict)

    df['Unit'] = 'gal'
    df['Sector'] = df.get('SECTOR', 'Fleet')
    if 'Sector' not in df.columns or df['Sector'].isna().all():
        df['Sector'] = 'Fleet'

    df['FillDate'] = pd.to_datetime(df['FillDate'])
    df['Year'] = df['FillDate'].dt.year.astype(str)
    df['Month'] = df['FillDate'].dt.month
    df['FillDate'] = df['FillDate'].dt.date

    group_cols = [
        'ACCT_ID', 'Vehicle Class', 'LowOrg', 'Vehicle Location',
        'Subtype', 'FillDate', 'Unit', 'Sector', 'Year', 'Month'
    ]

    # Filter to only columns that exist
    existing_group_cols = [c for c in group_cols if c in df.columns]

    df_agg = df.groupby(existing_group_cols, as_index=False)['Consumption'].sum()
    return df_agg


def calculate_emissions(df_agg: pd.DataFrame, gwp_dict: dict, ef_dict: dict) -> pd.DataFrame:
    """
    Expand GHG rows and calculate mtCO2e.

    Args:
        df_agg: Aggregated fleet data DataFrame
        gwp_dict: GWP lookup dictionary
        ef_dict: Emission factor lookup dictionary

    Returns:
        DataFrame with calculated emissions
    """
    ghg_list = ['CO2', 'CH4', 'N2O']
    df_expanded = pd.concat([df_agg.assign(GHG=g) for g in ghg_list], ignore_index=True)

    df_expanded["GHG_MTperUnit"] = df_expanded.apply(
        lambda r: ef_dict.get((r["Subtype"], r["GHG"], r["Sector"]), None),
        axis=1
    )
    df_expanded["GWP"] = df_expanded["GHG"].map(gwp_dict)

    df_expanded["mtCO2e_calc"] = (
        df_expanded["Consumption"] *
        df_expanded["GHG_MTperUnit"] *
        df_expanded["GWP"]
    )

    final_cols = [
        "ACCT_ID", "Vehicle Location", "Vehicle Class", "FillDate",
        "Subtype", "GHG", "Consumption", "GHG_MTperUnit", "GWP", "mtCO2e_calc",
        "Year", "Month", "LowOrg", "Unit", "Sector"
    ]

    # Filter to only columns that exist
    existing_cols = [c for c in final_cols if c in df_expanded.columns]
    return df_expanded[existing_cols]


def load_reference_data() -> tuple[dict, dict]:
    """
    Load GWP and EF dictionaries from EFID_Reference.csv.

    Returns:
        Tuple of (gwp_dict, ef_dict)
    """
    efid_df = read_raw_file("efid")

    # Create GWP DataFrame (unique GHG -> GWP mapping)
    gwp_df = efid_df[['GHG', 'GWP']].drop_duplicates()
    gwp_df = gwp_df.rename(columns={'GHG': 'GHG_Name', 'GWP': 'AR5_GWP'})

    # EF DataFrame is the EFID reference itself
    ef_df = efid_df

    return create_lookup_dicts(gwp_df, ef_df, gwp_version="AR5_GWP")


def run_calculation_pipeline(session_id: str) -> pd.DataFrame:
    """
    Run the full calculation pipeline.

    1. Load session data (or raw if no session)
    2. Load reference data
    3. Preprocess fleet data
    4. Calculate emissions
    5. Save results

    Args:
        session_id: The session identifier

    Returns:
        DataFrame with calculated emissions
    """
    # Load session data or raw data
    df = read_session_data(session_id, "fleetfuel")
    if df is None:
        df = read_raw_file("fleetfuel")

    # Load reference data
    gwp_dict, ef_dict = load_reference_data()

    # Preprocess fleet data
    df_preprocessed = preprocess_fleet_data(df)

    # Calculate emissions
    df_calculated = calculate_emissions(df_preprocessed, gwp_dict, ef_dict)

    # Save results
    save_calculated_data(session_id, df_calculated)

    return df_calculated


def prepare_chart_data(df: pd.DataFrame) -> dict:
    """
    Transform calculated data to chart-ready format.

    Args:
        df: DataFrame with calculated emissions

    Returns:
        Dictionary with pieChart and barChart data
    """
    # Pie chart: group by Subtype
    pie_grouped = df.groupby('Subtype')['mtCO2e_calc'].sum()
    total = pie_grouped.sum()

    pie_data = [
        {
            "label": subtype,
            "value": round(value, 5),
            "percentage": round((value / total) * 100, 1) if total != 0 else 0
        }
        for subtype, value in pie_grouped.items()
    ]

    # Bar chart: group by FillDate
    bar_grouped = df.groupby('FillDate')['mtCO2e_calc'].sum()

    bar_data = {
        "title": "mtCO2e over FillDate",
        "type": "bar",
        "labels": [str(d) for d in bar_grouped.index],
        "datasets": [{
            "label": "mtCO2e",
            "data": [round(v, 5) for v in bar_grouped.values]
        }]
    }

    return {
        "pieChart": {
            "title": "mtCO2e by Subtype",
            "type": "pie",
            "data": pie_data
        },
        "barChart": bar_data
    }


def add_formula_references(df: pd.DataFrame) -> list[dict]:
    """
    Add formulaReference to each record for transparency.

    Args:
        df: DataFrame with calculated emissions

    Returns:
        List of records with formulaReference added
    """
    records = []

    for _, row in df.iterrows():
        record = row.to_dict()

        # Convert date objects to strings for JSON serialization
        if 'FillDate' in record:
            record['FillDate'] = str(record['FillDate'])

        # Add formula reference
        consumption = record.get('Consumption', 0)
        ef = record.get('GHG_MTperUnit', 0)
        gwp = record.get('GWP', 0)
        result = record.get('mtCO2e_calc', 0)

        # Create EF_ID from Subtype and GHG
        subtype = record.get('Subtype', '')
        ghg = record.get('GHG', '')
        ef_id = f"Fuel_{subtype}_20XX_{ghg}"

        record['EF_ID'] = ef_id
        record['formulaReference'] = {
            "formula": "mtCO2e = Consumption x GHG_MTperUnit x GWP",
            "calculation": f"{consumption} x {ef} x {gwp} = {result}",
            "efidSource": ef_id
        }

        records.append(record)

    return records
