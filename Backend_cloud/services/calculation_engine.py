"""
Emissions calculation engine for EcoMetrics Backend.

Handles EF_ID construction, emission factor lookup, GWP lookup,
and final emissions calculation.
"""
from typing import Optional

import pandas as pd

from config import Config
from .storage import StorageService
from .mapping_engine import MappingEngine


class CalculationEngine:
    """Handles emissions calculations."""

    def __init__(self):
        self.mapping_engine = MappingEngine()
        self.efid_df: Optional[pd.DataFrame] = None
        self.gwp_df: Optional[pd.DataFrame] = None

    def load_reference_tables(self):
        """Load EFID and GWP reference tables."""
        if self.efid_df is None:
            try:
                self.efid_df = StorageService.load_reference_table("EFID")
            except FileNotFoundError:
                raise ValueError("EFID reference table not found. Please add EFID.xlsx to data/raw/REFERENCE/")

        if self.gwp_df is None:
            try:
                self.gwp_df = StorageService.load_reference_table("GWP")
            except FileNotFoundError:
                raise ValueError("GWP reference table not found. Please add GWPs.xlsx to data/raw/REFERENCE/")

    @staticmethod
    def construct_efid(df: pd.DataFrame, efid_config: dict) -> pd.DataFrame:
        """
        Construct EF_ID column based on pattern.

        Args:
            df: DataFrame with standard schema + GHG
            efid_config: Config with pattern and strategy

        Returns:
            DataFrame with EF_ID column
        """
        df = df.copy()
        pattern = efid_config.get("pattern", "Fuel_{Subtype}_20XX_{GHG}")

        # Start with the pattern
        df["EF_ID"] = pattern

        # Replace {Subtype}
        if "{Subtype}" in pattern:
            df["EF_ID"] = df.apply(
                lambda row: row["EF_ID"].replace("{Subtype}", str(row["Subtype"])),
                axis=1
            )

        # Replace {GHG}
        if "{GHG}" in pattern:
            df["EF_ID"] = df.apply(
                lambda row: row["EF_ID"].replace("{GHG}", str(row["GHG"])),
                axis=1
            )

        # Replace {Year}
        if "{Year}" in pattern:
            df["EF_ID"] = df.apply(
                lambda row: row["EF_ID"].replace("{Year}", str(int(row["Year"]))),
                axis=1
            )

        # Replace {Sector}
        if "{Sector}" in pattern and "Sector" in df.columns:
            df["EF_ID"] = df.apply(
                lambda row: row["EF_ID"].replace("{Sector}", str(row["Sector"])),
                axis=1
            )

        return df

    def join_emission_factors(
        self,
        df: pd.DataFrame,
        efid_config: dict,
    ) -> pd.DataFrame:
        """
        Join with EFID reference table to get GHG_MTperUnit.

        Args:
            df: DataFrame with EF_ID column
            efid_config: Config with sector filter

        Returns:
            DataFrame with GHG_MTperUnit column
        """
        self.load_reference_tables()

        efid_df = self.efid_df.copy()

        # Filter EFID table by sector if specified
        sector_filter = efid_config.get("sectorFilter")
        if sector_filter and "Sector" in efid_df.columns:
            efid_df = efid_df[efid_df["Sector"] == sector_filter]

        # Join on EF_ID
        df = df.merge(
            efid_df[["EF_ID", "GHG_MTperUnit"]],
            on="EF_ID",
            how="left"
        )

        # Check for missing emission factors
        missing_ef = df[df["GHG_MTperUnit"].isna()]["EF_ID"].unique()
        if len(missing_ef) > 0:
            raise ValueError(f"Missing emission factors for: {missing_ef.tolist()}")

        return df

    def join_gwp(
        self,
        df: pd.DataFrame,
        gwp_version: str = "AR5",
    ) -> pd.DataFrame:
        """
        Join with GWP reference table to get global warming potential.

        Args:
            df: DataFrame with GHG column
            gwp_version: 'AR5' or 'AR4'

        Returns:
            DataFrame with GWP column
        """
        self.load_reference_tables()

        gwp_column = f"{gwp_version}_GWP"

        # Check if the column exists
        if gwp_column not in self.gwp_df.columns:
            available_cols = [c for c in self.gwp_df.columns if "GWP" in c]
            raise ValueError(f"GWP column '{gwp_column}' not found. Available: {available_cols}")

        # Create lookup dictionary
        gwp_dict = self.gwp_df.set_index("GHG_Name")[gwp_column].to_dict()

        # Map GWP values
        df = df.copy()
        df["GWP"] = df["GHG"].map(gwp_dict)

        # Check for missing GWP values
        missing_gwp = df[df["GWP"].isna()]["GHG"].unique()
        if len(missing_gwp) > 0:
            raise ValueError(f"Missing GWP values for: {missing_gwp.tolist()}")

        return df

    @staticmethod
    def calculate_emissions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CO2-equivalent emissions.

        Formula: mtCO2e_calc = Consumption × GHG_MTperUnit × GWP

        Args:
            df: DataFrame with Consumption, GHG_MTperUnit, and GWP

        Returns:
            DataFrame with mtCO2e_calc column
        """
        df = df.copy()

        df["mtCO2e_calc"] = (
            df["Consumption"] *
            df["GHG_MTperUnit"] *
            df["GWP"]
        )

        # Add formula reference for transparency
        df["formulaReference"] = df.apply(
            lambda row: {
                "formula": "mtCO2e = Consumption × GHG_MTperUnit × GWP",
                "calculation": f"{row['Consumption']} × {row['GHG_MTperUnit']} × {row['GWP']} = {row['mtCO2e_calc']}",
                "efidSource": row["EF_ID"]
            },
            axis=1
        )

        return df

    def process_source(
        self,
        source_id: str,
        config: dict,
        gwp_version: str = "AR5",
    ) -> pd.DataFrame:
        """
        Complete processing pipeline for a single source.

        Args:
            source_id: Source data identifier
            config: Column mapping configuration
            gwp_version: GWP version to use

        Returns:
            DataFrame with calculated emissions
        """
        # 1. Load raw data
        df_raw = StorageService.load_raw_data(source_id, config.get("sourceSheet"))

        # 2. Apply mappings (filter, map, expand, aggregate)
        # EF_ID is now a required column mapping - no longer auto-constructed
        df_mapped = self.mapping_engine.process_mappings(df_raw, config)

        # 3. Validate EF_ID column exists
        if "EF_ID" not in df_mapped.columns:
            raise ValueError("EF_ID column is missing. Ensure EF_ID is mapped in columnMappings.")

        # 4. Join emission factors
        efid_config = config.get("efidLookup", {})
        df_with_ef = self.join_emission_factors(df_mapped, efid_config)

        # 5. Join GWP values
        df_with_gwp = self.join_gwp(df_with_ef, gwp_version)

        # 6. Calculate emissions
        df_final = self.calculate_emissions(df_with_gwp)

        return df_final

    def calculate_for_session(
        self,
        session_id: str,
        source_ids: list[str],
        gwp_version: str = "AR5",
    ) -> dict:
        """
        Execute emissions calculations for multiple sources in a session.

        Args:
            session_id: Session identifier
            source_ids: List of source IDs to calculate
            gwp_version: GWP version to use

        Returns:
            Dictionary with calculation results
        """
        from datetime import datetime

        calculation_id = StorageService.generate_calculation_id(session_id)
        started_at = datetime.utcnow().isoformat() + "Z"

        results = {}
        total_mtco2e = 0.0
        all_years = []

        for source_id in source_ids:
            # Get config for this source
            config = StorageService.get_source_config(session_id, source_id)
            if config is None:
                raise ValueError(f"No configuration found for source: {source_id}")

            # Process the source
            df = self.process_source(source_id, config, gwp_version)

            # Save calculated data
            file_path = StorageService.save_calculated_data(session_id, source_id, df)

            # Collect statistics
            source_total = df["mtCO2e_calc"].sum()
            total_mtco2e += source_total

            if "Year" in df.columns:
                all_years.extend(df["Year"].dropna().unique().tolist())

            results[source_id] = {
                "inputRows": len(StorageService.load_raw_data(source_id, config.get("sourceSheet"))),
                "outputRows": len(df),
                "totalMtCO2e": round(source_total, 6),
                "filePath": file_path,
            }

        completed_at = datetime.utcnow().isoformat() + "Z"

        # Update session metadata
        session = StorageService.get_session(session_id)
        session["calculations"].append({
            "calculationId": calculation_id,
            "timestamp": completed_at,
            "gwpVersion": gwp_version,
            "totalMtCO2e": round(total_mtco2e, 6),
        })

        for source in session.get("sources", []):
            if source["sourceId"] in source_ids:
                source["hasCalculation"] = True
                source["calculationId"] = calculation_id
                source["totalMtCO2e"] = results[source["sourceId"]]["totalMtCO2e"]

        StorageService.update_session(session_id, {
            "sources": session["sources"],
            "calculations": session["calculations"],
            "hasCalculations": True,
        })

        # Determine year range
        year_range = {}
        if all_years:
            all_years = [int(y) for y in all_years if pd.notna(y)]
            if all_years:
                year_range = {"min": min(all_years), "max": max(all_years)}

        return {
            "calculationId": calculation_id,
            "status": "completed",
            "startedAt": started_at,
            "completedAt": completed_at,
            "results": results,
            "aggregated": {
                "totalMtCO2e": round(total_mtco2e, 6),
                "totalSources": len(source_ids),
                "yearRange": year_range,
            },
        }
