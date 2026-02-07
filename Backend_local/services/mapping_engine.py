"""
Column mapping engine for EcoMetrics Backend.

Handles the transformation of raw data to standard schema based on
user-defined column mappings.
"""
from typing import Optional

import pandas as pd

from .storage import StorageService


class MappingEngine:
    """Handles data mapping transformations."""

    def __init__(self):
        self.reference_tables_cache: dict[str, pd.DataFrame] = {}

    def apply_filters(self, df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
        """
        Apply user-defined filters to raw data.

        Args:
            df: Raw DataFrame
            filters: List of filter configurations

        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()

        for filter_config in filters:
            column = filter_config["column"]
            operator = filter_config["operator"]
            value = filter_config.get("value")

            # Handle date columns
            if df_filtered[column].dtype == "object":
                try:
                    df_filtered[column] = pd.to_datetime(df_filtered[column])
                    if isinstance(value, str):
                        value = pd.to_datetime(value)
                except (ValueError, TypeError):
                    pass

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
            elif operator == "is_null":
                df_filtered = df_filtered[df_filtered[column].isna()]
            elif operator == "not_null":
                df_filtered = df_filtered[df_filtered[column].notna()]
            else:
                raise ValueError(f"Unsupported operator: {operator}")

        return df_filtered

    def map_columns(
        self,
        df: pd.DataFrame,
        column_mappings: dict,
    ) -> tuple[pd.DataFrame, Optional[dict]]:
        """
        Map raw columns to standard schema based on user configuration.

        Args:
            df: Filtered DataFrame
            column_mappings: Dictionary of column mapping configurations

        Returns:
            Tuple of (DataFrame with mapped columns, GHG config if present)
        """
        df_mapped = pd.DataFrame(index=df.index)
        ghg_config = None

        for target_column, mapping in column_mappings.items():
            # Handle GHG expansion separately
            if "ghgType" in mapping:
                ghg_config = mapping
                continue

            # 1. Direct column mapping
            if "sourceColumn" in mapping:
                source_col = mapping["sourceColumn"]
                df_mapped[target_column] = df[source_col].values

            # 2. Static value
            elif "staticValue" in mapping:
                df_mapped[target_column] = mapping["staticValue"]

            # 3. Derived value (date extraction)
            elif "derivedFrom" in mapping:
                source_col = mapping["derivedFrom"]
                extract_type = mapping["extractType"]

                # Convert to datetime
                date_series = pd.to_datetime(df[source_col])

                if extract_type == "year":
                    df_mapped[target_column] = date_series.dt.year
                elif extract_type == "month":
                    df_mapped[target_column] = date_series.dt.month
                elif extract_type == "day":
                    df_mapped[target_column] = date_series.dt.day
                elif extract_type == "quarter":
                    df_mapped[target_column] = date_series.dt.quarter
                elif extract_type == "dayofweek":
                    df_mapped[target_column] = date_series.dt.dayofweek
                else:
                    raise ValueError(f"Unsupported extractType: {extract_type}")

            # 4. Reference table join
            elif "referenceTable" in mapping:
                ref_table_path = mapping["referenceTable"]
                ref_key = mapping["refKey"]
                source_key = mapping["sourceKey"]
                column_joined = mapping["columnJoined"]
                ref_sheet = mapping.get("refSheet")

                # Load reference table (use cache if available)
                cache_key = f"{ref_table_path}:{ref_sheet}"
                if cache_key not in self.reference_tables_cache:
                    ref_df = StorageService.load_custom_reference_table(ref_table_path, ref_sheet)
                    self.reference_tables_cache[cache_key] = ref_df
                else:
                    ref_df = self.reference_tables_cache[cache_key]

                # Create lookup dictionary
                lookup_dict = ref_df.set_index(ref_key)[column_joined].to_dict()

                # Map values
                df_mapped[target_column] = df[source_key].map(lookup_dict)

            else:
                raise ValueError(f"Invalid mapping configuration for column: {target_column}")

        return df_mapped, ghg_config

    @staticmethod
    def expand_ghg_rows(df: pd.DataFrame, ghg_config: Optional[dict]) -> pd.DataFrame:
        """
        Expand rows for multiple GHG types.

        If user selects ["CO2", "CH4", "N2O"], each input row becomes 3 output rows.

        Args:
            df: Mapped DataFrame (without GHG column)
            ghg_config: GHG configuration with ghgType list

        Returns:
            DataFrame with GHG column and expanded rows
        """
        if ghg_config is None:
            # No GHG expansion, assign default
            df = df.copy()
            df["GHG"] = "CO2"
            return df

        ghg_types = ghg_config["ghgType"]

        # Create one DataFrame per GHG type
        dfs = []
        for ghg in ghg_types:
            df_ghg = df.copy()
            df_ghg["GHG"] = ghg
            dfs.append(df_ghg)

        # Concatenate all GHG DataFrames
        df_expanded = pd.concat(dfs, ignore_index=True)

        return df_expanded

    @staticmethod
    def aggregate_consumption(df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by all non-consumption columns and sum consumption.

        This ensures unique rows based on all dimension columns.

        Args:
            df: DataFrame with expanded GHG rows

        Returns:
            DataFrame with aggregated consumption
        """
        # Identify grouping columns (all except Consumption)
        group_cols = [col for col in df.columns if col != "Consumption"]

        if not group_cols:
            return df

        # Group and sum consumption
        df_aggregated = df.groupby(group_cols, as_index=False, dropna=False)["Consumption"].sum()

        return df_aggregated

    def process_mappings(
        self,
        df: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        """
        Apply full mapping pipeline: filters -> map -> expand -> aggregate.

        Args:
            df: Raw DataFrame
            config: Mapping configuration

        Returns:
            Processed DataFrame ready for emission factor lookup
        """
        # 1. Apply filters
        filters = config.get("filters", [])
        df_filtered = self.apply_filters(df, filters)

        # 2. Map columns
        df_mapped, ghg_config = self.map_columns(df_filtered, config["columnMappings"])

        # 3. Expand GHG rows
        df_expanded = self.expand_ghg_rows(df_mapped, ghg_config)

        # 4. Aggregate consumption
        df_aggregated = self.aggregate_consumption(df_expanded)

        return df_aggregated

    def clear_cache(self):
        """Clear the reference table cache."""
        self.reference_tables_cache.clear()
