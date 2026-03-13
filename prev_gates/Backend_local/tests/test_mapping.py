"""
Unit tests for column mapping functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.mapping_engine import MappingEngine


class TestDirectColumnMapping:
    """Tests for direct column mapping."""

    def test_map_direct_column(self):
        """Test direct column mapping copies values correctly."""
        df = pd.DataFrame({"EQ_EQUIP_NO": ["1011", "1012", "1013"]})
        mapping = {"ACCT_ID": {"sourceColumn": "EQ_EQUIP_NO"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert "ACCT_ID" in result.columns
        assert result["ACCT_ID"].iloc[0] == "1011"
        assert result["ACCT_ID"].iloc[1] == "1012"
        assert len(result) == 3

    def test_map_multiple_direct_columns(self):
        """Test mapping multiple columns at once."""
        df = pd.DataFrame({
            "EQ_EQUIP_NO": ["1011", "1012"],
            "QTY_FUEL": [25.5, 30.0],
            "FUEL_TYPE": ["Gasoline", "Diesel"]
        })
        mapping = {
            "ACCT_ID": {"sourceColumn": "EQ_EQUIP_NO"},
            "Consumption": {"sourceColumn": "QTY_FUEL"},
            "Subtype": {"sourceColumn": "FUEL_TYPE"}
        }

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert "ACCT_ID" in result.columns
        assert "Consumption" in result.columns
        assert "Subtype" in result.columns
        assert result["Consumption"].iloc[0] == 25.5


class TestStaticValueMapping:
    """Tests for static value mapping."""

    def test_map_static_value(self):
        """Test static value assignment."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        mapping = {"Unit": {"staticValue": "gal"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["Unit"].iloc[0] == "gal"
        assert result["Unit"].iloc[1] == "gal"
        assert result["Unit"].iloc[2] == "gal"

    def test_map_static_numeric_value(self):
        """Test static numeric value assignment."""
        df = pd.DataFrame({"col": [1, 2]})
        mapping = {"GWP": {"staticValue": 28}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["GWP"].iloc[0] == 28


class TestDerivedValueMapping:
    """Tests for derived value mapping (date extraction)."""

    def test_map_derived_year(self):
        """Test year extraction from date."""
        df = pd.DataFrame({"FTK_DATE": ["2021-10-15", "2022-03-20", "2023-01-05"]})
        mapping = {"Year": {"derivedFrom": "FTK_DATE", "extractType": "year"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["Year"].iloc[0] == 2021
        assert result["Year"].iloc[1] == 2022
        assert result["Year"].iloc[2] == 2023

    def test_map_derived_month(self):
        """Test month extraction from date."""
        df = pd.DataFrame({"FTK_DATE": ["2021-10-15", "2022-03-20"]})
        mapping = {"Month": {"derivedFrom": "FTK_DATE", "extractType": "month"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["Month"].iloc[0] == 10
        assert result["Month"].iloc[1] == 3

    def test_map_derived_day(self):
        """Test day extraction from date."""
        df = pd.DataFrame({"FTK_DATE": ["2021-10-15", "2022-03-20"]})
        mapping = {"Day": {"derivedFrom": "FTK_DATE", "extractType": "day"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["Day"].iloc[0] == 15
        assert result["Day"].iloc[1] == 20

    def test_map_derived_quarter(self):
        """Test quarter extraction from date."""
        df = pd.DataFrame({"FTK_DATE": ["2021-01-15", "2021-05-20", "2021-09-10"]})
        mapping = {"Quarter": {"derivedFrom": "FTK_DATE", "extractType": "quarter"}}

        engine = MappingEngine()
        result, _, _ = engine.map_columns(df, mapping)

        assert result["Quarter"].iloc[0] == 1
        assert result["Quarter"].iloc[1] == 2
        assert result["Quarter"].iloc[2] == 3


class TestGHGExpansion:
    """Tests for GHG row expansion."""

    def test_expand_ghg_rows_single(self):
        """Test GHG expansion with single gas."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011"],
            "Consumption": [25.5]
        })
        config = {"ghgType": ["CO2"]}

        result = MappingEngine.expand_ghg_rows(df, config)

        assert len(result) == 1
        assert result["GHG"].iloc[0] == "CO2"

    def test_expand_ghg_rows_multiple(self):
        """Test GHG expansion with multiple gases."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011"],
            "Consumption": [25.5]
        })
        config = {"ghgType": ["CO2", "CH4", "N2O"]}

        result = MappingEngine.expand_ghg_rows(df, config)

        assert len(result) == 3
        assert set(result["GHG"]) == {"CO2", "CH4", "N2O"}
        # Each row should have the same consumption
        assert all(result["Consumption"] == 25.5)

    def test_expand_ghg_rows_preserves_data(self):
        """Test that GHG expansion preserves all other columns."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011", "1012"],
            "Consumption": [25.5, 30.0],
            "Subtype": ["Gasoline", "Diesel"]
        })
        config = {"ghgType": ["CO2", "CH4"]}

        result = MappingEngine.expand_ghg_rows(df, config)

        assert len(result) == 4  # 2 rows × 2 GHGs
        assert "ACCT_ID" in result.columns
        assert "Consumption" in result.columns
        assert "Subtype" in result.columns

    def test_expand_ghg_rows_no_config(self):
        """Test default GHG assignment when no config provided."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011"],
            "Consumption": [25.5]
        })

        result = MappingEngine.expand_ghg_rows(df, None)

        assert len(result) == 1
        assert result["GHG"].iloc[0] == "CO2"


class TestAggregation:
    """Tests for consumption aggregation."""

    def test_aggregate_consumption(self):
        """Test basic consumption aggregation."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011", "1011", "1011"],
            "Subtype": ["Gasoline", "Gasoline", "Gasoline"],
            "GHG": ["CO2", "CO2", "CO2"],
            "Consumption": [10.5, 15.0, 8.3]
        })

        result = MappingEngine.aggregate_consumption(df)

        assert len(result) == 1
        assert abs(result["Consumption"].iloc[0] - 33.8) < 0.001

    def test_aggregate_consumption_by_group(self):
        """Test aggregation respects grouping columns."""
        df = pd.DataFrame({
            "ACCT_ID": ["1011", "1011", "1012"],
            "Subtype": ["Gasoline", "Gasoline", "Gasoline"],
            "GHG": ["CO2", "CO2", "CO2"],
            "Consumption": [10.0, 15.0, 20.0]
        })

        result = MappingEngine.aggregate_consumption(df)

        assert len(result) == 2
        # ACCT_ID 1011 should have 25.0
        row_1011 = result[result["ACCT_ID"] == "1011"]
        assert abs(row_1011["Consumption"].iloc[0] - 25.0) < 0.001
        # ACCT_ID 1012 should have 20.0
        row_1012 = result[result["ACCT_ID"] == "1012"]
        assert abs(row_1012["Consumption"].iloc[0] - 20.0) < 0.001


class TestPatternMapping:
    """Tests for pattern-based column construction."""

    def test_pattern_without_ghg(self):
        """Test pattern construction that doesn't use GHG."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline", "Diesel"],
            "YEAR": [2021, 2022]
        })
        mapping = {
            "Subtype": {"sourceColumn": "FUEL_TYPE"},
            "Year": {"sourceColumn": "YEAR"},
            "EF_ID": {"pattern": "Fuel_{Subtype}_{Year}"}
        }

        engine = MappingEngine()
        result, _, deferred = engine.map_columns(df, mapping)

        # Pattern without GHG should be processed immediately
        assert len(deferred) == 0
        assert "EF_ID" in result.columns
        assert result["EF_ID"].iloc[0] == "Fuel_Gasoline_2021"
        assert result["EF_ID"].iloc[1] == "Fuel_Diesel_2022"

    def test_pattern_with_ghg_is_deferred(self):
        """Test that patterns containing {GHG} are deferred."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline"],
            "YEAR": [2021]
        })
        mapping = {
            "Subtype": {"sourceColumn": "FUEL_TYPE"},
            "Year": {"sourceColumn": "YEAR"},
            "EF_ID": {"pattern": "Fuel_{Subtype}_20XX_{GHG}"}
        }

        engine = MappingEngine()
        result, _, deferred = engine.map_columns(df, mapping)

        # Pattern with GHG should be deferred
        assert "EF_ID" in deferred
        assert deferred["EF_ID"] == "Fuel_{Subtype}_20XX_{GHG}"
        assert "EF_ID" not in result.columns  # Not processed yet

    def test_deferred_pattern_applied_after_ghg_expansion(self):
        """Test full pipeline with deferred GHG pattern."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline"],
            "QTY": [100.0]
        })
        config = {
            "columnMappings": {
                "Subtype": {"sourceColumn": "FUEL_TYPE"},
                "Consumption": {"sourceColumn": "QTY"},
                "EF_ID": {"pattern": "Fuel_{Subtype}_20XX_{GHG}"},
                "GHG": {"ghgType": ["CO2", "CH4", "N2O"]}
            }
        }

        engine = MappingEngine()
        result = engine.process_mappings(df, config)

        # Should have 3 rows (one per GHG)
        assert len(result) == 3

        # Each row should have unique EF_ID with correct GHG
        ef_ids = result["EF_ID"].tolist()
        assert "Fuel_Gasoline_20XX_CO2" in ef_ids
        assert "Fuel_Gasoline_20XX_CH4" in ef_ids
        assert "Fuel_Gasoline_20XX_N2O" in ef_ids

    def test_apply_deferred_patterns(self):
        """Test apply_deferred_patterns method directly."""
        df = pd.DataFrame({
            "Subtype": ["Gasoline", "Diesel"],
            "GHG": ["CO2", "CH4"],
            "Year": [2021, 2022]
        })
        deferred = {
            "EF_ID": "Fuel_{Subtype}_20XX_{GHG}"
        }

        result = MappingEngine.apply_deferred_patterns(df, deferred)

        assert "EF_ID" in result.columns
        assert result["EF_ID"].iloc[0] == "Fuel_Gasoline_20XX_CO2"
        assert result["EF_ID"].iloc[1] == "Fuel_Diesel_20XX_CH4"


class TestFilters:
    """Tests for filter application."""

    def test_filter_equals(self):
        """Test equals filter."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline", "Diesel", "B20"],
            "QTY_FUEL": [10, 20, 30]
        })
        filters = [{"column": "FUEL_TYPE", "operator": "=", "value": "Gasoline"}]

        engine = MappingEngine()
        result = engine.apply_filters(df, filters)

        assert len(result) == 1
        assert result["FUEL_TYPE"].iloc[0] == "Gasoline"

    def test_filter_greater_than(self):
        """Test greater than filter."""
        df = pd.DataFrame({
            "QTY_FUEL": [5, 10, 15, 20]
        })
        filters = [{"column": "QTY_FUEL", "operator": ">", "value": 10}]

        engine = MappingEngine()
        result = engine.apply_filters(df, filters)

        assert len(result) == 2
        assert all(result["QTY_FUEL"] > 10)

    def test_filter_in_list(self):
        """Test in list filter."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline", "Diesel", "B20", "B5"]
        })
        filters = [{"column": "FUEL_TYPE", "operator": "in", "value": ["Gasoline", "Diesel"]}]

        engine = MappingEngine()
        result = engine.apply_filters(df, filters)

        assert len(result) == 2
        assert set(result["FUEL_TYPE"]) == {"Gasoline", "Diesel"}

    def test_filter_not_null(self):
        """Test not null filter."""
        df = pd.DataFrame({
            "QTY_FUEL": [10, None, 20, None, 30]
        })
        filters = [{"column": "QTY_FUEL", "operator": "not_null"}]

        engine = MappingEngine()
        result = engine.apply_filters(df, filters)

        assert len(result) == 3
        assert result["QTY_FUEL"].isna().sum() == 0

    def test_multiple_filters(self):
        """Test combining multiple filters."""
        df = pd.DataFrame({
            "FUEL_TYPE": ["Gasoline", "Gasoline", "Diesel", "Diesel"],
            "QTY_FUEL": [5, 15, 10, 20]
        })
        filters = [
            {"column": "FUEL_TYPE", "operator": "=", "value": "Gasoline"},
            {"column": "QTY_FUEL", "operator": ">", "value": 10}
        ]

        engine = MappingEngine()
        result = engine.apply_filters(df, filters)

        assert len(result) == 1
        assert result["FUEL_TYPE"].iloc[0] == "Gasoline"
        assert result["QTY_FUEL"].iloc[0] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
