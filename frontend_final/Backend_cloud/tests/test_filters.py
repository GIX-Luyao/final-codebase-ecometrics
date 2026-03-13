"""
Unit tests for filter operations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.mapping_engine import MappingEngine


class TestComparisonFilters:
    """Tests for comparison filter operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()
        self.df = pd.DataFrame({
            "value": [1, 5, 10, 15, 20],
            "name": ["a", "b", "c", "d", "e"]
        })

    def test_filter_equals(self):
        """Test = operator."""
        filters = [{"column": "value", "operator": "=", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 1
        assert result["value"].iloc[0] == 10

    def test_filter_not_equals(self):
        """Test != operator."""
        filters = [{"column": "value", "operator": "!=", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 4
        assert 10 not in result["value"].values

    def test_filter_greater_than(self):
        """Test > operator."""
        filters = [{"column": "value", "operator": ">", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2
        assert all(result["value"] > 10)

    def test_filter_greater_than_or_equal(self):
        """Test >= operator."""
        filters = [{"column": "value", "operator": ">=", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 3
        assert all(result["value"] >= 10)

    def test_filter_less_than(self):
        """Test < operator."""
        filters = [{"column": "value", "operator": "<", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2
        assert all(result["value"] < 10)

    def test_filter_less_than_or_equal(self):
        """Test <= operator."""
        filters = [{"column": "value", "operator": "<=", "value": 10}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 3
        assert all(result["value"] <= 10)


class TestListFilters:
    """Tests for list-based filter operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()
        self.df = pd.DataFrame({
            "fuel_type": ["Gasoline", "Diesel", "B20", "B5", "E85"],
            "quantity": [100, 200, 150, 80, 120]
        })

    def test_filter_in(self):
        """Test in operator."""
        filters = [{"column": "fuel_type", "operator": "in", "value": ["Gasoline", "Diesel"]}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2
        assert set(result["fuel_type"]) == {"Gasoline", "Diesel"}

    def test_filter_not_in(self):
        """Test not_in operator."""
        filters = [{"column": "fuel_type", "operator": "not_in", "value": ["Gasoline", "Diesel"]}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 3
        assert "Gasoline" not in result["fuel_type"].values
        assert "Diesel" not in result["fuel_type"].values


class TestNullFilters:
    """Tests for null handling filter operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()
        self.df = pd.DataFrame({
            "value": [1, None, 3, None, 5],
            "name": ["a", "b", None, "d", "e"]
        })

    def test_filter_is_null(self):
        """Test is_null operator."""
        filters = [{"column": "value", "operator": "is_null"}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2
        assert result["value"].isna().all()

    def test_filter_not_null(self):
        """Test not_null operator."""
        filters = [{"column": "value", "operator": "not_null"}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 3
        assert result["value"].notna().all()


class TestDateFilters:
    """Tests for date-based filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()
        self.df = pd.DataFrame({
            "date": ["2019-01-15", "2020-06-20", "2021-03-10", "2022-12-05"],
            "value": [100, 200, 300, 400]
        })

    def test_filter_date_greater_than_or_equal(self):
        """Test date >= filter."""
        filters = [{"column": "date", "operator": ">=", "value": "2020-01-01"}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 3  # 2020, 2021, 2022

    def test_filter_date_less_than(self):
        """Test date < filter."""
        filters = [{"column": "date", "operator": "<", "value": "2021-01-01"}]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2  # 2019, 2020

    def test_filter_date_range(self):
        """Test date range filter with multiple conditions."""
        filters = [
            {"column": "date", "operator": ">=", "value": "2020-01-01"},
            {"column": "date", "operator": "<", "value": "2022-01-01"}
        ]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2  # 2020, 2021


class TestCombinedFilters:
    """Tests for combining multiple filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()
        self.df = pd.DataFrame({
            "fuel_type": ["Gasoline", "Gasoline", "Diesel", "Diesel", "B20"],
            "quantity": [50, 150, 100, 200, 75],
            "year": [2020, 2021, 2020, 2021, 2021]
        })

    def test_combined_string_and_numeric(self):
        """Test combining string and numeric filters."""
        filters = [
            {"column": "fuel_type", "operator": "=", "value": "Gasoline"},
            {"column": "quantity", "operator": ">", "value": 100}
        ]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 1
        assert result["fuel_type"].iloc[0] == "Gasoline"
        assert result["quantity"].iloc[0] == 150

    def test_combined_list_and_comparison(self):
        """Test combining list and comparison filters."""
        filters = [
            {"column": "fuel_type", "operator": "in", "value": ["Gasoline", "Diesel"]},
            {"column": "year", "operator": "=", "value": 2021}
        ]
        result = self.engine.apply_filters(self.df, filters)

        assert len(result) == 2
        assert all(result["year"] == 2021)
        assert set(result["fuel_type"]) == {"Gasoline", "Diesel"}


class TestFilterEdgeCases:
    """Tests for edge cases in filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MappingEngine()

    def test_empty_filters(self):
        """Test with empty filter list."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        filters = []
        result = self.engine.apply_filters(df, filters)

        assert len(result) == 3

    def test_filter_results_in_empty_df(self):
        """Test filter that produces empty result."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        filters = [{"column": "value", "operator": ">", "value": 100}]
        result = self.engine.apply_filters(df, filters)

        assert len(result) == 0

    def test_filter_preserves_all_columns(self):
        """Test that filtering preserves all columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.1, 2.2, 3.3]
        })
        filters = [{"column": "a", "operator": "=", "value": 2}]
        result = self.engine.apply_filters(df, filters)

        assert list(result.columns) == ["a", "b", "c"]
        assert result["b"].iloc[0] == "y"
        assert result["c"].iloc[0] == 2.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
