"""
Unit tests for emissions calculation functions.
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.calculation_engine import CalculationEngine


class TestEFIDConstruction:
    """Tests for EF_ID construction from patterns."""

    def test_construct_efid_basic(self):
        """Test basic EF_ID construction with Subtype and GHG."""
        df = pd.DataFrame({
            "Subtype": ["Gasoline", "Diesel"],
            "GHG": ["CO2", "CH4"]
        })
        efid_config = {"pattern": "Fuel_{Subtype}_20XX_{GHG}"}

        result = CalculationEngine.construct_efid(df, efid_config)

        assert result["EF_ID"].iloc[0] == "Fuel_Gasoline_20XX_CO2"
        assert result["EF_ID"].iloc[1] == "Fuel_Diesel_20XX_CH4"

    def test_construct_efid_with_year(self):
        """Test EF_ID construction with Year placeholder."""
        df = pd.DataFrame({
            "Subtype": ["PSE"],
            "GHG": ["CO2"],
            "Year": [2022]
        })
        efid_config = {"pattern": "Elec_{Subtype}_{Year}_{GHG}"}

        result = CalculationEngine.construct_efid(df, efid_config)

        assert result["EF_ID"].iloc[0] == "Elec_PSE_2022_CO2"

    def test_construct_efid_with_sector(self):
        """Test EF_ID construction with Sector placeholder."""
        df = pd.DataFrame({
            "Subtype": ["R-134A"],
            "GHG": ["R-134A"],
            "Sector": ["Facilities"]
        })
        efid_config = {"pattern": "FacAC_{Subtype}_{Sector}"}

        result = CalculationEngine.construct_efid(df, efid_config)

        assert result["EF_ID"].iloc[0] == "FacAC_R-134A_Facilities"

    def test_construct_efid_default_pattern(self):
        """Test EF_ID construction with default pattern."""
        df = pd.DataFrame({
            "Subtype": ["B20"],
            "GHG": ["N2O"]
        })
        efid_config = {}  # No pattern specified

        result = CalculationEngine.construct_efid(df, efid_config)

        assert result["EF_ID"].iloc[0] == "Fuel_B20_20XX_N2O"


class TestEmissionsCalculation:
    """Tests for emissions calculation formula."""

    def test_calculate_emissions_basic(self):
        """Test basic emissions calculation."""
        df = pd.DataFrame({
            "Consumption": [25.5],
            "GHG_MTperUnit": [0.008887],
            "GWP": [1],
            "EF_ID": ["Fuel_Gasoline_20XX_CO2"]
        })

        result = CalculationEngine.calculate_emissions(df)

        assert "mtCO2e_calc" in result.columns
        # 25.5 × 0.008887 × 1 = 0.2266185
        assert abs(result["mtCO2e_calc"].iloc[0] - 0.2266185) < 0.0001

    def test_calculate_emissions_with_gwp(self):
        """Test emissions calculation with GWP multiplier."""
        df = pd.DataFrame({
            "Consumption": [25.5],
            "GHG_MTperUnit": [2.27797e-07],
            "GWP": [28],
            "EF_ID": ["Fuel_Gasoline_20XX_CH4"]
        })

        result = CalculationEngine.calculate_emissions(df)

        # 25.5 × 2.27797e-07 × 28 = 0.000162657
        expected = 25.5 * 2.27797e-07 * 28
        assert abs(result["mtCO2e_calc"].iloc[0] - expected) < 1e-10

    def test_calculate_emissions_formula_reference(self):
        """Test that formula reference is added."""
        df = pd.DataFrame({
            "Consumption": [100],
            "GHG_MTperUnit": [0.01],
            "GWP": [1],
            "EF_ID": ["Test_EF_ID"]
        })

        result = CalculationEngine.calculate_emissions(df)

        assert "formulaReference" in result.columns
        formula_ref = result["formulaReference"].iloc[0]
        assert formula_ref["formula"] == "mtCO2e = Consumption × GHG_MTperUnit × GWP"
        assert "efidSource" in formula_ref
        assert formula_ref["efidSource"] == "Test_EF_ID"

    def test_calculate_emissions_multiple_rows(self):
        """Test emissions calculation for multiple rows."""
        df = pd.DataFrame({
            "Consumption": [100, 200, 300],
            "GHG_MTperUnit": [0.01, 0.01, 0.01],
            "GWP": [1, 28, 265],
            "EF_ID": ["EF1", "EF2", "EF3"]
        })

        result = CalculationEngine.calculate_emissions(df)

        # Row 1: 100 × 0.01 × 1 = 1
        assert abs(result["mtCO2e_calc"].iloc[0] - 1.0) < 0.001
        # Row 2: 200 × 0.01 × 28 = 56
        assert abs(result["mtCO2e_calc"].iloc[1] - 56.0) < 0.001
        # Row 3: 300 × 0.01 × 265 = 795
        assert abs(result["mtCO2e_calc"].iloc[2] - 795.0) < 0.001

    def test_calculate_emissions_negative_consumption(self):
        """Test emissions calculation with negative consumption (e.g., B20 biofuel offset)."""
        df = pd.DataFrame({
            "Consumption": [-16.8],
            "GHG_MTperUnit": [0.008887],
            "GWP": [1],
            "EF_ID": ["Fuel_B20_20XX_CO2"]
        })

        result = CalculationEngine.calculate_emissions(df)

        # Should produce negative emissions (offset)
        assert result["mtCO2e_calc"].iloc[0] < 0


class TestCalculationPipeline:
    """Integration tests for the calculation pipeline."""

    def test_full_gasoline_calculation(self):
        """Test complete calculation for gasoline combustion."""
        # This simulates the example from the spec:
        # 25.5 gallons of gasoline → CO2, CH4, N2O emissions

        consumption = 25.5

        # CO2 calculation
        df_co2 = pd.DataFrame({
            "Consumption": [consumption],
            "GHG_MTperUnit": [0.008887],
            "GWP": [1],
            "EF_ID": ["Fuel_Gasoline_20XX_CO2"]
        })
        result_co2 = CalculationEngine.calculate_emissions(df_co2)
        mtco2e_co2 = result_co2["mtCO2e_calc"].iloc[0]

        # CH4 calculation
        df_ch4 = pd.DataFrame({
            "Consumption": [consumption],
            "GHG_MTperUnit": [2.27797e-07],
            "GWP": [28],
            "EF_ID": ["Fuel_Gasoline_20XX_CH4"]
        })
        result_ch4 = CalculationEngine.calculate_emissions(df_ch4)
        mtco2e_ch4 = result_ch4["mtCO2e_calc"].iloc[0]

        # N2O calculation
        df_n2o = pd.DataFrame({
            "Consumption": [consumption],
            "GHG_MTperUnit": [1.11899e-07],
            "GWP": [265],
            "EF_ID": ["Fuel_Gasoline_20XX_N2O"]
        })
        result_n2o = CalculationEngine.calculate_emissions(df_n2o)
        mtco2e_n2o = result_n2o["mtCO2e_calc"].iloc[0]

        # Total emissions
        total = mtco2e_co2 + mtco2e_ch4 + mtco2e_n2o

        # Expected values from spec:
        # CO2: 0.226618
        # CH4: 0.000163
        # N2O: 0.000755
        # Total: ~0.227536

        assert abs(mtco2e_co2 - 0.226618) < 0.001
        assert abs(total - 0.227536) < 0.001


class TestGWPValues:
    """Tests for GWP value handling."""

    def test_standard_gwp_values(self):
        """Test standard AR5 GWP values are used correctly."""
        standard_gwp = {
            "CO2": 1,
            "CH4": 28,
            "N2O": 265,
            "R-410A": 1924,
            "R-134A": 1300,
        }

        for ghg, gwp in standard_gwp.items():
            df = pd.DataFrame({
                "Consumption": [100],
                "GHG_MTperUnit": [0.001],
                "GWP": [gwp],
                "EF_ID": [f"Test_{ghg}"]
            })

            result = CalculationEngine.calculate_emissions(df)

            expected = 100 * 0.001 * gwp
            assert abs(result["mtCO2e_calc"].iloc[0] - expected) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
