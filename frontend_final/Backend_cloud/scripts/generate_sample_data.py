"""
Generate sample data files for EcoMetrics GHG Emissions testing.
Creates 20 rows per source file with diversity for testing data configuration functions.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CONSUMPTION_DIR = os.path.join(BASE_DIR, 'CONSUMPTION')
REFERENCE_DIR = os.path.join(BASE_DIR, 'REFERENCE')

os.makedirs(CONSUMPTION_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# REFERENCE TABLES
# ============================================================

def create_efid():
    """Create EFID.xlsx emission factor reference table."""
    rows = [
        # Fleet fuels
        {"EF_ID": "Fuel_Gasoline_20XX_CO2", "GHG_MTperUnit": 0.008887, "Unit": "gal", "Subtype": "Gasoline", "GHG": "CO2", "Sector": "Fleet", "GWP": 1},
        {"EF_ID": "Fuel_Gasoline_20XX_CH4", "GHG_MTperUnit": 2.27797e-07, "Unit": "gal", "Subtype": "Gasoline", "GHG": "CH4", "Sector": "Fleet", "GWP": 28},
        {"EF_ID": "Fuel_Gasoline_20XX_N2O", "GHG_MTperUnit": 1.11899e-07, "Unit": "gal", "Subtype": "Gasoline", "GHG": "N2O", "Sector": "Fleet", "GWP": 265},
        {"EF_ID": "Fuel_Diesel_20XX_CO2", "GHG_MTperUnit": 0.010180, "Unit": "gal", "Subtype": "Diesel", "GHG": "CO2", "Sector": "Fleet", "GWP": 1},
        {"EF_ID": "Fuel_Diesel_20XX_CH4", "GHG_MTperUnit": 2.26796e-07, "Unit": "gal", "Subtype": "Diesel", "GHG": "CH4", "Sector": "Fleet", "GWP": 28},
        {"EF_ID": "Fuel_Diesel_20XX_N2O", "GHG_MTperUnit": 2.26796e-07, "Unit": "gal", "Subtype": "Diesel", "GHG": "N2O", "Sector": "Fleet", "GWP": 265},
        {"EF_ID": "Fuel_B20_20XX_CO2", "GHG_MTperUnit": 0.010058, "Unit": "gal", "Subtype": "B20", "GHG": "CO2", "Sector": "Fleet", "GWP": 1},
        {"EF_ID": "Fuel_B20_20XX_CH4", "GHG_MTperUnit": 2.05183e-07, "Unit": "gal", "Subtype": "B20", "GHG": "CH4", "Sector": "Fleet", "GWP": 28},
        {"EF_ID": "Fuel_B20_20XX_N2O", "GHG_MTperUnit": 2.20270e-07, "Unit": "gal", "Subtype": "B20", "GHG": "N2O", "Sector": "Fleet", "GWP": 265},
        {"EF_ID": "Fuel_UNL_20XX_CO2", "GHG_MTperUnit": 0.008887, "Unit": "gal", "Subtype": "UNL", "GHG": "CO2", "Sector": "Fleet", "GWP": 1},
        {"EF_ID": "Fuel_UNL_20XX_CH4", "GHG_MTperUnit": 2.27797e-07, "Unit": "gal", "Subtype": "UNL", "GHG": "CH4", "Sector": "Fleet", "GWP": 28},
        {"EF_ID": "Fuel_UNL_20XX_N2O", "GHG_MTperUnit": 1.11899e-07, "Unit": "gal", "Subtype": "UNL", "GHG": "N2O", "Sector": "Fleet", "GWP": 265},
        # PSE Electricity
        {"EF_ID": "Elec_PSE_2022_CO2", "GHG_MTperUnit": 0.000415945, "Unit": "KWH", "Subtype": "PSE", "GHG": "CO2", "Sector": "Facilities", "GWP": 1},
        {"EF_ID": "Elec_PSE_2022_CH4", "GHG_MTperUnit": 2.63084e-08, "Unit": "KWH", "Subtype": "PSE", "GHG": "CH4", "Sector": "Facilities", "GWP": 28},
        {"EF_ID": "Elec_PSE_2022_N2O", "GHG_MTperUnit": 3.62874e-09, "Unit": "KWH", "Subtype": "PSE", "GHG": "N2O", "Sector": "Facilities", "GWP": 265},
        # SCL Electricity
        {"EF_ID": "Elec_SCL_2022_CO2", "GHG_MTperUnit": 1.05097e-05, "Unit": "KWH", "Subtype": "SCL", "GHG": "CO2", "Sector": "Facilities", "GWP": 1},
        {"EF_ID": "Elec_SCL_2022_CH4", "GHG_MTperUnit": 1.36078e-08, "Unit": "KWH", "Subtype": "SCL", "GHG": "CH4", "Sector": "Facilities", "GWP": 28},
        {"EF_ID": "Elec_SCL_2022_N2O", "GHG_MTperUnit": 1.81437e-09, "Unit": "KWH", "Subtype": "SCL", "GHG": "N2O", "Sector": "Facilities", "GWP": 265},
        # Facility AC
        {"EF_ID": "FacAC_ComACHP_20XX", "GHG_MTperUnit": 0.0001, "Unit": "Refrig kg", "Subtype": "Facility AC", "GHG": "R-410A", "Sector": "Facilities", "GWP": 1924},
        {"EF_ID": "FacAC_ComACSp_20XX", "GHG_MTperUnit": 0.0001, "Unit": "Refrig kg", "Subtype": "Facility AC", "GHG": "R-410A", "Sector": "Facilities", "GWP": 1924},
        {"EF_ID": "FacAC_ComACHP_R134A_20XX", "GHG_MTperUnit": 0.0001, "Unit": "Refrig kg", "Subtype": "Facility AC", "GHG": "R-134A", "Sector": "Facilities", "GWP": 1300},
        # Fleet MVAC
        {"EF_ID": "MVAC_LightDuty_20XX_R-134A", "GHG_MTperUnit": 0.0002, "Unit": "Refrig kg", "Subtype": "LightDuty", "GHG": "R-134A", "Sector": "Fleet", "GWP": 1300},
        {"EF_ID": "MVAC_HeavyDuty_20XX_R-134A", "GHG_MTperUnit": 0.0002, "Unit": "Refrig kg", "Subtype": "HeavyDuty", "GHG": "R-134A", "Sector": "Fleet", "GWP": 1300},
        {"EF_ID": "MVAC_OtherMobile_20XX_R-134A", "GHG_MTperUnit": 0.0002, "Unit": "Refrig kg", "Subtype": "OtherMobile", "GHG": "R-134A", "Sector": "Fleet", "GWP": 1300},
        # Compressed Gases
        {"EF_ID": "ComprGas_Acetylene_20XX_CO2", "GHG_MTperUnit": 0.00011, "Unit": "SCF", "Subtype": "Acetylene", "GHG": "CO2", "Sector": "Facilities", "GWP": 1},
        {"EF_ID": "ComprGas_CO2_20XX_CO2", "GHG_MTperUnit": 0.000453593, "Unit": "lbs", "Subtype": "CO2", "GHG": "CO2", "Sector": "Facilities", "GWP": 1},
        # Landfill
        {"EF_ID": "Landfill_CH4", "GHG_MTperUnit": 1.0, "Unit": "mt CH4", "Subtype": "Landfill Methane", "GHG": "CH4", "Sector": "Landfill", "GWP": 28},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(REFERENCE_DIR, 'EFID.xlsx'), index=False)
    print(f"  Created EFID.xlsx ({len(df)} rows)")


def create_gwps():
    """Create GWPs.xlsx global warming potential reference table."""
    rows = [
        {"GHG_Name": "CO2", "GHG_ID": "CO2", "AR5_GWP": 1, "AR4_GWP": 1},
        {"GHG_Name": "CH4", "GHG_ID": "CH4", "AR5_GWP": 28, "AR4_GWP": 25},
        {"GHG_Name": "N2O", "GHG_ID": "N2O", "AR5_GWP": 265, "AR4_GWP": 298},
        {"GHG_Name": "R-134A", "GHG_ID": "R-134A", "AR5_GWP": 1300, "AR4_GWP": 1430},
        {"GHG_Name": "R-410A", "GHG_ID": "R-410A", "AR5_GWP": 1924, "AR4_GWP": 2088},
        {"GHG_Name": "R-22", "GHG_ID": "R-22", "AR5_GWP": 1760, "AR4_GWP": 1810},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(REFERENCE_DIR, 'GWPs.xlsx'), index=False)
    print(f"  Created GWPs.xlsx ({len(df)} rows)")


def create_lob_loworgs():
    """Create LOB_LowOrgList.xlsx organizational hierarchy reference."""
    rows = [
        {"LowOrg": "SU086", "LOB": "Shared Services", "Division": "Admin", "LowOrg_Name": "Finance & Admin"},
        {"LowOrg": "SU087", "LOB": "Shared Services", "Division": "Admin", "LowOrg_Name": "Human Resources"},
        {"LowOrg": "SU088", "LOB": "Water", "Division": "Water Operations", "LowOrg_Name": "Water Treatment"},
        {"LowOrg": "SU089", "LOB": "Water", "Division": "Water Operations", "LowOrg_Name": "Water Distribution"},
        {"LowOrg": "SU090", "LOB": "Water", "Division": "Water Operations", "LowOrg_Name": "Water Quality"},
        {"LowOrg": "SU091", "LOB": "Water", "Division": "Field Operations", "LowOrg_Name": "Water Maintenance"},
        {"LowOrg": "SU092", "LOB": "Water", "Division": "Field Operations", "LowOrg_Name": "Water Construction"},
        {"LowOrg": "SU093", "LOB": "Drainage", "Division": "Drainage Ops", "LowOrg_Name": "Drainage Maintenance"},
        {"LowOrg": "SU094", "LOB": "Drainage", "Division": "Drainage Ops", "LowOrg_Name": "Drainage Engineering"},
        {"LowOrg": "SU095", "LOB": "Solid Waste", "Division": "Solid Waste Ops", "LowOrg_Name": "Collection"},
        {"LowOrg": "SU096", "LOB": "Solid Waste", "Division": "Solid Waste Ops", "LowOrg_Name": "Transfer Stations"},
        {"LowOrg": "SU097", "LOB": "DWW LOB", "Division": "Wastewater", "LowOrg_Name": "Wastewater Treatment"},
        {"LowOrg": "SU098", "LOB": "DWW LOB", "Division": "Wastewater", "LowOrg_Name": "Wastewater Collection"},
        {"LowOrg": "SU099", "LOB": "Shared Services", "Division": "Fleet", "LowOrg_Name": "Fleet Management"},
        {"LowOrg": "SU100", "LOB": "CMD + Emerg Mgmt", "Division": "Corporate", "LowOrg_Name": "Emergency Management"},
        {"LowOrg": "SU110", "LOB": "Solid Waste", "Division": "Landfill", "LowOrg_Name": "Landfill Operations"},
        {"LowOrg": "SU125", "LOB": "Water", "Division": "Field Operations", "LowOrg_Name": "Pump Stations"},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(REFERENCE_DIR, 'LOB_LowOrgList.xlsx'), index=False)
    print(f"  Created LOB_LowOrgList.xlsx ({len(df)} rows)")


# ============================================================
# CONSUMPTION DATA SOURCES
# ============================================================

def create_pse_data():
    """Create PSE_Data_2022.xlsx - PSE electricity consumption."""
    acct_ids = ["200003770308", "200003770309", "200004120115", "200004120116", "200005880220"]
    rates = ["SCH_31EC", "SCH_24R", "SCH_31EC", "SCH_7R", "SCH_24R"]
    months_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="MS")

    rows = []
    for i in range(20):
        acct_idx = i % len(acct_ids)
        month = months_2022[i % 12]
        end = month + pd.offsets.MonthEnd(0)
        days = end.day
        kwh = np.random.uniform(50000, 800000)
        rows.append({
            "ACCT_ID": acct_ids[acct_idx],
            "BP": 1000456035 + acct_idx,
            "Customer": "SEATTLE PUBLIC UTILITIES",
            "Meter": f"MTR-{1000 + i}",
            "Rate": rates[acct_idx],
            "StartDate": month.strftime("%m/%d/%y"),
            "EndDate": end.strftime("%m/%d/%y"),
            "THM": round(np.random.uniform(0, 500), 2) if acct_idx % 3 == 0 else 0,
            "KWH": round(kwh, 2),
            "KW": round(kwh / (days * 24) * 1.1, 2),
            "KVH": 0,
            "Billed": round(kwh * 0.085, 2),
            "OrigConsumption": round(kwh, 2),
            "Unit": "KWH",
            "DaysInBill": days,
            "AvgPerDay": round(kwh / days, 2),
            "InvYear": 2022,
            "EF_ID": f"Elec_PSE_2022_CO2",
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'PSE_Data_2022.xlsx'), index=False)
    print(f"  Created PSE_Data_2022.xlsx ({len(df)} rows)")


def create_scl_data():
    """Create SCL_2022.xlsx - SCL electricity consumption."""
    acct_ids = ["0010710000", "0010710001", "0010720050", "0010720051", "0010730100"]
    sa_types = ["ECOM-SM", "ECOM-LG", "ECOM-SM", "ECOM-LG", "ECOM-SM"]

    rows = []
    for i in range(20):
        acct_idx = i % len(acct_ids)
        month_offset = i % 12
        bill_start = datetime(2022, 1, 1) + timedelta(days=month_offset * 29)
        bill_end = bill_start + timedelta(days=np.random.randint(28, 32))
        bill_dt = bill_end + timedelta(days=np.random.randint(3, 8))
        days = (bill_end - bill_start).days
        kwh = np.random.uniform(200, 15000)
        timespan = "2021/2022" if month_offset < 6 else "2022/2023"

        rows.append({
            "ACCT_ID": acct_ids[acct_idx],
            "SA_ID": f"00101865{i:02d}",
            "SA_TYPE_CD": sa_types[acct_idx],
            "SA_END_DT": "",
            "PREM_ID": f"0010186{acct_idx:03d}",
            "SP_ID": f"SP-{5000 + i}",
            "BILL_ID": f"BILL-{90000 + i}",
            "BILL_DT": bill_dt.strftime("%Y-%m-%d"),
            "BILL_START": bill_start.strftime("%Y-%m-%d"),
            "BILL_END": bill_end.strftime("%Y-%m-%d"),
            "DAYS": days,
            "EST_SW": "N" if i % 5 != 0 else "Y",
            "TOTAL_KWH": round(kwh, 2),
            "Avg/Day": round(kwh / days, 4),
            "Timespan": timespan,
            "2021/2022": round(kwh * 0.5, 2) if timespan == "2021/2022" else 0,
            "2022/2023": round(kwh * 0.5, 2) if timespan == "2022/2023" else 0,
            "YearPro": round(kwh * 0.48, 2),
            "ConsPro": round(kwh * 0.52, 2),
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'SCL_2022.xlsx'), index=False)
    print(f"  Created SCL_2022.xlsx ({len(df)} rows)")


def create_ac_facilities():
    """Create AC_Facilities.xlsx - Facilities air conditioning refrigerant data."""
    facilities = [
        ("Airport Way Center (AWC- bldg D)", "220015316056", "CMD + Emerg Mgmt", "Leased from FAS"),
        ("Ballard Operation Building (BOB)", "220012349050", "DWW LOB", "SPU"),
        ("Cedar Falls Admin Building", "220013470100", "Water", "SPU"),
        ("South Operations Center (SOC)", "220018920075", "Shared Services", "SPU"),
        ("North Operations Center (NOC)", "220018920076", "Water", "SPU"),
    ]
    equip_types = ["ComACHP", "ComACSp", "ComACHP", "ComACSp", "ComACHP"]
    refrigerants = ["R-410A", "R-410A", "R-134A", "R-410A", "R-134A"]
    statuses = ["Active", "Active", "Active", "Decommissioned", "Active"]
    brands = ["Carrier", "Trane", "Lennox", "Daikin", "Carrier"]

    rows = []
    for i in range(20):
        fac_idx = i % len(facilities)
        fac_name, acct, lob, ownership = facilities[fac_idx]
        tonnage = np.random.choice([2.0, 3.5, 5.0, 7.5, 10.0])
        cap_lbs = round(tonnage * 0.5, 2)
        cap_kg = round(cap_lbs * 0.453592, 5)
        start_year = np.random.choice([2015, 2016, 2017, 2018, 2019])
        end_year = np.random.choice([2023, 2024, 2025, 0])
        year_frac = 1.0 if end_year == 0 else round(np.random.uniform(0.25, 1.0), 4)
        ops_rate = round(cap_kg * 0.0001, 6)

        rows.append({
            "Facilities_withAC": fac_name,
            "Account Number": acct,
            "LOB": lob,
            "Ownership": ownership,
            "AC": "Yes",
            "EquipmentType": equip_types[fac_idx],
            "Description": f"Unit {i + 1} - {equip_types[fac_idx]}",
            "Brand": brands[fac_idx],
            "Tonnage": tonnage,
            "Capacity_lbs": str(cap_lbs),
            "Capacity_kg": cap_kg,
            "RefrigerantType": refrigerants[fac_idx],
            "Type": equip_types[fac_idx],
            "StartDate": f"01/01/{start_year}",
            "EndDate": f"12/31/{end_year}" if end_year > 0 else "",
            "Status": statuses[fac_idx] if i < 15 else "Decommissioned",
            "EF_ID": f"FacAC_{equip_types[fac_idx]}_20XX",
            "StartYear": start_year,
            "EndYear": end_year if end_year > 0 else "",
            "EndMonth": "",
            "YearFraction": year_frac,
            "2019_Install": 0 if start_year < 2019 else round(cap_kg * 0.02, 6),
            "2019_Ops": round(ops_rate, 6),
            "2019_Disposal": 0,
            "2019_RecovEffic": 0,
            "2019_EOL": 0,
            "2019_Total_kg": round(ops_rate + (cap_kg * 0.02 if start_year == 2019 else 0), 6),
            "2019_mtCO2e": round((ops_rate + (cap_kg * 0.02 if start_year == 2019 else 0)) * (1924 if refrigerants[fac_idx] == "R-410A" else 1300), 4),
            "2020_Install": 0 if start_year < 2020 else round(cap_kg * 0.02, 6),
            "2020_Ops": round(ops_rate, 6),
            "2020_Disposal": 0,
            "2020_RecovEffic": 0,
            "2020_EOL": 0,
            "2020_Total_kg": round(ops_rate + (cap_kg * 0.02 if start_year == 2020 else 0), 6),
            "2020_mtCO2e": round((ops_rate + (cap_kg * 0.02 if start_year == 2020 else 0)) * (1924 if refrigerants[fac_idx] == "R-410A" else 1300), 4),
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'AC_Facilities.xlsx'), index=False)
    print(f"  Created AC_Facilities.xlsx ({len(df)} rows)")


def create_compressed_gases():
    """Create CompressedGases_2022.xlsx - Compressed industrial gases."""
    materials = ["Acetylene", "CO2", "Acetylene", "CO2"]
    units_map = {"Acetylene": "SCF", "CO2": "lbs"}
    low_orgs = ["SU086", "SU087", "SU090", "SU091", "SU092", "SU093", "N/A"]

    rows = []
    for i in range(20):
        mat = materials[i % len(materials)]
        year = np.random.choice([2019, 2020, 2021, 2022])
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        volume = np.random.choice([50, 100, 132, 200, 220, 264, 300, 400, 528])

        rows.append({
            "Invoice": f"SU-{np.random.randint(1000000, 9999999):07d}",
            "Date": f"{month}/{day}/{year}",
            "Material": mat,
            "Volume": volume,
            "Unit": units_map[mat],
            "LowOrg": np.random.choice(low_orgs),
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'CompressedGases_2022.xlsx'), index=False)
    print(f"  Created CompressedGases_2022.xlsx ({len(df)} rows)")


def create_fleet_fuel():
    """Create FleetFuel_2019-2022.xlsx - Fleet fuel consumption."""
    equip_nos = [1011, 4386, 5504, 16407, 16411, 22050, 34386, 45200]
    manufacturers = ["FLTC", "FORD", "CHEV", "INTL", "FORD", "CHEV", "KENW", "FORD"]
    models = ["SPU", "F-150", "SILVERADO", "4300", "TRANSIT", "EXPRESS", "T370", "F-250"]
    equip_types = ["FUEL CODE", "PICKUP", "PICKUP", "TRUCK", "VAN", "VAN", "TRUCK", "PICKUP"]
    vehicle_classes = ["Unknown", "Light", "Light", "Heavy", "Medium", "Medium", "Heavy", "Light"]
    dept_codes = ["SU099", "SU088", "SU091", "SU092", "SU093", "SU095", "SU092", "SU090"]
    locations = ["SOC", "NOC", "OCC-SPU", "HALLER LAKE", "SOC", "NOC", "OCC-SPU", "CEDAR FALLS"]
    fuel_types = ["UNL", "Gasoline", "Diesel", "B20", "Gasoline", "UNL", "Diesel", "Gasoline"]

    rows = []
    for i in range(20):
        eq_idx = i % len(equip_nos)
        year = np.random.choice([2019, 2020, 2021, 2022])
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        qty = round(np.random.uniform(3.0, 85.0), 2)
        # Occasionally add a negative (correction) value for testing
        if i == 15:
            qty = round(-np.random.uniform(5.0, 20.0), 2)
        # Occasionally set qty to 0 for filter testing
        if i == 18:
            qty = 0.0

        model_year = np.random.randint(2005, 2020)

        rows.append({
            "EQ_EQUIP_NO": equip_nos[eq_idx],
            "YEAR": model_year,
            "MANUFACTURER": manufacturers[eq_idx],
            "MODEL": models[eq_idx],
            "EQTYP_EQUIP_TYPE": equip_types[eq_idx],
            "DESCRIPTION": f"{equip_types[eq_idx]} - {dept_codes[eq_idx]}",
            "Vehicle Class": vehicle_classes[eq_idx],
            "IN_SERVICE_DATE": f"01/15/{model_year}",
            "RETIRE_DATE": "",
            "EST_REPLACE_YR": model_year + 12,
            "DEPT_DEPT_CODE": dept_codes[eq_idx],
            "LOC_STATION_LOC": locations[eq_idx],
            "METER_1_LIFE_TOTAL": round(np.random.uniform(10000, 200000), 1),
            "METER_2_LIFE_TOTAL": 0,
            "FUEL_TYPE": fuel_types[eq_idx],
            "QTY_FUEL": qty,
            "mtCO2": round(qty * 0.008887, 4) if qty > 0 else 0,
            "mtCH4": round(qty * 2.27797e-07, 8) if qty > 0 else 0,
            "mtN2O": round(qty * 1.11899e-07, 8) if qty > 0 else 0,
            "mtCO2e": round(qty * 0.009, 4) if qty > 0 else 0,
            "FTK_DATE": f"{month:02d}/{day:02d}/{year}",
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'FleetFuel_2019-2022.xlsx'), sheet_name='Data', index=False)
    print(f"  Created FleetFuel_2019-2022.xlsx ({len(df)} rows)")


def create_ac_fleet():
    """Create AC_Fleet_2022.xlsx - Fleet MVAC (Mobile Vehicle Air Conditioning)."""
    equip_ids = [4396, 5504, 16407, 16411, 22050, 34386, 45200, 50100]
    manufacturers = ["MGSX", "SRCO", "FORD", "FORD", "CHEV", "KENW", "FORD", "INTL"]
    models = ["TRAILER", "UNKN", "F-150", "TRANSIT", "EXPRESS", "T370", "F-250", "4300"]
    equip_types = ["TRAILER", "TRAILER", "PICKUP", "VAN", "VAN", "TRUCK", "PICKUP", "TRUCK"]
    dept_ids = ["SU125", "SU125", "SU088", "SU091", "SU093", "SU092", "SU090", "SU095"]
    locations = ["HALLER LAKE", "HALLER LAKE", "NOC", "OCC", "SOC", "OCC-SPU", "CEDAR FALLS", "NOC"]
    statuses = ["OWN", "UTIL", "OWN", "OWN", "OWN", "OWN", "OWN", "UTIL"]
    has_ac_vals = ["NO", "NO", "YES", "YES", "YES", "YES", "YES", "YES"]
    rs_nrs = ["NRS", "NRS", "RS", "RS", "RS", "NRS", "RS", "RS"]

    rows = []
    for i in range(20):
        eq_idx = i % len(equip_ids)
        model_year = np.random.randint(1986, 2022)
        in_service = datetime(model_year, np.random.randint(1, 13), np.random.randint(1, 28))
        retire = "" if i % 4 != 0 else (in_service + timedelta(days=365 * np.random.randint(8, 15))).strftime("%m/%d/%Y")

        rows.append({
            "Equipment ID": equip_ids[eq_idx],
            "Model Year": model_year,
            "Manufacturer ID": manufacturers[eq_idx],
            "Model ID": models[eq_idx],
            "Equipment Type": equip_types[eq_idx],
            "Equipment Description": f"{equip_types[eq_idx]} - {models[eq_idx]}",
            "Department ID": dept_ids[eq_idx],
            "Vehicle Location": locations[eq_idx],
            "Operator Name": f"Operator {i + 1}",
            "Status Codes": statuses[eq_idx],
            "Life Cycle Status Code ID": "ACTIVE" if not retire else "RETIRED",
            "Actual In Service Date": in_service.strftime("%m/%d/%Y"),
            "Retirement Date": retire,
            "Estimated Replacement Year": model_year + 12,
            "Billing Type ID": "INTERNAL",
            "Checked": "Y" if i % 3 == 0 else "N",
            "2022 Check": "Y" if i % 2 == 0 else "",
            "Has AC": has_ac_vals[eq_idx],
            "RS/NRS": rs_nrs[eq_idx],
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'AC_Fleet_2022.xlsx'), index=False)
    print(f"  Created AC_Fleet_2022.xlsx ({len(df)} rows)")


def create_landfill_ghg():
    """Create LandfillGHG.xlsx - Landfill methane emissions."""
    landfills = [
        ("Kent Highlands Landfill", "SU110"),
        ("Midway Landfill", "SU110"),
        ("South Park Landfill", "SU110"),
        ("Georgetown Landfill", "SU110"),
    ]

    rows = []
    for i in range(20):
        lf_idx = i % len(landfills)
        name, loworgs = landfills[lf_idx]
        # Vary the conversion factor
        lf_conv = round(np.random.uniform(0.85, 1.15), 4)
        # Vary emissions across years with some realistic patterns
        base = np.random.uniform(200, 800)
        y2019 = round(base * np.random.uniform(0.9, 1.1), 4)
        y2020 = round(base * np.random.uniform(0.85, 1.05), 4)
        y2021 = round(base * np.random.uniform(0.80, 1.0), 4)
        y2022 = round(base * np.random.uniform(0.75, 0.95), 4)

        rows.append({
            "ACCT_ID": name,
            "LowOrg": loworgs,
            "Lfconversion": lf_conv,
            "2019_orig_CO2e": y2019,
            "GWP2019": 21,
            "2019_AR5": round(y2019 * 28 / 21, 4),
            "2020_orig_CO2e": y2020,
            "GWP2020": 21,
            "2020_AR5": round(y2020 * 28 / 21, 4),
            "2021_orig_CO2e": y2021,
            "GWP20212": 25,
            "2021_AR5": round(y2021 * 28 / 25, 4),
            "2022_orig_CO2e": y2022,
            "GWP2022": 28,
            "2022_AR5": round(y2022 * 28 / 28, 4),
        })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(CONSUMPTION_DIR, 'LandfillGHG.xlsx'), index=False)
    print(f"  Created LandfillGHG.xlsx ({len(df)} rows)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Creating reference tables...")
    create_efid()
    create_gwps()
    create_lob_loworgs()

    print("\nCreating consumption data sources...")
    create_pse_data()
    create_scl_data()
    create_ac_facilities()
    create_compressed_gases()
    create_fleet_fuel()
    create_ac_fleet()
    create_landfill_ghg()

    print("\nDone! All sample data files created.")
