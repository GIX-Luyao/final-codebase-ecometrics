# EcoMetrics GHG Emissions - Data Source Schema Documentation

This document provides comprehensive schema definitions and sample data for all data sources used in the GHG emissions calculation system.

---

## Table of Contents

1. [Facilities: PSE](#1-facilities-pse)
2. [Facilities: SCL](#2-facilities-scl)
3. [Facilities: AC Facilities](#3-facilities-ac-facilities)
4. [Facilities: Compressed Gases](#4-facilities-compressed-gases)
5. [Fleet: Fleet Fuel](#5-fleet-fleet-fuel)
6. [Fleet: Fleet MVAC](#6-fleet-fleet-mvac)
7. [Landfill: Landfill Methane](#7-landfill-landfill-methane)

---

## 1. Facilities: PSE

**Source File:** `PSE_Data_20XX.xlsx`

### Raw Data Schema (PSE_Data_20XX.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account identifier |
| BP | integer | Business partner number |
| Customer | string | Customer name |
| Meter | string | Meter identifier |
| Rate | string | Rate schedule code |
| StartDate | date | Billing period start date |
| EndDate | date | Billing period end date |
| THM | float | Therms consumed |
| KWH | float | Kilowatt-hours consumed |
| KW | float | Kilowatt demand |
| KVH | float | Kilovolt-hours |
| Billed | float | Billed amount ($) |
| OrigConsumption | float | Original consumption value |
| Unit | string | Unit of measurement |
| DaysInBill | integer | Number of days in billing period |
| AvgPerDay | float | Average consumption per day |
| 2020/2021 | string | Fiscal year indicator |
| 2021/2022 | string | Fiscal year indicator |
| YearProrate | float | Year-prorated consumption |
| ConsProrated | float | Prorated consumption |
| Timespan | string | Time period |
| InvYear | integer | Inventory year |
| EF_ID | string | Emission factor identifier |

### Sample Raw Data

| ACCT_ID | BP | Customer | Rate | StartDate | EndDate | KWH | Unit | DaysInBill | AvgPerDay |
|---------|-----|----------|------|-----------|---------|-----|------|------------|-----------|
| 200003770308 | 1000456035 | SEATTLE PUBLIC UTILITIES | SCH_31EC | 01/01/21 | 01/31/21 | 634055.52 | KWH | 31 | 20,453.40 |
| 200003770308 | 1000456035 | SEATTLE PUBLIC UTILITIES | SCH_31EC | 02/01/21 | 02/28/21 | 562769.85 | KWH | 28 | 20,098.92 |

### Processed Data Schema (PSE20XX_calc)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account identifier |
| Utility | string | Utility provider (PSE) |
| Year | string | Inventory year |
| Consumption | float | Energy consumption |
| Unit | string | Unit of measurement (KWH) |
| GHG | string | Greenhouse gas type (CO2, CH4, N2O) |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor (metric tons per unit) |
| GWP | integer | Global warming potential |
| mtCO2e_calc | float | Calculated emissions (metric tons CO2 equivalent) |

### Sample Processed Data

| ACCT_ID | Utility | Year | Consumption | Unit | GHG | EF_ID | GHG_MTperUnit | GWP | mtCO2e_calc |
|---------|---------|------|-------------|------|-----|-------|---------------|-----|-------------|
| 200003770308 | PSE | 2022 | 7918489.562 | KWH | CH4 | Elec_PSE_2022_CH4 | 2.63084E-08 | 28 | 5.833035647 |
| 200003770308 | PSE | 2022 | 7918489.562 | KWH | CO2 | Elec_PSE_2022_CO2 | 0.000415945 | 1 | 3293.653749 |
| 200003770308 | PSE | 2022 | 7918489.562 | KWH | N2O | Elec_PSE_2022_N2O | 3.62874E-09 | 265 | 7.614553924 |

---

## 2. Facilities: SCL

**Source File:** `SCL_2022.xlsx`

### Raw Data Schema (SCL_2022.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account identifier |
| SA_ID | string | Service agreement ID |
| SA_TYPE_CD | string | Service agreement type code |
| SA_END_DT | date | Service agreement end date |
| PREM_ID | string | Premise identifier |
| SP_ID | string | Service point ID |
| BILL_ID | string | Bill identifier |
| BILL_DT | datetime | Bill date |
| BILL_START | datetime | Billing period start |
| BILL_END | datetime | Billing period end |
| DAYS | integer | Number of days |
| EST_SW | string | Estimate switch indicator |
| TOTAL_KWH | float | Total kilowatt-hours |
| Avg/Day | float | Average consumption per day |
| Timespan | string | Time period indicator |
| 2021/2022 | float | Fiscal year consumption |
| 2022/2023 | float | Fiscal year consumption |
| YearPro | float | Year-prorated value |
| ConsPro | float | Prorated consumption |

### Sample Raw Data

| ACCT_ID | SA_ID | SA_TYPE_CD | PREM_ID | BILL_DT | BILL_START | BILL_END | DAYS | TOTAL_KWH | Avg/Day | Timespan |
|---------|-------|------------|---------|---------|------------|----------|------|-----------|---------|----------|
| 0010710000 | 0010186500 | ECOM-SM | 0010186183 | 2022-02-04 | 2021-12-29 | 2022-01-27 | 29 | 4373 | 150.7931 | 2021/2022 |
| 0010710000 | 0010186501 | ECOM-SM | 0010186183 | 2022-02-04 | 2021-12-29 | 2022-01-27 | 29 | 330 | 11.37931 | 2021/2022 |

### Processed Data Schema (SCL_calc_20XX)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account identifier |
| Utility | string | Utility provider (SCL) |
| Year | string | Inventory year |
| Unit | string | Unit of measurement (KWH) |
| GHG | string | Greenhouse gas type (CO2, CH4, N2O) |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor (metric tons per unit) |
| GWP | integer | Global warming potential |
| Consumption | float | Energy consumption |
| mtCO2e_calc | float | Calculated emissions (metric tons CO2 equivalent) |

### Sample Processed Data

| ACCT_ID | Utility | Year | Unit | GHG | EF_ID | GHG_MTperUnit | GWP | Consumption | mtCO2e_calc |
|---------|---------|------|------|-----|-------|---------------|-----|-------------|-------------|
| 0010710000 | SCL | 2022 | KWH | CH4 | Elec_SCL_2022_CH4 | 1.36078E-08 | 28 | 694130.7931 | 0.264476357 |
| 0010710000 | SCL | 2022 | KWH | CO2 | Elec_SCL_2022_CO2 | 1.05097E-05 | 1 | 694130.7931 | 7.295139514 |
| 0010710000 | SCL | 2022 | KWH | N2O | Elec_SCL_2022_N2O | 1.81437E-09 | 265 | 694130.7931 | 0.333743974 |

---

## 3. Facilities: AC Facilities

**Source File:** `AC_Facilities.xlsx`

### Raw Data Schema (AC_Facilities.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| Facilities_withAC | string | Facility name |
| Account Number | string | Account identifier |
| LOB | string | Line of business |
| Ownership | string | Ownership type |
| AC | string | Air conditioning indicator (Yes/No) |
| EquipmentType | string | Type of equipment |
| Description | string | Equipment description |
| Brand | string | Equipment brand |
| Tonnage | float | Cooling capacity (tons) |
| Capacity_lbs | string | Refrigerant capacity (pounds) |
| Capacity_kg | float | Refrigerant capacity (kilograms) |
| RefrigerantType | string | Type of refrigerant |
| Type | string | AC system type |
| StartDate | date | Installation date |
| EndDate | date | Decommission date |
| Status | string | Equipment status |
| EF_ID | string | Emission factor identifier |
| StartYear | integer | Start year |
| EndYear | integer | End year |
| EndMonth | string | End month |
| YearFraction | float | Fraction of year active |
| 2019_Install | float | 2019 installation emissions |
| 2019_Ops | float | 2019 operational emissions |
| 2019_Disposal | float | 2019 disposal emissions |
| 2019_RecovEffic | float | 2019 recovery efficiency |
| 2019_EOL | float | 2019 end-of-life emissions |
| 2019_Total_kg | float | 2019 total kg refrigerant |
| 2019_mtCO2e | float | 2019 total mtCO2e |
| 2020_Install | float | 2020 installation emissions |
| 2020_Ops | float | 2020 operational emissions |
| 2020_Disposal | float | 2020 disposal emissions |
| 2020_RecovEffic | float | 2020 recovery efficiency |
| 2020_EOL | float | 2020 end-of-life emissions |
| 2020_Total_kg | float | 2020 total kg refrigerant |
| 2020_mtCO2e | float | 2020 total mtCO2e |

### Sample Raw Data

| Facilities_withAC | Account Number | LOB | Ownership | AC | RefrigerantType | Type | EF_ID | 2019_Ops | 2019_mtCO2e | 2020_Ops | 2020_mtCO2e |
|-------------------|----------------|-----|-----------|-----|-----------------|------|-------|----------|-------------|----------|-------------|
| Airport Way Center (AWC- bldg D) | 220015316056 | CMD + Emerg Mgmt | Leased from FAS | Yes | R-410A | ComACHP | FacAC_ComACHP_20XX | 1.00000 | 1.924 | 1.00000 | 1.924 |
| Ballard Operation Building (BOB) | 220012349050 | DWW LOB | SPU | Yes | R-410A | ComACHP | FacAC_ComACHP_20XX | 1.00000 | 1.924 | 1.00000 | 1.924 |

### Processed Data Schema (AC_Facilities)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account identifier |
| Capacity_kg | float | Refrigerant capacity (kg) |
| GHG | string | Greenhouse gas (refrigerant type) |
| Subtype | string | Equipment subtype |
| Year | string | Inventory year |
| Unit | string | Unit of measurement |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor |
| GWP | integer | Global warming potential |
| mtCO2e_calc | float | Calculated emissions |
| Consumption | float | Refrigerant consumption |

### Sample Processed Data

| ACCT_ID | Capacity_kg | GHG | Subtype | Year | Unit | EF_ID | GHG_MTperUnit | GWP | mtCO2e_calc | Consumption |
|---------|-------------|-----|---------|------|------|-------|---------------|-----|-------------|-------------|
| 1495000000 | 0.94006942 | R-410A | Facility AC | 2019 | Refrig kg | FacAC_ComACHP_ | 0.0001 | 1924 | 0.180869356 | 0.094006942 |
| 1495000000 | 0.94006942 | R-410A | Facility AC | 2020 | Refrig kg | FacAC_ComACHP_ | 0.0001 | 1924 | 0.180869356 | 0.094006942 |
| 1495000000 | 0.94006942 | R-410A | Facility AC | 2021 | Refrig kg | FacAC_ComACHP_ | 0.0001 | 1924 | 0.180869356 | 0.094006942 |

---

## 4. Facilities: Compressed Gases

**Source File:** `CompressedGases_20XX.xlsx`

### Raw Data Schema (CompressedGases_20XX.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| Invoice | string | Invoice number |
| Date | date | Purchase date |
| Material | string | Gas type (Acetylene, CO2) |
| Volume | integer | Volume purchased |
| Unit | string | Unit of measurement (SCF, lbs) |
| LowOrg | string | Low-level organization code |

### Sample Raw Data

| Invoice | Date | Material | Volume | Unit | LowOrg |
|---------|------|----------|--------|------|--------|
| SU-0000013315 | 1/20/2019 | Acetylene | 200 | SCF | N/A |
| SU0-0000015291 | 1/30/2019 | CO2 | 50 | lbs | SU086 |
| SU0-0000015291 | 1/31/2019 | CO2 | 220 | lbs | SU087 |
| SU0-0000023639 | 8/27/2019 | Acetylene | 132 | SCF | SU091 |
| SU0-0000015626 | 2/13/2019 | Acetylene | 528 | SCF | SU091 |

### Processed Data Schema (CompressedGases)

| Column | Data Type | Description |
|--------|-----------|-------------|
| Consumption | float | Gas consumption |
| Unit | string | Unit of measurement |
| Year | integer | Inventory year |
| Subtype | string | Gas type |
| SourceNotes | string | Data source notes |
| ACCT_ID | string | Account identifier |
| GHG | string | Greenhouse gas type |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor |
| GWP | integer | Global warming potential |
| mtCO2e_calc | float | Calculated emissions |
| LowOrg | string | Low-level organization code |
| LOB | string | Line of business |
| Modified LOB | string | Modified line of business |

### Sample Processed Data

| Consumption | Unit | Year | Subtype | SourceNotes | ACCT_ID | GHG | EF_ID | GHG_MTperUnit | GWP | mtCO2e_calc | LowOrg | LOB | Modified LOB |
|-------------|------|------|---------|-------------|---------|-----|-------|---------------|-----|-------------|--------|-----|--------------|
| 20 | SCF | 2019 | Acetylene | Industrial Gases | Compressed Gases | CO2 | ComprGas_Acetyle | 0.00011 | 1 | 0.0022 | SU090 | Water | Water |
| 32 | SCF | 2020 | Acetylene | Industrial Gases | Compressed Gases | CO2 | ComprGas_Acetyle | 0.00011 | 1 | 0.00352 | SU090 | Water | Water |
| 50 | lbs | 2019 | CO2 | Industrial Gases | Compressed Gases | CO2 | ComprGas_CO2_2 | 0.000453593 | 1 | 0.022679645 | SU086 | Shared Services | Shared Services |
| 50 | lbs | 2019 | CO2 | Industrial Gases | Compressed Gases | CO2 | ComprGas_CO2_2 | 0.000453593 | 1 | 0.022679645 | SU086 | Shared Services | Shared Services |

---

## 5. Fleet: Fleet Fuel

**Source File:** `FleetFuel_20XX.xlsx`

### Raw Data Schema (FleetFuel_20XX.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| EQ_EQUIP_NO | integer | Equipment/vehicle number |
| YEAR | integer | Model year |
| MANUFACTURER | string | Vehicle manufacturer |
| MODEL | string | Vehicle model |
| EQTYP_EQUIP_TYPE | string | Equipment type code |
| DESCRIPTION | string | Equipment description |
| Vehicle Class | string | Vehicle classification |
| IN_SERVICE_DATE | date | In-service date |
| RETIRE_DATE | date | Retirement date |
| EST_REPLACE_YR | integer | Estimated replacement year |
| DEPT_DEPT_CODE | string | Department code |
| LOC_STATION_LOC | string | Station location |
| METER_1_LIFE_TOTAL | float | Odometer reading |
| METER_2_LIFE_TOTAL | float | Secondary meter reading |
| FUEL_TYPE | string | Fuel type |
| QTY_FUEL | float | Fuel quantity |
| mtCO2 | float | CO2 emissions (metric tons) |
| mtCH4 | float | CH4 emissions (metric tons) |
| mtN2O | float | N2O emissions (metric tons) |
| mtCO2e | float | Total CO2e emissions |
| FTK_DATE | date | Fuel transaction date |

### Sample Raw Data

| EQ_EQUIP_NO | YEAR | MANUFACTURER | MODEL | EQTYP_EQUIP_TYPE | DESCRIPTION | Vehicle Class | IN_SERVICE_DATE | DEPT_DEPT_CODE | LOC_STATION_LOC | FUEL_TYPE | QTY_FUEL | mtCO2e | FTK_DATE |
|-------------|------|--------------|-------|------------------|-------------|---------------|-----------------|----------------|-----------------|-----------|----------|--------|----------|
| 1011 | 2012 | FLTC | SPU | FUEL CODE | OFFROAD FUEL - SU099 | Unknown | 08/29/2012 | SU099 | 2012 FLTC SPU OFFRO | UNL | 5.00 | 0.05 | 06/26/2019 |
| 1011 | 2012 | FLTC | SPU | FUEL CODE | OFFROAD FUEL - SU099 | Unknown | 08/29/2012 | SU099 | 2012 FLTC SPU OFFRO | UNL | 9.19 | 0.08 | 10/15/2019 |

### Processed Data Schema (FleetFuel_lean)

| Column | Data Type | Description |
|--------|-----------|-------------|
| Consumption | float | Fuel consumption |
| Unit | string | Unit of measurement (gal) |
| Subtype | string | Fuel type |
| Year | integer | Inventory year |
| LowOrg | string | Low-level organization code |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor |
| GWP | integer | Global warming potential |
| mtCO2e_calc | float | Calculated emissions |
| ACCT_ID | string | Account/equipment identifier |
| LOB | string | Line of business |
| Modified LOB | string | Modified line of business |
| SourceNotes | string | Data source notes |
| Vehicle Location | string | Vehicle location |
| Month | integer | Month of transaction |
| FillDate | date | Fuel fill date |
| GHG | string | Greenhouse gas type |
| Sector | string | Emission sector |
| EF_ID_group | string | Emission factor group |
| Vehicle Class | string | Vehicle classification |

### Sample Processed Data

| Consumption | Unit | Subtype | Year | LowOrg | EF_ID | GHG_MTperUnit | GWP | mtCO2e_calc | ACCT_ID | LOB | SourceNotes | Vehicle Location | Month | FillDate | GHG | Sector | Vehicle Class |
|-------------|------|---------|------|--------|-------|---------------|-----|-------------|---------|-----|-------------|------------------|-------|----------|-----|--------|---------------|
| -16.8 | gal | B20 | 2021 | SU092 | Fuel_B20_20XX_CH4_ | 2.05183E-07 | 28 | -9.65182E-05 | 34386 | Water | Fleet Fuel | OCC-SPU | 10 | Saturday, October 30, 2021 | CH4 | Fleet | Heavy |
| -16.8 | gal | B20 | 2021 | SU092 | Fuel_B20_20XX_CO2_ | 0.010058 | 1 | -0.1689744 | 34386 | Water | Fleet Fuel | OCC-SPU | 10 | Saturday, October 30, 2021 | CO2 | Fleet | Heavy |
| -16.8 | gal | B20 | 2021 | SU092 | Fuel_B20_20XX_N2O_ | 2.2027E-07 | 265 | -0.000980643 | 34386 | Water | Fleet Fuel | OCC-SPU | 10 | Saturday, October 30, 2021 | N2O | Fleet | Heavy |

---

## 6. Fleet: Fleet MVAC

**Source File:** `AC_Fleet_20xx.xlsx`

### Raw Data Schema (AC_Fleet_20xx.xlsx)

| Column | Data Type | Description |
|--------|-----------|-------------|
| Equipment ID | integer | Equipment identifier |
| Model Year | integer | Vehicle model year |
| Manufacturer ID | string | Manufacturer identifier |
| Model ID | string | Model identifier |
| Equipment Type | string | Type of equipment |
| Equipment Description | string | Equipment description |
| Department ID | string | Department identifier |
| Vehicle Location | string | Vehicle location |
| Operator Name | string | Operator name |
| Status Codes | string | Status code |
| Life Cycle Status Code ID | string | Life cycle status |
| Actual In Service Date | date | In-service date |
| Retirement Date | date | Retirement date |
| Estimated Replacement Year | integer | Est. replacement year |
| Billing Type ID | string | Billing type |
| Checked | string | Verification flag |
| 2022 Check | string | 2022 verification |
| Has AC | string | Has air conditioning (YES/NO) |
| RS/NRS | string | Road/Non-road status |

### Sample Raw Data

| Equipment ID | Model Year | Manufacturer ID | Model ID | Equipment Type | Equipment Description | Department ID | Vehicle Location | Status Codes | Actual In Service Date | Has AC | RS/NRS |
|--------------|------------|-----------------|----------|----------------|----------------------|---------------|------------------|--------------|------------------------|--------|--------|
| 4396 | 2007 | MGSX | TRAILER | TRAILER | TRAILER - WATER PUMP | SU125 | HALLER LAKE | OWN | 8/7/2010 | NO | NRS |
| 5504 | 1986 | SRCO | UNKN | TRAILER | TRAILER - SEWER DRAG MACHINE | SU125 | HALLER LAKE | UTIL | 8/7/1986 | NO | NRS |

### Processed Data Schema (FleetMVAC_lean)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account/equipment identifier |
| Vehicle Location | string | Vehicle location |
| MVAC_Capacity_kg | float | MVAC refrigerant capacity (kg) |
| GHG | string | Refrigerant type |
| EF_ID | string | Emission factor identifier |
| GHG_MTperUnit | float | Emission factor |
| mtCO2e_calc | float | Calculated emissions |
| LOB | string | Line of business |
| Status Codes | string | Status codes |
| Group | string | Emission group |
| Subtype | string | Equipment subtype |
| Sector | string | Emission sector |
| Activity | string | Activity type |
| Scope | string | Emission scope |
| Label | string | Emission label |
| SourceNotes | string | Data source notes |
| 2021 Check | string | 2021 verification |
| LowOrg | string | Low-level organization |
| Year | string | Inventory year |
| GWP | integer | Global warming potential |
| Modified LOB | string | Modified line of business |

### Sample Processed Data

| ACCT_ID | Vehicle Location | MVAC_Capacity_kg | GHG | EF_ID | GHG_MTperUnit | mtCO2e_calc | LOB | Group | Subtype | Sector | Activity | Scope | Label | SourceNotes | LowOrg | Year | GWP | Modified LOB |
|---------|------------------|------------------|-----|-------|---------------|-------------|-----|-------|---------|--------|----------|-------|-------|-------------|--------|------|-----|--------------|
| - | - | 2 | R-134A | MVAC_OtherMobile_20XX_R- | 0.0002 | 0.52 | - | MVAC_20XX | OtherMobile | Fleet | Fugitive | Scope 1 | Fleet Fugitive Refrigerants | MVAC | - | 2022 | 1300 | - |
| 16407 | NOC | 2 | R-134A | MVAC_OtherMobile_20XX_R- | 0.0002 | 0.52 | Water | MVAC_20XX | OtherMobile | Fleet | Fugitive | Scope 1 | Fleet Fugitive Refrigerants | MVAC | SU088 | 2019 | 1300 | Water |
| 16411 | OCC | 2 | R-134A | MVAC_OtherMobile_20XX_R- | 0.0002 | 0.52 | Water | MVAC_20XX | OtherMobile | Fleet | Fugitive | Scope 1 | Fleet Fugitive Refrigerants | MVAC | SU091 | 2019 | 1300 | Water |

---

## 7. Landfill: Landfill Methane

**Source File:** `LandfillGHG.xlsx`

### Raw Data Schema (LandfillGHG)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Landfill name/identifier |
| LowOrg | string | Low-level organization code |
| Lfconversion | float | Landfill conversion factor |
| 2019_orig_CO2e | float | 2019 original CO2e |
| GWP2019 | integer | 2019 GWP value |
| 2019_AR5 | float | 2019 AR5 adjusted value |
| 2020_orig_CO2e | float | 2020 original CO2e |
| GWP2020 | integer | 2020 GWP value |
| 2020_AR5 | float | 2020 AR5 adjusted value |
| 2021_orig_CO2e | float | 2021 original CO2e |
| GWP20212 | integer | 2021 GWP value |
| 2021_AR5 | float | 2021 AR5 adjusted value |
| 2022_orig_CO2e | float | 2022 original CO2e |
| GWP2022 | integer | 2022 GWP value |
| 2022_AR5 | float | 2022 AR5 adjusted value |

### Processed Data Schema (Landfill_CH4_lean2)

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Landfill name/identifier |
| LowOrg | string | Low-level organization code |
| Subtype | string | Emission subtype |
| SourceNotes | string | Data source notes |
| GHG | string | Greenhouse gas (CH4) |
| AR5_GWP | integer | AR5 global warming potential |
| mtCO2e_calc | float | Calculated emissions |
| LOB | string | Line of business |
| Modified LOB | string | Modified line of business |
| EF_ID | string | Emission factor identifier |
| Consumption | float | Methane consumption (mt CH4) |
| Year | string | Inventory year |
| Unit | string | Unit of measurement |

### Sample Processed Data

| ACCT_ID | LowOrg | 2019_orig_CO2e | GWP2019 | 2019_AR5 | 2020_orig_CO2e | GWP2020 | 2020_AR5 | 2021_orig_CO2e | GWP20212 | 2021_AR5 | Consumption | Year | Unit |
|---------|--------|----------------|---------|----------|----------------|---------|----------|----------------|----------|----------|-------------|------|------|
| Kent Highlands Landfill | SU110 | 508.3998378 | 21 | 677.8664504 | 475.9084112 | 21 | 634.5445482 | 488.9919606 | 25 | 547.6709959 | 17.86214286 | 2022 | mt CH4 |
| Kent Highlands Landfill | SU110 | 508.3998378 | 21 | 677.8664504 | 475.9084112 | 21 | 634.5445482 | 488.9919606 | 25 | 547.6709959 | 19.55967842 | 2021 | mt CH4 |
| Kent Highlands Landfill | SU110 | 508.3998378 | 21 | 677.8664504 | 475.9084112 | 21 | 634.5445482 | 488.9919606 | 25 | 547.6709959 | 22.66230529 | 2020 | mt CH4 |

### Processed Summary Data

| Subtype | SourceNotes | GHG | AR5_GWP | mtCO2e_calc | LOB | Modified LOB | EF_ID |
|---------|-------------|-----|---------|-------------|-----|--------------|-------|
| Landfill Methane | Landfill Methane | CH4 | 28 | 500.14 | Solid Waste | Solid Waste | Landfill_CH4 |
| Landfill Methane | Landfill Methane | CH4 | 28 | 547.6709959 | Solid Waste | Solid Waste | Landfill_CH4 |
| Landfill Methane | Landfill Methane | CH4 | 28 | 634.5445482 | Solid Waste | Solid Waste | Landfill_CH4 |

---

## Emission Factor Reference

### GWP Values (AR5)

| GHG | GWP (AR5) |
|-----|-----------|
| CO2 | 1 |
| CH4 | 28 |
| N2O | 265 |
| R-134A | 1300 |
| R-410A | 1924 |

### Core Calculation Formula

```
mtCO2e_calc = Consumption × GHG_MTperUnit × GWP
```

Where:
- **Consumption**: Activity data (kWh, gallons, kg, etc.)
- **GHG_MTperUnit**: Emission factor (metric tons of GHG per unit of activity)
- **GWP**: Global Warming Potential (converts to CO2 equivalent)

---

## Final Output Schema (MT_CO2e)

The final combined dataset merges all processed data sources into a unified schema:

| Column | Data Type | Description |
|--------|-----------|-------------|
| ACCT_ID | string | Account/source identifier |
| Year | string | Inventory year |
| Consumption | float | Activity consumption |
| Unit | string | Unit of measurement |
| EF_ID | string | Emission factor identifier |
| mtCO2e_calc | float | Calculated CO2e emissions |
| LowOrg | string | Low-level organization |
| Subtype | string | Emission subtype |
| GHG | string | Greenhouse gas type |
| GHG_MTperUnit | float | Emission factor |
| GWP | integer | Global warming potential |
| LOB | string | Line of business |
| Modified LOB | string | Modified line of business |
| SourceNotes | string | Data source identifier |

---

*Document generated from EcoMetrics GHG Emissions Data Processing System*
