# EcoMetrics Backend Integration Testing - Project Specification

## Project Overview

Build a Flask REST API backend for the EcoMetrics GHG Emissions Data Engine integration testing. The backend supports three interfaces for frontend integration testing.

## Tech Stack

- **Framework**: Flask
- **Storage**: Local file system (CSV/JSON files)
- **Data Processing**: Pandas
- **Logging**: Python logging module
- **Port**: localhost:5000
- **CORS**: Enabled (frontend on different port)

<!-- ## Database Schema

### Table: raw_files
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| file_type | TEXT | "fleetfuel" or "efid" |
| file_name | TEXT | Original filename |
| data_json | TEXT | JSON blob of file data |
| is_invalid | BOOLEAN | True for invalid test files |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update time |

### Table: modified_data
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| session_id | TEXT | Auto-generated session ID |
| file_type | TEXT | "fleetfuel" or "efid" |
| original_file_id | INTEGER FK | Reference to raw_files.id |
| data_json | TEXT | JSON blob with modifications |
| change_log | TEXT | JSON array of all changes |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update time |

### Table: audit_log
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| session_id | TEXT | Session identifier |
| action | TEXT | "load", "edit", "save", "calculate" |
| status | TEXT | "success", "error", "warning" |
| message | TEXT | Human-readable message |
| details_json | TEXT | Additional details as JSON |
| timestamp | TIMESTAMP | When action occurred |

--- -->

## File Storage Structure
```
backend/
├── data/
│   ├── raw/                        # Original uploaded files (read-only)
│   │   ├── FleetFuel_Raw.csv
│   │   ├── FleetFuel_Invalid.csv   # For error testing
│   │   └── EFID_Reference.csv
│   ├── sessions/                   # Session-specific modified data
│   │   ├── sess_abc123/
│   │   │   ├── fleetfuel_modified.csv
│   │   │   └── changes.json        # Log of all changes made
│   │   └── sess_def456/
│   │       └── ...
│   └── calculated/                 # Calculated output files
│       └── sess_abc123/
│           └── fleetfuel_lean.csv
├── logs/
│   └── backend.log                 # Audit log (append-only)
```

### Storage Behavior

**Raw Files** (`data/raw/`):
- Original CSV files, never modified
- Loaded on API request

**Session Files** (`data/sessions/{sessionId}/`):
- Created when user makes first edit
- `{dataType}_modified.csv`: Current state of data with all edits applied
- `changes.json`: Array of all changes for audit trail

**Calculated Files** (`data/calculated/{sessionId}/`):
- Output of calculation engine
- `fleetfuel_lean.csv`: Calculated emissions data

**Audit Log** (`logs/backend.log`):
- Append-only text file
- Records all API actions with timestamps

## API Endpoints

### Interface 1: GET /api/raw-data

**Purpose**: Return raw data files for frontend preview

**Query Parameters**:
- `type` (required): "fleetfuel" or "efid"
- `invalid` (optional): "true" to get invalid test file for error testing

**Response (Success)**:
```json
{
  "success": true,
  "dataType": "fleetfuel",
  "fileName": "FleetFuel_Raw.csv",
  "rowCount": 2,
  "columns": ["EQ_EQUIP_NO", "YEAR", ...],
  "columnTypes": {"EQ_EQUIP_NO": "string", "YEAR": "integer", ...},
  "preview": [{"EQ_EQUIP_NO": "1011", ...}, ...]
}
```

**Response (Invalid file - for testing)**:
When `invalid=true`, return valid JSON but with schema issues:
- Missing required columns
- Wrong data types (strings where numbers expected)
- Null values in required fields

---

### Interface 2: POST /api/data/update

**Purpose**: Receive and validate table edits from frontend, save to database

**Request Body**:
```json
{
  "sessionId": "auto-generated-or-provided",
  "dataType": "fleetfuel",
  "changes": [
    {
      "changeType": "column_rename",
      "oldName": "FUEL_TYPE",
      "newName": "Fuel_Type_New"
    },
    {
      "changeType": "cell_edit",
      "rowIndex": 0,
      "column": "QTY_FUEL",
      "oldValue": 5.0,
      "newValue": 10.0
    }
  ]
}
```

**Validation Rules**:
- Column names must be strings
- Column names cannot be pure numbers (e.g., "123" is invalid)
- Column names must start with letter or underscore

**Response (Success)**:
```json
{
  "success": true,
  "sessionId": "sess_abc123",
  "changesApplied": 1,
  "validation": {"status": "passed"},
//   "storage": {
//     "status": "committed",
//     "storageId": 5,
//     "timestamp": "2026-01-25T10:31:05Z"
//   }
  "storage": {
  "status": "committed",
  "filePath": "data/sessions/sess_abc123/fleetfuel_modified.csv",
  "timestamp": "2026-01-25T10:31:05Z"
   }
}
```

**Response (Validation Error)**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Column name '123' is invalid: cannot be a pure number"
  }
}
```

**Logging Requirement**:
All requests must be logged to file and console:
```
[2026-01-25 10:31:00] [INFO] [sess_abc123] EDIT - RECEIVED - Column rename: FUEL_TYPE -> Fuel_Type_New
[2026-01-25 10:31:01] [INFO] [sess_abc123] VALIDATE - SUCCESS - Change validated
[2026-01-25 10:31:02] [INFO] [sess_abc123] SAVE - SUCCESS - Data saved to database (id=5)
[2026-01-25 10:32:00] [ERROR] [sess_def456] VALIDATE - FAILED - Column name "123" is invalid
```

---

### Interface 3: GET /api/calculated-data

**Purpose**: Return calculated emissions data in chart-ready format

**Query Parameters**:
- `sessionId` (required): Session identifier

**Response**:
```json
{
  "success": true,
  "sessionId": "sess_abc123",
  "calculationTimestamp": "2026-01-25T10:35:00Z",
  "summary": {
    "totalMtCO2e": -0.17587,
    "recordCount": 3
  },
  "charts": {
    "pieChart": {
      "title": "mtCO2e by Subtype",
      "type": "pie",
      "data": [
        {"label": "B20", "value": -0.17587, "percentage": 100.0}
      ]
    },
    "barChart": {
      "title": "mtCO2e over FillDate",
      "type": "bar",
      "labels": ["2021-10-30"],
      "datasets": [
        {"label": "mtCO2e", "data": [-0.17587]}
      ]
    }
  },
  "rawData": [
    {
      "Consumption": -16.8,
      "Unit": "gal",
      "Subtype": "B20",
      "Year": 2021,
      "LowOrg": "SU092",
      "EF_ID": "Fuel_B20_20XX_CH4",
      "GHG_MTperUnit": 2.05183e-07,
      "GWP": 28,
      "mtCO2e_calc": -9.65182e-05,
      "ACCT_ID": "4386",
      "LOB": "Water",
      "Vehicle Location": "OCC-SPU",
      "Month": 10,
      "FillDate": "2021-10-30",
      "GHG": "CH4",
      "Sector": "Fleet",
      "Vehicle Class": "Heavy",
      "formulaReference": {
        "formula": "mtCO2e = Consumption × GHG_MTperUnit × GWP",
        "calculation": "-16.8 × 2.05183e-07 × 28 = -9.65182e-05",
        "efidSource": "Fuel_B20_20XX_CH4"
      }
    }
  ]
}
```

---

## Data Schemas

### FleetFuel Raw Data Schema
| Column | Type |
|--------|------|
| EQ_EQUIP_NO | string |
| YEAR | integer |
| MANUFACTURER | string |
| MODEL | string |
| EQTYP_EQUIP_TYPE | string |
| DESCRIPTION | string |
| Vehicle Class | string |
| IN_SERVICE_DATE | date |
| RETIRE_DATE | date |
| EST_REPLACE_YR | integer |
| DEPT_DEPT_CODE | string |
| LOC_STATION_LOC | string |
| METER_1_LIFE_TOTAL | float |
| METER_2_LIFE_TOTAL | float |
| FUEL_TYPE | string |
| QTY_FUEL | float |
| mtCO2 | float |
| mtCH4 | float |
| mtN2O | float |
| mtCO2e | float |
| FTK_DATE | date |

### EFID Reference Schema
| Column | Type |
|--------|------|
| EF_ID | string |
| GHG_MTperUnit | float |
| Unit | string |
| Group | string |
| Subtype | string |
| Year | integer |
| GHG | string |
| GWP | integer |
| Sector | string |
| Activity | string |
| Description | string |
| Scope | string |
| Label | string |

### FleetFuel_lean (Calculated Output) Schema
| Column | Type |
|--------|------|
| Consumption | float |
| Unit | string |
| Subtype | string |
| Year | integer |
| LowOrg | string |
| EF_ID | string |
| GHG_MTperUnit | float |
| GWP | integer |
| mtCO2e_calc | float |
| ACCT_ID | string |
| LOB | string |
| Modified LOB | string |
| SourceNotes | string |
| Vehicle Location | string |
| Month | integer |
| FillDate | string |
| GHG | string |
| Sector | string |
| EF_ID_group | string |
| Vehicle Class | string |

---

## Existing Calculation Code

Use this existing Python code for emissions calculations:
```python
import pandas as pd

def create_lookup_dicts(gwp_df, ef_df, gwp_version="AR5_GWP"):
    """Return GWP and EF lookup dictionaries."""
    gwp_dict = gwp_df.set_index("GHG_Name")[gwp_version].to_dict()
    ef_dict = (
        ef_df
        .set_index(["Subtype", "GHG", "Sector"])["GHG_MTperUnit"]
        .to_dict()
    )
    return gwp_dict, ef_dict


def preprocess_fleet_data(df):
    """Clean and standardize the fleet fuel file."""
    columns_to_drop = [
        'YEAR', 'MANUFACTURER', 'MODEL', 'EQTYP_EQUIP_TYPE', 'DESCRIPTION',
        'IN_SERVICE_DATE', 'RETIRE_DATE', 'EST_REPLACE_YR',
        'METER_1_LIFE_TOTAL', 'METER_2_LIFE_TOTAL',
        'mtCO2', 'mtCH4', 'mtN2O', 'mtCO2e'
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    df = df.rename(columns={
        'EQ_EQUIP_NO': 'ACCT_ID',
        'DEPT_DEPT_CODE': 'LowOrg',
        'LOC_STATION_LOC': 'Vehicle Location',
        'FUEL_TYPE': 'Subtype',
        'QTY_FUEL': 'Consumption',
        'FTK_DATE': 'FillDate'
    })

    df['Unit'] = 'gal'
    df['Sector'] = df.get('SECTOR', 'Fleet')

    df['FillDate'] = pd.to_datetime(df['FillDate'])
    df['Year'] = df['FillDate'].dt.year.astype(str)
    df['Month'] = df['FillDate'].dt.month
    df['FillDate'] = df['FillDate'].dt.date

    group_cols = [
        'ACCT_ID', 'Vehicle Class', 'LowOrg', 'Vehicle Location',
        'Subtype', 'FillDate', 'Unit', 'Sector', 'Year', 'Month'
    ]
    
    df_agg = df.groupby(group_cols, as_index=False)['Consumption'].sum()
    return df_agg


def calculate_emissions(df_agg, gwp_dict, ef_dict):
    """Expand GHG rows and calculate mtCO2e."""
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
        "Subtype", "GHG", "Consumption", "mtCO2e_calc",
        "Year", "Month", "LowOrg", "Unit", "Sector"
    ]
    return df_expanded[final_cols]
```

---

## Project Structure
```
backend/
├── app.py                 # Flask entry point
├── config.py              # Configuration
├── routes/
│   ├── __init__.py
│   ├── raw_data.py        # Interface 1
│   ├── data_update.py     # Interface 2
│   └── calculated.py      # Interface 3
├── services/
│   ├── __init__.py
│   ├── calculation.py     # Emissions calculation
│   ├── validation.py      # Column name validation
│   └── storage.py         # File read/write operations
├── data/
│   ├── raw/               # Original CSV files
│   ├── sessions/          # Modified data by session
│   └── calculated/        # Calculation outputs
├── logs/
│   └── backend.log        # Audit log
└── requirements.txt
```

---

## Test Scenarios

### Interface 1 Tests
1. GET /api/raw-data?type=fleetfuel → Returns valid FleetFuel JSON
2. GET /api/raw-data?type=efid → Returns valid EFID JSON
3. GET /api/raw-data?type=fleetfuel&invalid=true → Returns JSON with schema issues

### Interface 2 Tests
1. POST valid column rename → Success, saved to session file
2. POST column rename to number (e.g., "123") → Validation error
3. Verify `data/sessions/{sessionId}/fleetfuel_modified.csv` contains updated data
4. Verify `data/sessions/{sessionId}/changes.json` logs the change

### Interface 3 Tests
1. GET /api/calculated-data → Returns chart-ready response
2. Verify pieChart data grouped by Subtype
3. Verify barChart data grouped by FillDate
4. Verify formulaReference in rawData records
