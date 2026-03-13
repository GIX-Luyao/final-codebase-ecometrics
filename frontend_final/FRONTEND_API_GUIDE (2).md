# Frontend API Guide

This guide explains how to use the EcoMetrics Backend API, with detailed examples for column mappings and all endpoints.

**Base URL (Cloud)**: Configured via `REACT_APP_API_URL` environment variable in the frontend.
- Development: set in `Frontend_cloud/.env.development.local`
- Production: set in `Frontend_cloud/.env.production`

Example: `REACT_APP_API_URL=https://ecometrics-api.adc.seattle.gov/api`

If the variable is not set, the frontend falls back to `https://ecometrics-api.adc.seattle.gov/api`.

---

## Table of Contents

1. [User Flow Overview](#user-flow-overview)
2. [API Endpoints](#api-endpoints)
3. [Column Mapping Rules (Detailed)](#column-mapping-rules-detailed)
4. [Filter Configuration](#filter-configuration)
5. [Complete Request Examples](#complete-request-examples)
6. [Error Handling](#error-handling)

> **Cloud Note:** In cloud mode, all files (uploads and reference data) are stored in AWS S3, not on the server's local disk. The `filePath` field in API responses will be in `s3://bucket/key` format.

---

## User Flow Overview

This section describes the typical user journey through the EcoMetrics system and which APIs to call at each step.

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FIRST-TIME USER FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ 1. Browse Source │───▶│ 2. Create        │───▶│ 3. Configure     │       │
│  │    Files         │    │    Session       │    │    Mappings      │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│          │                                                │                  │
│          ▼                                                ▼                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ Preview columns  │    │                  │    │ 4. Run           │       │
│  │ & sample data    │    │                  │◀───│    Calculation   │       │
│  └──────────────────┘    │                  │    └──────────────────┘       │
│                          │                  │             │                  │
│                          │  5. View Results │◀────────────┘                  │
│                          │     & Charts     │                                │
│                          └──────────────────┘                                │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           RETURNING USER FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐                               │
│  │ 6. List Existing │───▶│ View Saved       │                               │
│  │    Sessions      │    │ Analytics        │                               │
│  └──────────────────┘    └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 0: Upload a Source File (Cloud Mode)

**Purpose:** User uploads a local CSV or Excel file to S3 so the backend can process it.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| Upload file | `POST /api/files/upload` | Uploads file to S3 and returns a `sourceId` |

**Request:** `multipart/form-data`
- `file`: the File object from the browser file input
- `folder`: `"CONSUMPTION"` or `"REFERENCE"`

**Response:**
```json
{
  "success": true,
  "sourceId": "pse_data_2022",
  "name": "PSE_Data_2022",
  "fileName": "PSE_Data_2022.xlsx",
  "folder": "CONSUMPTION",
  "sheets": ["Sheet1", "Data"]
}
```

**Important:**
- Store the returned `sourceId` — it is used to reference the file in all subsequent API calls (`preview`, `saveSourceConfig`, `calculate`)
- The `sheets` field is only present for Excel files; null for CSV
- In cloud mode, the file is stored in `s3://spu-emissions-raw-data/{folder}/{filename}`

---

### Step 1: Browse Source Files (First-Time User)

**Purpose:** User explores available data files to decide which to work with.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| List all source files | `GET /api/sources` | Returns list of CSV/XLSX files from S3 |
| Preview a source file | `GET /api/sources/{sourceId}/preview` | Returns column metadata and first 10 rows |

**Typical Frontend Flow:**
1. Display list of available source files with file names and sizes
2. When user clicks a file, show column names, data types, and sample values
3. User decides which file(s) to use for emissions calculation

**Cloud Note:** In cloud mode, `filePath` in the response is an S3 URI (e.g., `s3://spu-emissions-raw-data/CONSUMPTION/PSE_Data_2022.xlsx`), not a local path.

---

### Step 2: Create a Session

**Purpose:** User creates a workspace to save their configuration and calculations.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| Create new session | `POST /api/sessions` | Creates a session with name and description |

**Request Body:**
```json
{
  "name": "Q4 2022 Fleet Analysis",
  "description": "Fleet fuel emissions for Q4 2022"
}
```

**What happens:**
- A unique `sessionId` is generated (e.g., `sess_a1b2c3`)
- Session directory is created to store configs and results
- Session appears in the session list for future access

---

### Step 3: Configure Column Mappings

**Purpose:** User defines how source columns map to the standard emissions schema.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| Save configuration | `POST /api/sessions/{sessionId}/sources/{sourceId}/config` | Validates and saves column mapping |
| Get saved config | `GET /api/sessions/{sessionId}/sources/{sourceId}/config` | Retrieves previously saved config |

**Frontend should provide UI for:**
- Mapping source columns to required fields (ACCT_ID, Consumption, Unit, Subtype, Year, EF_ID)
- Setting static values (e.g., Unit = "gal")
- Configuring GHG expansion (CO2, CH4, N2O)
- Building EF_ID patterns (e.g., `Fuel_{Subtype}_20XX_{GHG}`)
- Adding filters to exclude certain rows
- Joining with reference tables (e.g., LOB lookup)

**See:** [Column Mapping Rules](#column-mapping-rules-detailed) for all mapping options.

---

### Step 4: Run Emissions Calculation

**Purpose:** Execute the calculation pipeline to compute CO2-equivalent emissions.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| Run calculation | `POST /api/sessions/{sessionId}/calculate` | Processes sources and calculates emissions |

**Request Body:**
```json
{
  "sources": ["fleetfuel_2019_2022"],
  "gwpVersion": "AR5"
}
```

**What happens internally:**
1. Load raw data from source file
2. Apply filters
3. Map columns to standard schema
4. Expand GHG rows (1 row → 3 rows for CO2/CH4/N2O)
5. Apply deferred patterns (EF_ID with {GHG})
6. Aggregate consumption
7. Join emission factors from EFID table
8. Join GWP values
9. Calculate: `mtCO2e = Consumption × GHG_MTperUnit × GWP`
10. Save results to Parquet file

**Response includes:**
- Calculation ID and timestamps
- Row counts (input vs output)
- Total mtCO2e for each source
- Aggregated totals

---

### Step 5: View Analytics & Charts

**Purpose:** Display calculation results in summary and chart formats.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| Get summary | `GET /api/sessions/{sessionId}/analytics/summary` | Aggregated totals by sector, year, subtype, GHG |
| Get chart data | `POST /api/sessions/{sessionId}/analytics/chart-data` | Chart-ready data (pie, bar, line, stacked_bar) |

**Summary Response includes:**
- Total mtCO2e across all sources
- Breakdown by Sector, Year, Subtype, GHG
- Year range

**Chart Data Request:**
```json
{
  "chartType": "pie",
  "metric": "mtCO2e_calc",
  "groupBy": "Subtype",
  "filters": [{"column": "Year", "operator": "=", "value": 2022}]
}
```

**Available chart types:** `pie`, `bar`, `line`, `stacked_bar`

---

### Step 6: Return to Existing Sessions (Returning User)

**Purpose:** User returns to view or continue work on previous sessions.

| Action | API Endpoint | Description |
|--------|--------------|-------------|
| List all sessions | `GET /api/sessions` | Returns all sessions with summary info |
| Get session details | `GET /api/sessions/{sessionId}` | Full session metadata including sources and calculations |
| View saved analytics | `GET /api/sessions/{sessionId}/analytics/summary` | Retrieve previously calculated results |
| Delete session | `DELETE /api/sessions/{sessionId}` | Remove session and all associated data |

**Session List Response includes:**
- Session ID, name, description
- Creation date
- Number of sources configured
- Whether calculations have been run
- Total mtCO2e (if calculated)

---

### Quick Reference: API Endpoints by Step

| Step | User Action | API Call |
|------|-------------|----------|
| 0 | Upload file to S3 | `POST /api/files/upload` |
| 1a | Browse files | `GET /api/sources` |
| 1b | Preview file | `GET /api/sources/{sourceId}/preview?sheet=Sheet1` |
| 2 | Create session | `POST /api/sessions` |
| 3a | Save config | `POST /api/sessions/{sessionId}/sources/{sourceId}/config` |
| 3b | Get config | `GET /api/sessions/{sessionId}/sources/{sourceId}/config` |
| 4 | Calculate | `POST /api/sessions/{sessionId}/calculate` |
| 5a | View summary | `GET /api/sessions/{sessionId}/analytics/summary` |
| 5b | Get chart | `POST /api/sessions/{sessionId}/analytics/chart-data` |
| 6a | List sessions | `GET /api/sessions` |
| 6b | Get session | `GET /api/sessions/{sessionId}` |
| 6c | Delete session | `DELETE /api/sessions/{sessionId}` |

---

## API Endpoints

### 1. Upload File

```http
POST /api/files/upload
Content-Type: multipart/form-data
```

**Form Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | File object (`.csv`, `.xlsx`, `.xls`) |
| `folder` | No | `"CONSUMPTION"` (default) or `"REFERENCE"` |

**Response (HTTP 201):**
```json
{
  "success": true,
  "sourceId": "pse_data_2022",
  "name": "PSE_Data_2022",
  "fileName": "PSE_Data_2022.xlsx",
  "folder": "CONSUMPTION",
  "sheets": ["Sheet1", "Data"]
}
```

**JavaScript example:**
```js
async function uploadFile(file, folder) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('folder', folder); // "CONSUMPTION" or "REFERENCE"

  const response = await fetch(`${API_BASE_URL}/files/upload`, {
    method: 'POST',
    body: formData,
    // Do NOT set Content-Type — browser sets it automatically with correct boundary
  });
  const data = await response.json();
  if (!data.success) throw new Error(data.error?.message || 'Upload failed');
  return data; // { sourceId, name, fileName, folder, sheets }
}
```

**Error codes:**
| Code | HTTP | Cause |
|------|------|-------|
| `MISSING_PARAMETER` | 400 | No file in request |
| `INVALID_PARAMETER` | 400 | folder not CONSUMPTION or REFERENCE |
| `INVALID_FILE_TYPE` | 400 | File extension not .csv/.xlsx/.xls |
| `STORAGE_ERROR` | 500 | S3 write failed |

---

### 2. Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "service": "ecometrics-backend",
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### 2. List Source Files

```http
GET /api/sources
```

**Response:**
```json
{
  "success": true,
  "sources": [
    {
      "sourceId": "fleetfuel_2019_2022",
      "name": "FleetFuel_2019-2022",
      "filePath": "s3://spu-emissions-raw-data/CONSUMPTION/FleetFuel_2019-2022.xlsx",
      "fileSize": 1850000,
      "lastModified": "2026-01-15T10:30:00Z",
      "sheets": ["Sheet1", "Data"]
    }
  ]
}
```

---

### 3. Preview Source Data

```http
GET /api/sources/{sourceId}/preview?sheet=Sheet1
```

**Query Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `sheet` | No | Excel sheet name (for .xlsx files) |

**Response:**
```json
{
  "success": true,
  "sourceId": "fleetfuel_2019_2022",
  "fileName": "FleetFuel_2019-2022.xlsx",
  "sheet": "Sheet1",
  "totalRows": 15420,
  "columns": [
    {
      "name": "EQ_EQUIP_NO",
      "type": "string",
      "sampleValues": ["1011", "4386", "5504"],
      "nullCount": 0,
      "uniqueCount": 342
    },
    {
      "name": "QTY_FUEL",
      "type": "float",
      "sampleValues": [25.5, 45.2, 12.8],
      "nullCount": 5,
      "min": 0.5,
      "max": 450.2,
      "mean": 35.6
    }
  ],
  "preview": [
    {"EQ_EQUIP_NO": "1011", "QTY_FUEL": 25.5, "FUEL_TYPE": "Gasoline", ...}
  ]
}
```

---

### 4. Create Session

```http
POST /api/sessions
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Q4 2022 Fleet Analysis",
  "description": "Fleet emissions for Q4 2022"
}
```

**Response:**
```json
{
  "success": true,
  "sessionId": "sess_a1b2c3",
  "name": "Q4 2022 Fleet Analysis",
  "createdAt": "2026-01-25T10:00:00Z"
}
```

---

### 5. List Sessions

```http
GET /api/sessions
```

**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "sessionId": "sess_a1b2c3",
      "name": "Q4 2022 Fleet Analysis",
      "createdAt": "2026-01-25T10:00:00Z",
      "lastModified": "2026-01-25T14:30:00Z",
      "status": "active",
      "configuredSources": 2,
      "hasCalculations": true,
      "totalMtCO2e": 4523.847
    }
  ]
}
```

---

### 6. Get Session Details

```http
GET /api/sessions/{sessionId}
```

---

### 7. Delete Session

```http
DELETE /api/sessions/{sessionId}
```

**Response:**
```json
{
  "success": true,
  "sessionId": "sess_a1b2c3",
  "deletedFiles": [
    "data/sessions/sess_a1b2c3/session_meta.json",
    "data/calculated/sess_a1b2c3/fleet_emissions.parquet"
  ]
}
```

---

### 8. Save Column Mapping Configuration

```http
POST /api/sessions/{sessionId}/sources/{sourceId}/config
Content-Type: application/json
```

**See [Column Mapping Rules](#column-mapping-rules-detailed) for detailed request body structure.**

**Response:**
```json
{
  "success": true,
  "sessionId": "sess_a1b2c3",
  "sourceId": "fleetfuel_2019_2022",
  "configVersion": 1,
  "validation": {
    "status": "valid",
    "warnings": ["Column 'Vehicle Class' has 5% null values"],
    "errors": []
  },
  "savedAt": "2026-01-25T10:15:00Z"
}
```

---

### 9. Get Saved Configuration

```http
GET /api/sessions/{sessionId}/sources/{sourceId}/config
```

---

### 10. Run Emissions Calculation

```http
POST /api/sessions/{sessionId}/calculate
Content-Type: application/json
```

**Request Body:**
```json
{
  "sources": ["fleetfuel_2019_2022", "pse_2022"],
  "gwpVersion": "AR5"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `sources` | No | All configured | Array of source IDs to calculate |
| `gwpVersion` | No | `"AR5"` | GWP version: `"AR5"` or `"AR4"` |

**Response:**
```json
{
  "success": true,
  "sessionId": "sess_a1b2c3",
  "calculationId": "calc_a1b2c3_d4e5f6",
  "status": "completed",
  "startedAt": "2026-01-25T10:30:00Z",
  "completedAt": "2026-01-25T10:30:45Z",
  "results": {
    "fleetfuel_2019_2022": {
      "inputRows": 12850,
      "outputRows": 38550,
      "totalMtCO2e": 4523.847,
      "filePath": "data/calculated/sess_a1b2c3/fleetfuel_2019_2022_emissions.parquet"
    }
  },
  "aggregated": {
    "totalMtCO2e": 4523.847,
    "totalSources": 1,
    "yearRange": {"min": 2019, "max": 2022}
  }
}
```

---

### 11. Get Analytics Summary

```http
GET /api/sessions/{sessionId}/analytics/summary
```

**Response:**
```json
{
  "success": true,
  "sessionId": "sess_a1b2c3",
  "summary": {
    "totalMtCO2e": 7822.303,
    "totalSources": 2,
    "yearRange": {"min": 2019, "max": 2022},
    "bySector": {"Fleet": 4523.847, "Facilities": 3298.456},
    "byYear": {"2019": 1823.45, "2020": 1756.89, "2021": 1892.34, "2022": 2349.62},
    "bySubtype": {"Diesel": 1842.34, "Gasoline": 1523.67, "B20": 456.78},
    "byGHG": {"CO2": 7500.0, "CH4": 200.0, "N2O": 122.303}
  }
}
```

---

### 12. Get Chart Data

```http
POST /api/sessions/{sessionId}/analytics/chart-data
Content-Type: application/json
```

**Request Body:**
```json
{
  "chartType": "pie",
  "metric": "mtCO2e_calc",
  "groupBy": "Subtype",
  "filters": [
    {"column": "Year", "operator": "=", "value": 2022}
  ]
}
```

| Field | Required | Default | Options |
|-------|----------|---------|---------|
| `chartType` | No | `"pie"` | `"pie"`, `"bar"`, `"line"`, `"stacked_bar"` |
| `metric` | No | `"mtCO2e_calc"` | Any numeric column |
| `groupBy` | No | `"Subtype"` | Any column name |
| `filters` | No | `[]` | Array of filter objects |
| `stackBy` | No | `"GHG"` | For `stacked_bar` only |

**Pie Chart Response:**
```json
{
  "success": true,
  "chartType": "pie",
  "title": "mtCO2e_calc by Subtype (Year=2022)",
  "data": [
    {"label": "Diesel", "value": 1842.34, "percentage": 45.2, "color": "#FF6384"},
    {"label": "Gasoline", "value": 1523.67, "percentage": 37.4, "color": "#36A2EB"}
  ],
  "total": 4075.24
}
```

**Bar/Line Chart Response:**
```json
{
  "success": true,
  "chartType": "bar",
  "title": "mtCO2e_calc by Year",
  "labels": ["2019", "2020", "2021", "2022"],
  "datasets": [
    {"label": "mtCO2e_calc", "data": [1823.45, 1756.89, 1892.34, 2349.62], "backgroundColor": "#FF6384"}
  ]
}
```

---

## Column Mapping Rules (Detailed)

The `columnMappings` object defines how to transform source columns to the standard schema. Each key is the **target column name**, and the value is a **mapping rule object**.

### Required Target Columns

| Column | Type | Description |
|--------|------|-------------|
| `ACCT_ID` | string | Account/Asset identifier |
| `Year` | integer | Reporting year |
| `Consumption` | float | Activity quantity |
| `Unit` | string | Unit of consumption |
| `Subtype` | string | Fuel/utility type |
| `EF_ID` | string | Emission factor identifier (direct or pattern-based) |
| `GHG` | string | Greenhouse gas (use GHG expansion) |

### Mapping Rule Types

---

#### 1. Direct Column Mapping (`sourceColumn`)

Copy values directly from a source column.

```json
{
  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "EQ_EQUIP_NO"
    },
    "Consumption": {
      "sourceColumn": "QTY_FUEL"
    },
    "Subtype": {
      "sourceColumn": "FUEL_TYPE"
    },
    "LowOrg": {
      "sourceColumn": "DEPT_DEPT_CODE"
    },
    "Vehicle Location": {
      "sourceColumn": "LOC_STATION_LOC"
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `sourceColumn` | Yes | Name of column in source data |

---

#### 2. Static Value (`staticValue`)

Assign a constant value to all rows.

```json
{
  "columnMappings": {
    "Unit": {
      "staticValue": "gal"
    },
    "Sector": {
      "staticValue": "Fleet"
    },
    "SourceNotes": {
      "staticValue": "Fleet Fuel Data 2019-2022"
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `staticValue` | Yes | Constant value (string, number, or boolean) |

**Common Use Cases:**
- `Unit`: `"gal"`, `"KWH"`, `"kg"`, `"therms"`
- `Sector`: `"Fleet"`, `"Facilities"`, `"Landfill"`

---

#### 3. Derived Value (`derivedFrom` + `extractType`)

Extract a component from a date/datetime column.

```json
{
  "columnMappings": {
    "Year": {
      "derivedFrom": "FTK_DATE",
      "extractType": "year"
    },
    "Month": {
      "derivedFrom": "FTK_DATE",
      "extractType": "month"
    },
    "Quarter": {
      "derivedFrom": "FTK_DATE",
      "extractType": "quarter"
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `derivedFrom` | Yes | Source column containing date/datetime |
| `extractType` | Yes | What to extract (see table below) |

**Supported `extractType` Values:**

| extractType | Output | Example (from "2021-10-15") |
|-------------|--------|------------------------------|
| `year` | Integer year | `2021` |
| `month` | Integer month (1-12) | `10` |
| `day` | Integer day of month | `15` |
| `quarter` | Integer quarter (1-4) | `4` |
| `dayofweek` | Integer day of week (Monday=0) | `4` (Friday) |

---

#### 4. Reference Table Join (`referenceTable`)

Look up a value from an external reference table.

```json
{
  "columnMappings": {
    "LOB": {
      "referenceTable": "data/raw/REFERENCE/LOB_LowOrgList.xlsx",
      "refSheet": "Sheet1",
      "refKey": "LowOrg",
      "sourceKey": "DEPT_DEPT_CODE",
      "columnJoined": "LOB"
    },
    "Division": {
      "referenceTable": "data/raw/REFERENCE/LOB_LowOrgList.xlsx",
      "refKey": "LowOrg",
      "sourceKey": "DEPT_DEPT_CODE",
      "columnJoined": "Division"
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `referenceTable` | Yes | Path to reference file (relative to backend root) |
| `refSheet` | No | Sheet name for Excel files |
| `refKey` | Yes | Key column in reference table |
| `sourceKey` | Yes | Key column in source data |
| `columnJoined` | Yes | Column to retrieve from reference table |

**How it works:**
1. For each row in source data, get the value of `sourceKey`
2. Find matching row in reference table where `refKey` equals that value
3. Return the value from `columnJoined`

---

#### 5. GHG Expansion (`ghgType`)

Expand each row into multiple rows, one per greenhouse gas.

```json
{
  "columnMappings": {
    "GHG": {
      "ghgType": ["CO2", "CH4", "N2O"]
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `ghgType` | Yes | Array of GHG names to expand |

**How it works:**
- Input: 1 row with `Consumption = 25.5`
- Output: 3 rows, each with `Consumption = 25.5` and `GHG` = `"CO2"`, `"CH4"`, `"N2O"`

**Common GHG Lists:**
- Combustion fuels: `["CO2", "CH4", "N2O"]`
- Electricity: `["CO2", "CH4", "N2O"]`
- Refrigerants: `["R-134A"]` or `["R-410A"]`

---

#### 6. Pattern-Based Column Construction (`pattern`)

Construct a column value by combining multiple columns using a pattern template. This is especially useful for building `EF_ID` values.

**Option A: Use existing source column directly**
```json
{
  "columnMappings": {
    "EF_ID": {
      "sourceColumn": "EMISSION_FACTOR_ID"
    }
  }
}
```

**Option B: Construct from pattern with placeholders**
```json
{
  "columnMappings": {
    "EF_ID": {
      "pattern": "Fuel_{Subtype}_20XX_{GHG}"
    }
  }
}
```

| Property | Required | Description |
|----------|----------|-------------|
| `pattern` | Yes | Template string with `{column_name}` placeholders |

**How it works:**
1. Placeholders like `{Subtype}` are replaced with values from that column
2. Placeholders can reference source columns OR already-mapped target columns
3. Pattern is applied row-by-row

**Special Handling for `{GHG}` Placeholder:**

The `{GHG}` placeholder is special because the GHG column doesn't exist in the source data—it's created by the system during GHG expansion. The backend automatically handles this:

1. Patterns **without** `{GHG}` are processed immediately during column mapping
2. Patterns **with** `{GHG}` are **deferred** until after GHG expansion
3. Once GHG expansion creates rows with `GHG = "CO2"`, `"CH4"`, `"N2O"`, the deferred pattern is applied

This means you can safely use `{GHG}` in your patterns even though the GHG column doesn't exist yet:

```json
{
  "columnMappings": {
    "GHG": {"ghgType": ["CO2", "CH4", "N2O"]},
    "EF_ID": {"pattern": "Fuel_{Subtype}_20XX_{GHG}"}
  }
}
```

The system processes this as:
1. First: Other columns mapped (Subtype, etc.)
2. Then: GHG expansion creates 3 rows per input (one each for CO2, CH4, N2O)
3. Finally: EF_ID pattern is applied using the now-available GHG values

**Pattern Placeholder Examples:**

| Placeholder | Source | Example Value |
|-------------|--------|---------------|
| `{Subtype}` | Mapped column | `Gasoline`, `Diesel`, `B20` |
| `{GHG}` | From GHG expansion | `CO2`, `CH4`, `N2O` |
| `{Year}` | Mapped column | `2022` |
| `{Sector}` | Mapped column | `Fleet`, `Facilities` |
| `{FUEL_TYPE}` | Source column | `Gasoline` |

**Common EF_ID Patterns:**

| Data Type | Pattern | Example Output |
|-----------|---------|----------------|
| Fleet Fuel | `Fuel_{Subtype}_20XX_{GHG}` | `Fuel_Gasoline_20XX_CO2` |
| Electricity (PSE) | `Elec_PSE_{Year}_{GHG}` | `Elec_PSE_2022_CO2` |
| Electricity (SCL) | `Elec_SCL_{Year}_{GHG}` | `Elec_SCL_2022_CO2` |
| Natural Gas | `NG_{Subtype}_{GHG}` | `NG_Pipeline_CO2` |
| Refrigerants | `FacAC_{Subtype}_20XX` | `FacAC_R-134A_20XX` |

**Complete Example with EF_ID Pattern:**
```json
{
  "columnMappings": {
    "ACCT_ID": {"sourceColumn": "EQ_EQUIP_NO"},
    "Consumption": {"sourceColumn": "QTY_FUEL"},
    "Subtype": {"sourceColumn": "FUEL_TYPE"},
    "Year": {"derivedFrom": "FTK_DATE", "extractType": "year"},
    "Unit": {"staticValue": "gal"},
    "Sector": {"staticValue": "Fleet"},
    "GHG": {"ghgType": ["CO2", "CH4", "N2O"]},
    "EF_ID": {"pattern": "Fuel_{Subtype}_20XX_{GHG}"}
  }
}
```

**Processing Order:**
1. Direct mappings, static values, derived values are processed first
2. GHG expansion creates rows with GHG column
3. Pattern mappings are processed last (can reference mapped columns like `Subtype`, `GHG`)

---

## Filter Configuration

Filters are applied to raw data **before** column mapping.

```json
{
  "filters": [
    {
      "column": "FTK_DATE",
      "operator": ">=",
      "value": "2019-01-01"
    },
    {
      "column": "QTY_FUEL",
      "operator": ">",
      "value": 0
    },
    {
      "column": "FUEL_TYPE",
      "operator": "in",
      "value": ["Gasoline", "Diesel", "B20"]
    }
  ]
}
```

### Supported Operators

| Operator | Description | Value Type | Example |
|----------|-------------|------------|---------|
| `=` | Equals | any | `{"column": "Year", "operator": "=", "value": 2022}` |
| `!=` | Not equals | any | `{"column": "Status", "operator": "!=", "value": "Inactive"}` |
| `>` | Greater than | number/date | `{"column": "QTY_FUEL", "operator": ">", "value": 0}` |
| `>=` | Greater than or equal | number/date | `{"column": "FTK_DATE", "operator": ">=", "value": "2019-01-01"}` |
| `<` | Less than | number/date | `{"column": "Year", "operator": "<", "value": 2023}` |
| `<=` | Less than or equal | number/date | `{"column": "QTY_FUEL", "operator": "<=", "value": 500}` |
| `in` | In list | array | `{"column": "FUEL_TYPE", "operator": "in", "value": ["Gasoline", "Diesel"]}` |
| `not_in` | Not in list | array | `{"column": "Status", "operator": "not_in", "value": ["Deleted", "Archived"]}` |
| `is_null` | Is null/empty | (none) | `{"column": "Notes", "operator": "is_null"}` |
| `not_null` | Is not null | (none) | `{"column": "QTY_FUEL", "operator": "not_null"}` |

---

## Complete Request Examples

### Example 1: Fleet Fuel Data (EF_ID from pattern)

```json
POST /api/sessions/sess_a1b2c3/sources/fleetfuel_2019_2022/config

{
  "sourceSheet": "Sheet1",

  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "EQ_EQUIP_NO"
    },
    "Consumption": {
      "sourceColumn": "QTY_FUEL"
    },
    "Subtype": {
      "sourceColumn": "FUEL_TYPE"
    },
    "Year": {
      "derivedFrom": "FTK_DATE",
      "extractType": "year"
    },
    "Month": {
      "derivedFrom": "FTK_DATE",
      "extractType": "month"
    },
    "Unit": {
      "staticValue": "gal"
    },
    "Sector": {
      "staticValue": "Fleet"
    },
    "GHG": {
      "ghgType": ["CO2", "CH4", "N2O"]
    },
    "EF_ID": {
      "pattern": "Fuel_{Subtype}_20XX_{GHG}"
    },
    "LowOrg": {
      "sourceColumn": "DEPT_DEPT_CODE"
    },
    "LOB": {
      "referenceTable": "data/raw/REFERENCE/LOB_LowOrgList.xlsx",
      "refKey": "LowOrg",
      "sourceKey": "DEPT_DEPT_CODE",
      "columnJoined": "LOB"
    },
    "Vehicle Location": {
      "sourceColumn": "LOC_STATION_LOC"
    },
    "Vehicle Class": {
      "sourceColumn": "Vehicle Class"
    }
  },

  "filters": [
    {
      "column": "FTK_DATE",
      "operator": ">=",
      "value": "2019-01-01"
    },
    {
      "column": "QTY_FUEL",
      "operator": ">",
      "value": 0
    }
  ],

  "efidLookup": {
    "sectorFilter": "Fleet"
  }
}
```

### Example 2: PSE Electricity Data (EF_ID with Year)

```json
POST /api/sessions/sess_a1b2c3/sources/pse_2022/config

{
  "sourceSheet": "Data",

  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "Account_Number"
    },
    "Consumption": {
      "sourceColumn": "KWH_Usage"
    },
    "Subtype": {
      "staticValue": "PSE"
    },
    "Year": {
      "sourceColumn": "Bill_Year"
    },
    "Month": {
      "sourceColumn": "Bill_Month"
    },
    "Unit": {
      "staticValue": "KWH"
    },
    "Sector": {
      "staticValue": "Facilities"
    },
    "GHG": {
      "ghgType": ["CO2", "CH4", "N2O"]
    },
    "EF_ID": {
      "pattern": "Elec_PSE_{Year}_{GHG}"
    },
    "LowOrg": {
      "sourceColumn": "Dept_Code"
    },
    "LOB": {
      "referenceTable": "data/raw/REFERENCE/LOB_LowOrgList.xlsx",
      "refKey": "LowOrg",
      "sourceKey": "Dept_Code",
      "columnJoined": "LOB"
    }
  },

  "filters": [
    {
      "column": "KWH_Usage",
      "operator": ">",
      "value": 0
    }
  ],

  "efidLookup": {
    "sectorFilter": "Facilities"
  }
}
```

### Example 3: Refrigerant Data (Single GHG)

```json
POST /api/sessions/sess_a1b2c3/sources/ac_facilities/config

{
  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "Equipment_ID"
    },
    "Consumption": {
      "sourceColumn": "Refrigerant_kg"
    },
    "Subtype": {
      "sourceColumn": "Refrigerant_Type"
    },
    "Year": {
      "sourceColumn": "Service_Year"
    },
    "Unit": {
      "staticValue": "kg"
    },
    "Sector": {
      "staticValue": "Facilities"
    },
    "GHG": {
      "ghgType": ["R-134A"]
    },
    "EF_ID": {
      "pattern": "FacAC_{Subtype}_20XX"
    }
  },

  "efidLookup": {
    "sectorFilter": "Facilities"
  }
}
```

### Example 4: Data with Existing EF_ID Column

If your source data already has a complete EF_ID column (including GHG suffix):

```json
POST /api/sessions/sess_a1b2c3/sources/preprocessed_data/config

{
  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "Account_ID"
    },
    "Consumption": {
      "sourceColumn": "Usage_Amount"
    },
    "Subtype": {
      "sourceColumn": "Fuel_Type"
    },
    "Year": {
      "sourceColumn": "Report_Year"
    },
    "Unit": {
      "sourceColumn": "Usage_Unit"
    },
    "Sector": {
      "sourceColumn": "Business_Sector"
    },
    "GHG": {
      "ghgType": ["CO2", "CH4", "N2O"]
    },
    "EF_ID": {
      "sourceColumn": "Emission_Factor_ID"
    }
  }
}
```

### Example 5: Concatenate Source EFID Column with GHG

If your source data has a **partial** EF_ID column (e.g., `"Fuel_Gasoline_20XX"`) and you need to append the GHG suffix:

**Source data example:**
| EFID_Base | Fuel_Type | QTY |
|-----------|-----------|-----|
| Fuel_Gasoline_20XX | Gasoline | 100 |
| Fuel_Diesel_20XX | Diesel | 50 |

**Desired EF_ID output:** `Fuel_Gasoline_20XX_CO2`, `Fuel_Gasoline_20XX_CH4`, etc.

```json
POST /api/sessions/sess_a1b2c3/sources/partial_efid_data/config

{
  "columnMappings": {
    "ACCT_ID": {
      "sourceColumn": "Account_ID"
    },
    "Consumption": {
      "sourceColumn": "QTY"
    },
    "Subtype": {
      "sourceColumn": "Fuel_Type"
    },
    "Year": {
      "sourceColumn": "Report_Year"
    },
    "Unit": {
      "staticValue": "gal"
    },
    "Sector": {
      "staticValue": "Fleet"
    },
    "GHG": {
      "ghgType": ["CO2", "CH4", "N2O"]
    },
    "EF_ID": {
      "pattern": "{EFID_Base}_{GHG}"
    }
  }
}
```

**How it works:**
1. `EFID_Base` is read directly from the source column (e.g., `"Fuel_Gasoline_20XX"`)
2. GHG expansion creates 3 rows per input with `GHG` = `"CO2"`, `"CH4"`, `"N2O"`
3. The pattern `{EFID_Base}_{GHG}` combines them into:
   - `Fuel_Gasoline_20XX_CO2`
   - `Fuel_Gasoline_20XX_CH4`
   - `Fuel_Gasoline_20XX_N2O`

**Note:** Since the pattern contains `{GHG}`, it is automatically deferred until after GHG expansion, so the GHG values are available when constructing EF_ID.

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "errors": ["Specific error 1", "Specific error 2"]
    },
    "timestamp": "2026-01-25T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `MISSING_PARAMETER` | 400 | Required field missing in request |
| `VALIDATION_ERROR` | 400 | Config validation failed (see `details.errors`) |
| `INVALID_PARAMETER` | 400 | Invalid parameter value |
| `SOURCE_NOT_FOUND` | 404 | Source file not found |
| `SESSION_NOT_FOUND` | 404 | Session does not exist |
| `CONFIG_NOT_FOUND` | 404 | No config saved for source |
| `NO_DATA` | 404 | No calculated data available |
| `MISSING_EF_ID` | 422 | EF_ID column missing after mapping (ensure EF_ID is in columnMappings) |
| `MISSING_EMISSION_FACTOR` | 422 | EF_ID not found in EFID table |
| `MISSING_GWP` | 422 | GWP value not found for GHG |
| `CALCULATION_ERROR` | 422/500 | Error during calculation |
| `STORAGE_ERROR` | 500 | File system error |

### Validation Error Example

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Configuration validation failed",
    "details": {
      "errors": [
        "Missing required column mapping: 'ACCT_ID'",
        "Missing required column mapping: 'EF_ID'",
        "Source column 'INVALID_COLUMN' does not exist in raw data",
        "Reference table not found: data/raw/REFERENCE/Missing.xlsx"
      ]
    },
    "timestamp": "2026-01-25T10:30:00Z"
  }
}
```

**Note:** `EF_ID` is a required column. You must map it either:
- Using `sourceColumn` if your data already has an EF_ID column
- Using `pattern` to construct it (e.g., `"Fuel_{Subtype}_20XX_{GHG}"`)

---

## JavaScript/TypeScript Examples

### Fetch Wrapper

```typescript
const API_BASE = 'https://ecometrics-api.adc.seattle.gov/api';

async function apiCall<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error.message);
  }

  return data;
}
```

### Create Session and Configure Source

```typescript
// 1. Create session
const session = await apiCall<{sessionId: string}>('/sessions', {
  method: 'POST',
  body: JSON.stringify({
    name: 'Q4 2022 Analysis',
    description: 'Fleet emissions analysis'
  })
});

// 2. Save configuration
const config = await apiCall(`/sessions/${session.sessionId}/sources/fleet_fuel/config`, {
  method: 'POST',
  body: JSON.stringify({
    columnMappings: {
      ACCT_ID: { sourceColumn: 'EQ_EQUIP_NO' },
      Consumption: { sourceColumn: 'QTY_FUEL' },
      Subtype: { sourceColumn: 'FUEL_TYPE' },
      Year: { derivedFrom: 'FTK_DATE', extractType: 'year' },
      Unit: { staticValue: 'gal' },
      Sector: { staticValue: 'Fleet' },
      GHG: { ghgType: ['CO2', 'CH4', 'N2O'] },
      EF_ID: { pattern: 'Fuel_{Subtype}_20XX_{GHG}' }  // Required: construct from pattern
    },
    filters: [
      { column: 'QTY_FUEL', operator: '>', value: 0 }
    ],
    efidLookup: {
      sectorFilter: 'Fleet'
    }
  })
});

// 3. Run calculation
const result = await apiCall(`/sessions/${session.sessionId}/calculate`, {
  method: 'POST',
  body: JSON.stringify({
    sources: ['fleet_fuel'],
    gwpVersion: 'AR5'
  })
});

console.log(`Total emissions: ${result.aggregated.totalMtCO2e} mtCO2e`);
```
