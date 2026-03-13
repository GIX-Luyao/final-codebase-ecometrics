# Backend Modules Documentation

This document describes each module, file, and function in the EcoMetrics Backend.

---

## Directory Structure Overview

```
Backend_local/
├── app.py                      # Flask application entry point
├── config.py                   # Configuration management
├── routes/                     # API endpoint handlers
│   ├── sources.py              # Source file management
│   ├── sessions.py             # Session CRUD operations
│   ├── config.py               # Column mapping configuration
│   ├── calculate.py            # Emissions calculation trigger
│   └── analytics.py            # Summary and chart data
├── services/                   # Business logic layer
│   ├── storage.py              # File I/O operations
│   ├── validation.py           # Config and data validation
│   ├── mapping_engine.py       # Column transformation engine
│   └── calculation_engine.py   # Emissions calculation engine
├── data/                       # Data storage
│   ├── raw/CONSUMPTION/        # Source data files
│   ├── raw/REFERENCE/          # EFID, GWP, LOB reference tables
│   ├── sessions/               # Session configs and metadata
│   └── calculated/             # Parquet output files
└── tests/                      # Unit tests
```

---

## Core Files

### `app.py`
**Purpose**: Flask application entry point and factory.

| Function | Description |
|----------|-------------|
| `create_app()` | Creates and configures the Flask application with CORS, logging, and route registration |
| `setup_logging()` | Configures file and console logging |
| `SessionLogger` | Helper class for session-aware logging with component/action/status format |

### `config.py`
**Purpose**: Centralized configuration management.

| Attribute/Method | Description |
|------------------|-------------|
| `BASE_DIR` | Root directory of the backend |
| `DATA_DIR` | Path to `data/` directory |
| `CONSUMPTION_DIR` | Path to source files (`data/raw/CONSUMPTION/`) |
| `REFERENCE_DIR` | Path to reference tables (`data/raw/REFERENCE/`) |
| `SESSIONS_DIR` | Path to session data (`data/sessions/`) |
| `CALCULATED_DIR` | Path to calculation outputs (`data/calculated/`) |
| `EFID_FILE`, `GWP_FILE`, `LOB_FILE` | Paths to reference Excel files |
| `PORT` | Server port (default: 8000) |
| `ensure_directories()` | Creates required directories if they don't exist |
| `get_session_dir(session_id)` | Returns path to a specific session's directory |
| `get_calculated_dir(session_id)` | Returns path to a session's calculated output directory |

---

## Services Layer

### `services/storage.py`
**Purpose**: All file I/O operations for sources, sessions, configs, and calculated data.

| Function | Description |
|----------|-------------|
| **ID Generation** | |
| `generate_session_id()` | Creates unique session ID (e.g., `sess_a1b2c3`) |
| `generate_calculation_id(session_id)` | Creates unique calculation ID (e.g., `calc_a1b2c3_d4e5f6`) |
| **Source Operations** | |
| `list_source_files()` | Lists all CSV/XLSX files in CONSUMPTION directory with metadata |
| `get_source_path(source_id)` | Resolves source ID to file path |
| `load_raw_data(source_id, sheet_name)` | Loads source file as pandas DataFrame |
| **Reference Table Operations** | |
| `load_reference_table(table_name)` | Loads EFID, GWP, or LOB reference table |
| `load_custom_reference_table(file_path)` | Loads any reference file by path |
| **Session Operations** | |
| `load_session_registry()` | Loads master session list from `registry.json` |
| `save_session_registry(registry)` | Saves master session list |
| `create_session(name, description)` | Creates new session with metadata and directories |
| `get_session(session_id)` | Retrieves session metadata |
| `update_session(session_id, updates)` | Updates session metadata |
| `delete_session(session_id)` | Deletes session and all associated files |
| **Config Operations** | |
| `save_source_config(session_id, source_id, config)` | Saves column mapping config, returns version number |
| `get_source_config(session_id, source_id)` | Retrieves saved configuration |
| **Calculated Data Operations** | |
| `save_calculated_data(session_id, source_id, df)` | Saves DataFrame as Parquet file |
| `load_calculated_data(session_id, source_id)` | Loads single source's calculated data |
| `load_all_calculated_data(session_id)` | Loads and concatenates all calculated data for session |

---

### `services/validation.py`
**Purpose**: Validates configurations and data integrity.

| Function | Description |
|----------|-------------|
| `validate_config(config, source_id)` | Validates column mapping configuration. Returns `{valid, errors, warnings}`. Checks: required columns mapped, source columns exist, reference tables accessible, valid operators, etc. |
| `validate_data_after_mapping(df, column_mappings)` | Validates mapped data for null values and negative consumption |
| `validate_calculation_request(session_id, sources)` | Validates that session exists and all sources have configs |

---

### `services/mapping_engine.py`
**Purpose**: Transforms raw data to standard schema using declarative column mappings.

| Class/Function | Description |
|----------------|-------------|
| **MappingEngine** | Main class for data transformation |
| `__init__()` | Initializes with empty reference table cache |
| `apply_filters(df, filters)` | Applies filter conditions to DataFrame. Supports: `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `is_null`, `not_null` |
| `map_columns(df, column_mappings)` | Applies column mappings to create standard schema. Returns `(mapped_df, ghg_config, deferred_patterns)`. Patterns containing `{GHG}` are deferred. Supports 6 mapping types (see API Guide) |
| `apply_deferred_patterns(df, deferred_patterns)` | Applies pattern-based columns that were deferred until after GHG expansion (patterns containing `{GHG}`) |
| `expand_ghg_rows(df, ghg_config)` | Expands each row into multiple rows (one per GHG type). 1 row × 3 GHGs = 3 rows |
| `aggregate_consumption(df)` | Groups by all dimension columns and sums Consumption |
| `process_mappings(df, config)` | Full pipeline: filter → map → expand → apply deferred patterns → aggregate |
| `clear_cache()` | Clears reference table cache |

---

### `services/calculation_engine.py`
**Purpose**: Calculates GHG emissions using emission factors and GWP values.

| Class/Function | Description |
|----------------|-------------|
| **CalculationEngine** | Main class for emissions calculation |
| `__init__()` | Initializes with MappingEngine and empty reference tables |
| `load_reference_tables()` | Loads EFID and GWP tables into memory |
| `construct_efid(df, efid_config)` | (Legacy) Builds EF_ID column from pattern. Note: EF_ID is now a required user-provided column mapping |
| `join_emission_factors(df, efid_config)` | Joins EFID table to get `GHG_MTperUnit`. Optionally filters by sector |
| `join_gwp(df, gwp_version)` | Maps GHG names to GWP values (AR5 or AR4) |
| `calculate_emissions(df)` | Calculates `mtCO2e_calc = Consumption × GHG_MTperUnit × GWP`. Adds `formulaReference` for transparency |
| `process_source(source_id, config, gwp_version)` | Full pipeline for one source: load → map → validate EF_ID → join factors → calculate |
| `calculate_for_session(session_id, source_ids, gwp_version)` | Processes multiple sources, saves results, updates session metadata. Returns calculation summary |

---

## Routes Layer

### `routes/sources.py`
**Purpose**: Source file listing and preview endpoints.

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /api/sources` | `list_sources()` | Lists all source files in CONSUMPTION directory |
| `GET /api/sources/<source_id>/preview` | `preview_source()` | Returns column metadata and first 10 rows |

---

### `routes/sessions.py`
**Purpose**: Session CRUD operations.

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /api/sessions` | `list_sessions()` | Lists all sessions with summary info |
| `POST /api/sessions` | `create_session()` | Creates new session |
| `GET /api/sessions/<session_id>` | `get_session()` | Gets full session metadata |
| `DELETE /api/sessions/<session_id>` | `delete_session()` | Deletes session and all data |

---

### `routes/config.py`
**Purpose**: Column mapping configuration endpoints.

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `POST /api/sessions/<session_id>/sources/<source_id>/config` | `save_config()` | Validates and saves column mapping configuration |
| `GET /api/sessions/<session_id>/sources/<source_id>/config` | `get_config()` | Retrieves saved configuration |

---

### `routes/calculate.py`
**Purpose**: Emissions calculation trigger.

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `POST /api/sessions/<session_id>/calculate` | `calculate()` | Runs emissions calculations for specified sources. Returns calculation results with totals |

---

### `routes/analytics.py`
**Purpose**: Summary statistics and chart data.

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /api/sessions/<session_id>/analytics/summary` | `get_summary()` | Returns aggregated emissions by sector, year, subtype, GHG |
| `POST /api/sessions/<session_id>/analytics/chart-data` | `get_chart_data()` | Returns data formatted for charts (pie, bar, line, stacked_bar) |

| Internal Function | Description |
|-------------------|-------------|
| `apply_filters_to_df(df, filters)` | Applies filter conditions for chart data |
| `_generate_pie_chart()` | Generates pie chart data with percentages and colors |
| `_generate_bar_chart()` | Generates bar chart data |
| `_generate_line_chart()` | Generates time series line chart data |
| `_generate_stacked_bar_chart()` | Generates stacked bar chart with multiple datasets |

---

## Data Processing Pipeline

The full data processing flow:

```
1. Load Raw Data (storage.py)
        ↓
2. Apply Filters (mapping_engine.py)
        ↓
3. Map Columns to Standard Schema (mapping_engine.py)
   - Direct column mapping
   - Static values
   - Derived values (date extraction)
   - Reference table joins
   - Pattern-based columns (non-GHG patterns processed here)
   - Note: Patterns with {GHG} are DEFERRED to step 5
        ↓
4. Expand GHG Rows (mapping_engine.py)
   - 1 row becomes N rows (one per GHG)
   - Creates GHG column with values: CO2, CH4, N2O, etc.
        ↓
5. Apply Deferred Patterns (mapping_engine.py)
   - Process patterns containing {GHG} placeholder
   - EF_ID patterns like "Fuel_{Subtype}_20XX_{GHG}" applied here
        ↓
6. Aggregate Consumption (mapping_engine.py)
   - Group by dimensions, sum consumption
        ↓
7. Validate EF_ID (calculation_engine.py)
   - Ensure EF_ID column exists from user mapping
        ↓
8. Join Emission Factors (calculation_engine.py)
   - Look up GHG_MTperUnit from EFID table
        ↓
9. Join GWP Values (calculation_engine.py)
   - Look up Global Warming Potential
        ↓
10. Calculate Emissions (calculation_engine.py)
    - mtCO2e_calc = Consumption × GHG_MTperUnit × GWP
        ↓
11. Save to Parquet (storage.py)
```

---

## Error Handling

All routes use standardized error responses:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "timestamp": "2026-01-25T10:30:00Z"
  }
}
```

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `MISSING_PARAMETER` | 400 | Required parameter not provided |
| `VALIDATION_ERROR` | 400 | Configuration validation failed |
| `SOURCE_NOT_FOUND` | 404 | Source file doesn't exist |
| `SESSION_NOT_FOUND` | 404 | Session doesn't exist |
| `CONFIG_NOT_FOUND` | 404 | No config saved for source |
| `NO_DATA` | 404 | No calculated data found |
| `INVALID_PARAMETER` | 400 | Invalid request parameter |
| `MISSING_EMISSION_FACTOR` | 422 | EF_ID not found in reference |
| `MISSING_GWP` | 422 | GWP value not found for GHG |
| `CALCULATION_ERROR` | 422/500 | Error during calculation |
| `STORAGE_ERROR` | 500 | File I/O error |
