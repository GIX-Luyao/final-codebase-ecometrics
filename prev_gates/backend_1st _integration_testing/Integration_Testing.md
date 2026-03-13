# EcoMetrics Backend - Integration Testing Guide

This guide explains how to set up, run, and test the EcoMetrics GHG Emissions API backend.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Terminal/Command Line access

### Installation Steps

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd spu-uw-ghg-proj/backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - Flask 3.0.0 (web framework)
   - flask-cors 4.0.0 (CORS support)
   - pandas 2.1.4 (data processing)
   - numpy 1.26.3 (numerical operations)
   - python-dateutil 2.8.2 (date utilities)

---

## 2. Running the Backend

### Start the Server

From the `backend/` directory, run:

```bash
python app.py
```

You should see output similar to:
```
[2026-01-26 10:00:00] [INFO] EcoMetrics backend initialized
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://localhost:8000
```

The API is now available at `http://localhost:8000`.

### Verify the Server is Running

Test the health check endpoint:
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{"service":"ecometrics-backend","status":"healthy"}
```

### Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

---

## 3. Testing API Endpoints

### Interface 1: GET /api/raw-data

**Purpose**: Retrieve raw data files for frontend preview.

#### Test 1: Get FleetFuel data
```bash
curl "http://localhost:8000/api/raw-data?type=fleetfuel"
```

**Expected Response**:
```json
{
  "success": true,
  "dataType": "fleetfuel",
  "fileName": "FleetFuel_Raw.csv",
  "rowCount": 7,
  "columns": ["EQ_EQUIP_NO", "YEAR", "MANUFACTURER", ...],
  "columnTypes": {
    "EQ_EQUIP_NO": "integer",
    "YEAR": "integer",
    "QTY_FUEL": "float",
    ...
  },
  "preview": [
    {"EQ_EQUIP_NO": 1011, "FUEL_TYPE": "Gasoline", "QTY_FUEL": 25.5, ...},
    ...
  ]
}
```

#### Test 2: Get EFID reference data
```bash
curl "http://localhost:8000/api/raw-data?type=efid"
```

#### Test 3: Get invalid test file (for error testing)
```bash
curl "http://localhost:8000/api/raw-data?type=fleetfuel&invalid=true"
```

#### Test 4: Error case - missing type parameter
```bash
curl "http://localhost:8000/api/raw-data"
```

**Expected Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "MISSING_PARAMETER",
    "message": "Query parameter 'type' is required"
  }
}
```

---

### Interface 2: POST /api/data/update

**Purpose**: Submit and validate table edits, save to session.

#### Test 1: Rename a column (success case)
```bash
curl -X POST http://localhost:8000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "dataType": "fleetfuel",
    "changes": [
      {
        "changeType": "column_rename",
        "oldName": "FUEL_TYPE",
        "newName": "Fuel_Type"
      }
    ]
  }'
```

**Expected Response**:
```json
{
  "success": true,
  "sessionId": "sess_abc12345",
  "changesApplied": 1,
  "validation": {"status": "passed"},
  "storage": {
    "status": "committed",
    "filePath": "data/sessions/sess_abc12345/fleetfuel_modified.csv",
    "timestamp": "2026-01-26T10:30:00.000000+00:00"
  }
}
```

**Important**: Save the `sessionId` from the response - you'll need it for Interface 3.

#### Test 2: Edit a cell value
```bash
curl -X POST http://localhost:8000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "sess_abc12345",
    "dataType": "fleetfuel",
    "changes": [
      {
        "changeType": "cell_edit",
        "rowIndex": 0,
        "column": "QTY_FUEL",
        "oldValue": 25.5,
        "newValue": 30.0
      }
    ]
  }'
```

#### Test 3: Validation error - invalid column name (pure number)
```bash
curl -X POST http://localhost:8000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "dataType": "fleetfuel",
    "changes": [
      {
        "changeType": "column_rename",
        "oldName": "YEAR",
        "newName": "123"
      }
    ]
  }'
```

**Expected Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Change 0: Column name '123' is invalid: cannot be a pure number"
  }
}
```

#### Test 4: Multiple changes in one request
```bash
curl -X POST http://localhost:8000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "dataType": "fleetfuel",
    "changes": [
      {
        "changeType": "column_rename",
        "oldName": "MANUFACTURER",
        "newName": "Vehicle_Manufacturer"
      },
      {
        "changeType": "cell_edit",
        "rowIndex": 1,
        "column": "QTY_FUEL",
        "oldValue": 45.2,
        "newValue": 50.0
      }
    ]
  }'
```

---

### Interface 3: GET /api/calculated-data

**Purpose**: Get calculated emissions data in chart-ready format.

#### Test 1: Get calculated data for a session
```bash
# Replace sess_abc12345 with the actual sessionId from Interface 2
curl "http://localhost:8000/api/calculated-data?sessionId=sess_abc12345"
```

**Expected Response**:
```json
{
  "success": true,
  "sessionId": "sess_abc12345",
  "calculationTimestamp": "2026-01-26T10:35:00.000000+00:00",
  "summary": {
    "totalMtCO2e": 2.28537,
    "recordCount": 21
  },
  "charts": {
    "pieChart": {
      "title": "mtCO2e by Subtype",
      "type": "pie",
      "data": [
        {"label": "B20", "value": 0.38159, "percentage": 16.7},
        {"label": "Diesel", "value": 1.3649, "percentage": 59.7},
        {"label": "Gasoline", "value": 0.53888, "percentage": 23.6}
      ]
    },
    "barChart": {
      "title": "mtCO2e over FillDate",
      "type": "bar",
      "labels": ["2021-10-30", "2021-11-15", "2022-01-10", "2022-02-15", "2022-03-20"],
      "datasets": [
        {"label": "mtCO2e", "data": [0.46091, 0.16686, 0.54867, 0.29269, 0.81623]}
      ]
    }
  },
  "rawData": [
    {
      "ACCT_ID": 1011,
      "Consumption": 25.5,
      "Subtype": "Gasoline",
      "GHG": "CO2",
      "mtCO2e_calc": 0.226618,
      "formulaReference": {
        "formula": "mtCO2e = Consumption x GHG_MTperUnit x GWP",
        "calculation": "25.5 x 0.008887 x 1 = 0.226618",
        "efidSource": "Fuel_Gasoline_20XX_CO2"
      },
      ...
    },
    ...
  ]
}
```

#### Test 2: Error case - missing sessionId
```bash
curl "http://localhost:8000/api/calculated-data"
```

**Expected Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "MISSING_PARAMETER",
    "message": "Query parameter 'sessionId' is required"
  }
}
```

---

## 4. React Frontend Integration

### Setup CORS

The backend already has CORS enabled, so React apps running on different ports (e.g., `localhost:3000`) can make requests.

### API Base URL

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

### Example API Service (using fetch)

Create a file `src/services/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';

// Interface 1: Get raw data
export async function getRawData(type, invalid = false) {
  const params = new URLSearchParams({ type });
  if (invalid) params.append('invalid', 'true');

  const response = await fetch(`${API_BASE_URL}/api/raw-data?${params}`);
  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error.message);
  }
  return data;
}

// Interface 2: Update data
export async function updateData(dataType, changes, sessionId = null) {
  const body = {
    dataType,
    changes,
  };
  if (sessionId) {
    body.sessionId = sessionId;
  }

  const response = await fetch(`${API_BASE_URL}/api/data/update`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error.message);
  }
  return data;
}

// Interface 3: Get calculated data
export async function getCalculatedData(sessionId) {
  const response = await fetch(
    `${API_BASE_URL}/api/calculated-data?sessionId=${sessionId}`
  );
  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error.message);
  }
  return data;
}
```

### Example React Component Usage

```jsx
import React, { useState, useEffect } from 'react';
import { getRawData, updateData, getCalculatedData } from './services/api';

function DataPreview() {
  const [data, setData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [calculatedData, setCalculatedData] = useState(null);
  const [error, setError] = useState(null);

  // Load raw data on mount
  useEffect(() => {
    async function loadData() {
      try {
        const result = await getRawData('fleetfuel');
        setData(result);
      } catch (err) {
        setError(err.message);
      }
    }
    loadData();
  }, []);

  // Handle column rename
  const handleRenameColumn = async (oldName, newName) => {
    try {
      const result = await updateData('fleetfuel', [
        {
          changeType: 'column_rename',
          oldName,
          newName,
        }
      ], sessionId);

      setSessionId(result.sessionId);
      console.log('Changes saved:', result.changesApplied);
    } catch (err) {
      setError(err.message);
    }
  };

  // Handle cell edit
  const handleCellEdit = async (rowIndex, column, oldValue, newValue) => {
    try {
      const result = await updateData('fleetfuel', [
        {
          changeType: 'cell_edit',
          rowIndex,
          column,
          oldValue,
          newValue,
        }
      ], sessionId);

      setSessionId(result.sessionId);
    } catch (err) {
      setError(err.message);
    }
  };

  // Calculate emissions
  const handleCalculate = async () => {
    if (!sessionId) {
      setError('No session. Make some edits first.');
      return;
    }

    try {
      const result = await getCalculatedData(sessionId);
      setCalculatedData(result);
    } catch (err) {
      setError(err.message);
    }
  };

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div>Loading...</div>;

  return (
    <div>
      <h2>{data.fileName} ({data.rowCount} rows)</h2>

      {/* Render data table */}
      <table>
        <thead>
          <tr>
            {data.columns.map(col => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.preview.map((row, i) => (
            <tr key={i}>
              {data.columns.map(col => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <button onClick={handleCalculate}>Calculate Emissions</button>

      {/* Render charts if calculated */}
      {calculatedData && (
        <div>
          <h3>Total Emissions: {calculatedData.summary.totalMtCO2e} mtCO2e</h3>
          {/* Use chart library (Chart.js, Recharts, etc.) to render charts */}
        </div>
      )}
    </div>
  );
}

export default DataPreview;
```

### Using Axios (Alternative)

If you prefer Axios over fetch:

```bash
npm install axios
```

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function getRawData(type, invalid = false) {
  const params = { type };
  if (invalid) params.invalid = 'true';

  const { data } = await api.get('/api/raw-data', { params });
  if (!data.success) throw new Error(data.error.message);
  return data;
}

export async function updateData(dataType, changes, sessionId = null) {
  const { data } = await api.post('/api/data/update', {
    dataType,
    changes,
    ...(sessionId && { sessionId }),
  });
  if (!data.success) throw new Error(data.error.message);
  return data;
}

export async function getCalculatedData(sessionId) {
  const { data } = await api.get('/api/calculated-data', {
    params: { sessionId },
  });
  if (!data.success) throw new Error(data.error.message);
  return data;
}
```

### Environment Variables

For production, use environment variables for the API URL:

Create `.env` in your React project root:
```
REACT_APP_API_URL=http://localhost:8000
```

Then in your code:
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL;
```

---

## 5. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Connection refused` | Ensure the backend server is running (`python app.py`) |
| `CORS error` | Backend has CORS enabled; check if server is running on port 8000 |
| `Module not found` | Run `pip install -r requirements.txt` |
| `File not found` error | Ensure you're running from the `backend/` directory |
| `Port already in use` | Kill existing process: `lsof -ti:8000 | xargs kill -9` |

### Check Logs

The backend logs all operations to both console and `logs/backend.log`:
```bash
cat logs/backend.log
```

### Verify Session Files

After making edits, check session files were created:
```bash
ls -la data/sessions/
cat data/sessions/<session_id>/changes.json
```
