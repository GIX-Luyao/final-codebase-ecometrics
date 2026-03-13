# Frontend Cloud Integration Guide

This document explains what has changed and what you need to do when working on the frontend to connect to the cloud (AWS) backend.

---

## 1. Environment Setup

The API base URL is controlled by an environment variable. You must set it before running the app.

**For local development** — create or edit `Frontend_cloud/.env.development.local`:
```
REACT_APP_API_URL=https://ecometrics-api.adc.seattle.gov/api
```

**For production builds** — edit `Frontend_cloud/.env.production`:
```
REACT_APP_API_URL=https://ecometrics-api.adc.seattle.gov/api
```

If `REACT_APP_API_URL` is not set, the app falls back to `http://localhost:8000/api` (local backend).

---

## 2. Starting the App

```bash
cd Frontend_cloud
npm install     # only needed first time or after dependency changes
npm start       # starts on http://localhost:3000
```

---

## 3. How Files Work in Cloud Mode

In the cloud deployment, **all files are stored in AWS S3**, not on the server's local disk.

### Two ways to get files into the system:

| Method | How | When to use |
|--------|-----|-------------|
| **Upload from local** | User picks a file → `POST /api/files/upload` → stored in S3 | User has a new file to process |
| **Browse existing** | `GET /api/sources` → lists files already in S3 | Files already uploaded or synced to S3 |

### Key difference from local mode:
- `GET /api/sources` returns `filePath` as an **S3 URI**: `s3://spu-emissions-raw-data/CONSUMPTION/file.xlsx`
- In local mode it returned a local path like `data/raw/CONSUMPTION/file.xlsx`
- Frontend code should not rely on `filePath` format — use `sourceId` for all API calls

---

## 4. File Upload Flow

The upload function is already implemented in `src/services/api.js`:

```js
import { uploadFile } from './services/api';

// file: browser File object (from <input type="file">)
// folder: "CONSUMPTION" or "REFERENCE"
const result = await uploadFile(file, folder);

// result contains:
// {
//   sourceId: "pse_data_2022",      ← use this for all subsequent API calls
//   name: "PSE_Data_2022",
//   fileName: "PSE_Data_2022.xlsx",
//   folder: "CONSUMPTION",
//   sheets: ["Sheet1", "Data"]       ← null for CSV files
// }
```

**Important:** Always handle upload errors explicitly. The current implementation in `Input.jsx` catches upload errors silently. If you build new upload UIs, show the user an error message on failure:

```js
try {
  const result = await uploadFile(file, folder);
  // store result.sourceId for later use
} catch (err) {
  // Show error to user — file was NOT saved to S3
  console.error('Upload failed:', err.message);
}
```

---

## 5. Complete API Function Reference

All API functions are in `src/services/api.js`. Import what you need:

```js
import {
  healthCheck,
  uploadFile,
  listSources,
  previewSource,
  createSession,
  listSessions,
  deleteSession,
  saveSourceConfig,
  getSourceConfig,
  runCalculation,
  getAnalyticsSummary,
  getChartData,
} from './services/api';
```

### Function signatures:

| Function | Endpoint | Description |
|----------|----------|-------------|
| `healthCheck()` | `GET /api/health` | Check if backend is up |
| `uploadFile(file, folder)` | `POST /api/files/upload` | Upload file to S3 |
| `listSources()` | `GET /api/sources` | List all files in S3 |
| `previewSource(sourceId, sheet?)` | `GET /api/sources/{id}/preview` | Get columns + sample rows |
| `createSession(name, description?)` | `POST /api/sessions` | Create analysis session |
| `listSessions()` | `GET /api/sessions` | List all sessions |
| `deleteSession(sessionId)` | `DELETE /api/sessions/{id}` | Delete session |
| `saveSourceConfig(sessionId, sourceId, config)` | `POST /api/sessions/{id}/sources/{id}/config` | Save column mapping config |
| `getSourceConfig(sessionId, sourceId)` | `GET /api/sessions/{id}/sources/{id}/config` | Load saved config |
| `runCalculation(sessionId, sources?, gwpVersion?)` | `POST /api/sessions/{id}/calculate` | Run emissions calculation |
| `getAnalyticsSummary(sessionId)` | `GET /api/sessions/{id}/analytics/summary` | Get totals by year/sector/GHG |
| `getChartData(sessionId, chartType, metric, groupBy, filters?)` | `POST /api/sessions/{id}/analytics/chart-data` | Get chart-ready data |

---

## 6. Typical User Flow

```
1. User uploads a file
   └─ uploadFile(file, "CONSUMPTION")
      └─ returns sourceId (e.g. "fleetfuel_2019_2022")

2. User previews the file's columns
   └─ previewSource(sourceId, sheetName)

3. Session is created (lazily, on first need)
   └─ createSession("My Analysis")
      └─ returns sessionId (e.g. "sess_abc123")

4. User configures column mappings
   └─ saveSourceConfig(sessionId, sourceId, { columnMappings, filters, ... })

5. User runs calculation
   └─ runCalculation(sessionId, [sourceId])

6. User views results
   └─ getAnalyticsSummary(sessionId)
   └─ getChartData(sessionId, "bar", "mtCO2e_calc", "Year")
```

---

## 7. Error Handling Pattern

All API functions throw an `Error` with a descriptive message if `success: false`. Wrap calls in try/catch:

```js
try {
  const summary = await getAnalyticsSummary(sessionId);
  // use summary.summary.totalMtCO2e, etc.
} catch (err) {
  // err.message comes directly from the backend error response
  setErrorMessage(err.message);
}
```

Common error codes from the backend:

| Code | Meaning |
|------|---------|
| `SOURCE_NOT_FOUND` | sourceId doesn't exist in S3 |
| `SESSION_NOT_FOUND` | sessionId doesn't exist |
| `VALIDATION_ERROR` | Column mapping config is invalid |
| `MISSING_EMISSION_FACTOR` | EF_ID not found in reference table |
| `STORAGE_ERROR` | S3 read/write failed |

---

## 8. Reference

- Full API documentation: `Backend_cloud/docs/FRONTEND_API_GUIDE.md`
- Column mapping rules and examples: see the guide above
- Backend source: `Backend_cloud/`
- API client: `Frontend_cloud/src/services/api.js`
