# EcoMetrics Frontend — Cloud Version

This folder (`Frontend_cloud/`) is the AWS-ready version of the React frontend.
The original local version lives in `Frontend_E2E/` and is left completely untouched.

---

## What Changed from `Frontend_E2E/`

### Modified Files

| File | What Changed |
|------|-------------|
| `src/services/api.js` | Line 2: API base URL now reads from `REACT_APP_API_URL` env var (falls back to `localhost:8000` for local dev); added `uploadFile(file, folder)` function |
| `src/Input.jsx` | Imports `uploadFile`; `reader.onload` is now `async`; calls `uploadFile()` after local file parse — sends file to backend and attaches returned `sourceId` to the file object in state |

### New Files

| File | Purpose |
|------|---------|
| `.env.production` | Sets `REACT_APP_API_URL` to the ALB URL at build time. **Must be updated before running `npm run build` for deployment.** |

### Files Identical to `Frontend_E2E/` (not changed)

`App.jsx`, `DataCleaning.jsx`, `DataMapping.jsx`, `Dashboard.jsx`, all other components, `package.json`, `tailwind.config.js`, `postcss.config.js`, `public/`

---

## How the File Upload Flow Works

**Before (local version)**: `Input.jsx` parses the file client-side (using the `xlsx` library) and stores it in React state only. The backend never received the file — it just listed files already placed in `data/raw/CONSUMPTION/` on the server.

**After (cloud version)**: `Input.jsx` still parses the file client-side for immediate local preview. Additionally, after the user confirms the upload modal, the raw file is also sent to the backend via `POST /api/files/upload`, which saves it to S3. The backend returns a `sourceId` that is attached to the file object in state — this is what downstream pages (`DataMapping`, `DataCleaning`, `Dashboard`) use when calling backend API endpoints.

```
User picks file → modal confirms name + folder tag
  → FileReader parses locally (for immediate UI preview)
  → uploadFile(file, folder) sends to backend → saved to S3
  → sourceId attached to file object in state
  → downstream pages use sourceId for API calls (previewSource, saveSourceConfig, runCalculation)
```

The **"Import from SharePoint"** button is preserved — in cloud mode it calls `listSources()` which lists files already in the S3 raw bucket. Useful for reusing previously uploaded files without re-uploading.

---

## API Functions (`src/services/api.js`)

All existing functions are unchanged. One new function was added:

### `uploadFile(file, folder)`
```javascript
// Uploads a browser File object to the backend.
// folder: "CONSUMPTION" or "REFERENCE"
// Returns: { sourceId, name, fileName, folder, sheets }
await uploadFile(selectedFile, 'CONSUMPTION')
```

The `folder` value maps to the tag the user selects in the upload modal:
- `#Consumption` tag → `"CONSUMPTION"`
- `#Reference` tag → `"REFERENCE"`

---

## Environment Variables

### `.env.production` — **Update before each cloud build**

```
REACT_APP_API_URL=http://<ALB_DNS_NAME>/api
```

Replace `<ALB_DNS_NAME>` with the actual ALB DNS name from AWS (e.g. `ecometrics-alb-123456789.us-west-2.elb.amazonaws.com`).

This value is **baked into the static build** at build time — it is not read at runtime. If the ALB URL changes, you must rebuild.

### `.env.development` (optional, for local development against local backend)

Create this file if needed:
```
REACT_APP_API_URL=http://localhost:8000/api
```

If this file doesn't exist, `api.js` falls back to `http://localhost:8000/api` automatically.

---

## Building for Production

**Step 1**: Update `.env.production` with the correct ALB DNS name.

**Step 2**: Build:
```bash
cd Frontend_cloud
npm install       # only needed once, or after dependency changes
npm run build     # outputs to build/
```

**Step 3**: Upload to S3:
```bash
aws s3 sync build/ s3://ecometrics-frontend-<your-suffix>/ --delete
```

**Step 4**: Invalidate CloudFront cache so users get the new version immediately:
```bash
aws cloudfront create-invalidation \
  --distribution-id <DISTRIBUTION_ID> \
  --paths "/*"
```

---

## What AWS Expects

| Resource | Purpose |
|----------|---------|
| S3 bucket (`ecometrics-frontend-...`) | Stores the static `build/` output |
| CloudFront distribution | Serves the React app over HTTPS with CDN |
| CloudFront Origin Access Control (OAC) | Allows CloudFront to read from the private S3 bucket |
| Custom error pages (403 → `/index.html`, 404 → `/index.html`) | Required so React Router page refreshes don't 404 |

---

## Running Locally (for development)

No setup changes needed — `api.js` falls back to `localhost:8000`:

```bash
cd Frontend_cloud
npm install
npm start        # runs on http://localhost:3000
```

Make sure `Backend_cloud` (or `Backend_local`) is also running on port 8000.
