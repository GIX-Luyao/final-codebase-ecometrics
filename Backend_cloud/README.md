# EcoMetrics Backend — Cloud Version

This folder (`Backend_cloud/`) is the AWS-ready version of the Flask backend.
The original local version lives in `Backend_local/` and is left completely untouched.

---

## What Changed from `Backend_local/`

### Modified Files

| File | What Changed |
|------|-------------|
| `config.py` | Added S3 env vars (`STORAGE_BACKEND`, `S3_RAW_BUCKET`, `S3_DATA_BUCKET`, `AWS_DEFAULT_REGION`), `is_s3_mode()` helper, `ensure_directories()` is a no-op in S3 mode |
| `services/storage.py` | Added a transparent proxy shim at the bottom — routes all `StorageService` calls to `S3StorageService` when `STORAGE_BACKEND=s3`, otherwise falls back to the original local implementation |
| `routes/sources.py` | Added `POST /api/files/upload` endpoint (new); fixed `preview_source` to handle S3 string keys (not just `Path` objects) |
| `requirements.txt` | Added `boto3==1.34.0` (S3 access) and `gunicorn==21.2.0` (production server) |

### New Files

| File | Purpose |
|------|---------|
| `services/s3_storage.py` | Complete S3-backed twin of `StorageService`. Same method signatures — no other code needs to change. Handles listing/loading files from S3, reading/writing session configs and calculated results to S3. |
| `Dockerfile` | Builds the container image. Uses `python:3.11-slim`, installs deps, runs gunicorn on port 8000. |
| `.dockerignore` | Excludes `data/`, `logs/`, `__pycache__`, and `tests/` from the Docker image. |

### Files Identical to `Backend_local/` (not changed)

`app.py`, `routes/__init__.py`, `routes/sessions.py`, `routes/config.py`, `routes/calculate.py`, `routes/analytics.py`, `services/mapping_engine.py`, `services/calculation_engine.py`, `services/validation.py`

---

## How the Storage Switch Works

`services/storage.py` ends with a proxy class:

```python
class _StorageProxy:
    def __getattr__(self, name):
        from config import Config
        if Config.is_s3_mode():
            from services.s3_storage import S3StorageService as _backend
        else:
            _backend = _LocalStorageService
        return getattr(_backend, name)

StorageService = _StorageProxy()
```

Any code that does `from services.storage import StorageService` automatically gets the S3 version when `STORAGE_BACKEND=s3`. No routes or engines needed changing.

---

## Storage Architecture

```
S3 Raw Bucket  (ecometrics-raw-...)
├── CONSUMPTION/    ← user-uploaded consumption data files (.xlsx, .csv)
└── REFERENCE/      ← user-uploaded reference files (EFID.xlsx, GWPs.xlsx, LOB_LowOrgList.xlsx)

S3 Processed Bucket  (ecometrics-processed-...)
├── sessions/
│   ├── registry.json
│   └── {sessionId}/
│       ├── session_meta.json
│       └── sources/{sourceId}/config.json
└── calculated/
    └── {sessionId}/
        ├── {sourceId}_emissions.csv
        └── calculation_log.json
```

**Important**: Reference files (`EFID.xlsx`, `GWPs.xlsx`, `LOB_LowOrgList.xlsx`) are no longer bundled in the container. Users must upload them to the S3 raw bucket under the `REFERENCE/` folder before running calculations.

---

## New API Endpoint

### `POST /api/files/upload`

Uploads a file to S3 (cloud) or the local `CONSUMPTION/`/`REFERENCE/` directory (local mode).

**Request**: `multipart/form-data`
- `file` — the file bytes
- `folder` — `"CONSUMPTION"` or `"REFERENCE"`

**Response**:
```json
{
  "success": true,
  "sourceId": "fleet_fuel_2019_2022",
  "name": "FleetFuel_2019-2022",
  "fileName": "FleetFuel_2019-2022.xlsx",
  "folder": "CONSUMPTION",
  "sheets": ["Data", "Summary"]
}
```

---

## Environment Variables (set in ECS Task Definition)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `STORAGE_BACKEND` | Yes | Storage mode | `s3` |
| `S3_RAW_BUCKET` | Yes | Bucket for uploaded files | `ecometrics-raw-yw` |
| `S3_DATA_BUCKET` | Yes | Bucket for sessions + results | `ecometrics-processed-yw` |
| `AWS_DEFAULT_REGION` | Yes | AWS region | `us-west-2` |
| `CORS_ORIGINS` | Yes | Allowed frontend origin | `https://abc123.cloudfront.net` |
| `FLASK_DEBUG` | No | Debug mode | `false` |
| `FLASK_PORT` | No | Port (default 8000) | `8000` |

**boto3 credentials**: Do NOT set `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY`. When running on ECS Fargate, the container automatically inherits credentials from the **ECS Task IAM Role**. boto3 picks these up automatically.

---

## Running Locally (for development/testing)

You can still run this cloud version locally — it will use local file storage by default since `STORAGE_BACKEND` defaults to `s3` in this version, so override it:

```bash
cd Backend_cloud
pip install -r requirements.txt
STORAGE_BACKEND=local python app.py
```

Or to test S3 mode locally with real AWS credentials:
```bash
export STORAGE_BACKEND=s3
export S3_RAW_BUCKET=ecometrics-raw-yw
export S3_DATA_BUCKET=ecometrics-processed-yw
python app.py
```

---

## Building and Pushing the Docker Image

```bash
# From the Backend_cloud/ directory:
docker build -t ecometrics-backend .

# Tag and push to ECR (replace <ACCOUNT_ID> with your AWS account ID):
docker tag ecometrics-backend:latest \
  <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/ecometrics-backend:latest

docker push \
  <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/ecometrics-backend:latest
```

After pushing, force a new deployment in ECS:
ECS → Clusters → `ecometrics-cluster` → Services → `ecometrics-backend-service` → **"Update service"** → check **"Force new deployment"** → Update.

---

## IAM Permissions Required

The ECS Task Role (`ecometrics-ecs-task-role`) needs at minimum:

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::ecometrics-raw-<suffix>",
    "arn:aws:s3:::ecometrics-raw-<suffix>/*",
    "arn:aws:s3:::ecometrics-processed-<suffix>",
    "arn:aws:s3:::ecometrics-processed-<suffix>/*"
  ]
}
```

For initial setup, `AmazonS3FullAccess` is acceptable; restrict to specific buckets once everything is working.
