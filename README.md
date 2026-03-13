# EcoMetrics GHG Emissions Data Engine

A greenhouse gas (GHG) emissions calculation platform built for Seattle Public Utilities (SPU). Users upload raw consumption data (fuel, electricity, refrigerants, etc.), declaratively map columns to a standard schema, and receive CO₂-equivalent emissions calculations with interactive charts.

---

## Repository Structure

```
spu-uw-ghg-proj/
├── backend_cloud/          # Flask API — production backend (runs in Docker / AWS ECS)
│   ├── app.py              # App factory and entry point
│   ├── config.py           # Configuration (local vs. S3 storage mode)
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Container image definition
│   ├── routes/             # API route handlers
│   │   ├── sources.py      # GET /api/sources, /api/sources/{id}/preview
│   │   ├── sessions.py     # POST/GET/DELETE /api/sessions
│   │   ├── config.py       # POST/GET /api/sessions/{id}/sources/{id}/config
│   │   ├── calculate.py    # POST /api/sessions/{id}/calculate
│   │   └── analytics.py    # GET /api/sessions/{id}/analytics/*
│   ├── services/           # Business logic
│   │   ├── mapping_engine.py       # Column mapping & GHG row expansion
│   │   ├── calculation_engine.py   # Emission factor lookup & mtCO2e calc
│   │   ├── validation.py           # Config and data validation
│   │   ├── storage.py              # Local file I/O
│   │   └── s3_storage.py          # AWS S3 file I/O
│   ├── data/
│   │   ├── raw/CONSUMPTION/        # Source data files (xlsx)
│   │   └── raw/REFERENCE/          # Reference tables (read-only)
│   ├── tests/              # pytest test suite
│   └── logs/               # Application logs
├── frontend_final/         # React frontend (production UI)
│   ├── src/
│   │   ├── App.jsx                 # Root — page routing via currentPage state
│   │   ├── Input.jsx               # Step 1: Upload / browse source files
│   │   ├── DataCleaning.jsx        # Step 2: Preview & clean raw data
│   │   ├── DataMapping.jsx         # Step 3: Configure column mappings
│   │   ├── Dashboard.jsx           # Step 4: View results & charts
│   │   └── services/api.js         # All API calls (fetch wrappers)
│   ├── package.json
│   └── tailwind.config.js
├── docs/                   # Architecture and API documentation
│   ├── FRONTEND_API_GUIDE.md
│   ├── BACKEND_MODULES.md
│   └── data_source_schemas.md
└──  prev_gates/             # Previous milestone deliverables (reference only)
```

---

## How It Works

The system follows pipeline for each data source:

1. **Load** raw XLSX/CSV from storage (local disk or S3)
2. **Filter** rows by user-defined conditions (e.g., `QTY_FUEL > 0`)
3. **Map columns** to a standard schema using declarative rules:
   - `sourceColumn` — copy from a raw column
   - `staticValue` — assign a constant (e.g., `"Unit": "gal"`)
   - `derivedFrom` — extract year/month/day from a date column
   - `referenceTable` — look up a value from a reference file via a join key
   - `ghgType` — expand rows for multiple gases (CO2, CH4, N2O)
4. **Calculate** `mtCO2e = Consumption × GHG_MTperUnit × GWP`
5. **Analytics / Visualization**  show calculated data results and charts

---

## Environment Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- Docker (for cloud deployment)
- AWS CLI + SSO configured (for cloud deployment)

### Backend (local mode)

```bash
cd backend_cloud

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in local storage mode
STORAGE_BACKEND=local python app.py
```

The API will be available at `http://localhost:8000`.

**Environment variables (all optional):**

| Variable | Default | Description |
|---|---|---|
| `STORAGE_BACKEND` | `s3` | `local` for filesystem, `s3` for AWS |
| `FLASK_PORT` | `8000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `S3_RAW_BUCKET` | `spu-emissions-raw-data` | S3 bucket for source files |
| `S3_DATA_BUCKET` | `spu-emissions-processed-data` | S3 bucket for results |
| `AWS_DEFAULT_REGION` | `us-east-2` | AWS region |

### Frontend

```bash
cd frontend_final

npm install
npm start
```

The app will be available at `http://localhost:3000`. By default it points to the production API at `https://ecometrics-api.adc.seattle.gov/api`. To use a local backend instead, create `frontend_final/.env.development.local`:

```
REACT_APP_API_URL=http://localhost:8000/api
```

---

## Running Tests

```bash
cd backend_cloud
source venv/bin/activate

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

---

## Cloud Deployment (AWS ECS)

The backend runs as a Docker container on AWS ECS (Fargate) behind an Application Load Balancer. Data is stored in S3.

### Build and push Docker image

First, authenticate to ECR:

```bash
aws sso login
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  {proj_id}.dkr.ecr.us-east-2.amazonaws.com
```

Then build and push:

```bash
docker build --platform linux/amd64 -t ecometrics-backend-cloud backend_cloud/

docker tag ecometrics-backend-cloud:latest \
  {proj_id}.dkr.ecr.us-east-2.amazonaws.com/spu-emissions-backend:latest

docker push \
  {proj_id}.dkr.ecr.us-east-2.amazonaws.com/spu-emissions-backend:latest
```

> `--platform linux/amd64` is required — ECS tasks run on x86 hardware.

After pushing, redeploy the ECS service via the AWS Console or CLI to pick up the new image.

### Frontend (AWS Amplify)

The `frontend_final/` directory is deployed via AWS Amplify. The production API URL is baked into the build via `REACT_APP_API_URL=https://ecometrics-api.adc.seattle.gov/api`.

---

## What to Expect After Running

### Backend (`/api/health`)

```json
{
  "service": "ecometrics-backend",
  "status": "healthy",
  "version": "1.0.0"
}
```

### Frontend Workflow

**Step 1 — Upload:** Browse available source files (FleetFuel, PSE electricity, AC, etc.). Select a file and preview its columns and sample rows.

**Step 2 — Data Cleaning:** Review the raw data, inspect null counts, and flag rows to exclude before processing.

**Step 3 — Data Mapping:** Open the configuration modal for each data source. Map raw columns to the standard schema using the five mapping types. Set filters (e.g., exclude zero-fuel rows). Configure the EF_ID pattern for emission factor lookup.

**Step 4 — Dashboard:** After running the calculation, view:
- Total mtCO2e by year, sector, fuel type, and line of business
- Pie, bar, and stacked-bar charts (powered by Recharts)
- Drill-down into individual GHG contributions (CO2, CH4, N2O)

### Expected calculation output (standard schema)

| Column | Example |
|---|---|
| ACCT_ID | 1011 |
| Year | 2022 |
| Consumption | 25.5 |
| Unit | gal |
| Subtype | Gasoline |
| GHG | CO2 |
| EF_ID | Fuel_Gasoline_20XX_CO2 |
| GHG_MTperUnit | 0.008887 |
| GWP | 1 |
| mtCO2e_calc | 0.2267 |

---

## Key API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/sources` | List available source files |
| GET | `/api/sources/{id}/preview` | Preview columns and sample rows |
| POST | `/api/sessions` | Create a new analysis session |
| GET | `/api/sessions` | List all sessions |
| POST | `/api/sessions/{id}/sources/{srcId}/config` | Save column mapping config |
| GET | `/api/sessions/{id}/sources/{srcId}/config` | Retrieve saved config |
| POST | `/api/sessions/{id}/calculate` | Run emissions calculation |
| GET | `/api/sessions/{id}/analytics/summary` | Aggregated emissions summary |
| POST | `/api/sessions/{id}/analytics/chart-data` | Chart-ready data |
| DELETE | `/api/sessions/{id}` | Delete session and results |

Full request/response examples are in [`docs/FRONTEND_API_GUIDE.md`](docs/FRONTEND_API_GUIDE.md).

---

## Reference

- Backend module documentation: [`docs/BACKEND_MODULES.md`](docs/BACKEND_MODULES.md)
- Data source schemas: [`docs/data_source_schemas.md`](docs/data_source_schemas.md)

