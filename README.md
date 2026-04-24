# KZ Real Estate Price Estimator — FastAPI + Docker

Production-ready web application that estimates Kazakhstan apartment prices
from 11 user inputs (coordinates, area, floor, condition, material, year, etc.)
using a 55-feature LightGBM + Neural-Net + Ridge ensemble with per-region
calibration.

Test MAPE 9.64% on held-out data, 12.12% on March–April 2026 live listings
(56.9% within ±10%, 84.3% within ±20%).

---

## Architecture

```
┌───────── FastAPI app (main.py) ──────────────────┐
│  11 user inputs                                  │
│        │                                         │
│        ▼                                         │
│  FeaturePipeline ─── assembles 55 features       │
│        │  • city_encoder.json   (lat/lon → city) │
│        │  • region_grid         (0.01° snap)     │
│        │  • OSMDistances        (6.6M grid cells)│
│        │  • StatLoader          (per-region agg) │
│        │  • BLP / BFE lookups   (building-level) │
│        │  • price_index         (quarter trend)  │
│        ▼                                         │
│  NNInference                                     │
│    LightGBM (learners) ┐                         │
│    Neural Net   (learners) ├─ Ridge meta → exp → │
│                                 KZT/m²           │
│                                    │             │
│        region_calibration.json ─── × α → KZT/m²  │
│                                        × area    │
│                                        ─▶ total  │
└──────────────────────────────────────────────────┘
```

## Layout

```
├── main.py                    FastAPI app (/predict /batch /health /)
├── feature_pipeline.py        55-feature assembly from 11 user inputs
├── nn_inference.py            LGB + NN + Ridge ensemble loader & predict
├── osm_distances.py           O(1) grid lookup (distance_grid.parquet)
├── region_grid.py             (lat, lon) → REGION integer code
├── stat_loader.py             per-region statistical features
├── templates/index.html       UI (Yandex Maps + Leaflet)
├── static/                    CSS + JS
├── nn_model/                  14 artifacts: model.pt, lgb_model.txt, scalers,
│                              ridge_meta.joblib, feature_list, metadata,
│                              BLP / BFE lookups, price_index, cat_mappings
├── data/                      11 lookups + distance_grid.parquet (196 MB LFS)
├── scripts/
│   ├── smoke_local.py          local HTTP smoke test
│   └── webapp_local_test.py    20-point endpoint validation
├── Dockerfile                 python:3.11-slim + geospatial deps
├── docker-compose.yml         single-service build
├── requirements.txt
├── .dockerignore / .gitignore
└── .gitattributes             Git LFS rules for parquet/joblib/pt
```

## User inputs (11)

`ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, TOTAL_FLOORS,
FURNITURE, CONDITION, CEILING, MATERIAL, YEAR`

All other features (REGION, segment_code, city flags, OSM distances,
macro stats, building premiums, price-index momentum) are derived
deterministically from these 11 inputs — no user-visible mean / median /
NA fallbacks in the hot path.

## Quick start (Docker)

```bash
git lfs install                      # one-time — needed for 196 MB parquet
git clone <repo> && cd HousePricesApp_deploy
docker compose build
docker compose up -d
curl http://localhost:8000/health    # {"status":"ok","model_loaded":true}
open  http://localhost:8000/         # web UI
```

Stop / rebuild:

```bash
docker compose down
docker compose up --build -d
```

## Quick start (local, without Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path              | Description                                   |
|--------|-------------------|-----------------------------------------------|
| GET    | `/`               | HTML UI                                       |
| GET    | `/health`         | liveness + `model_loaded`                     |
| POST   | `/predict`        | JSON body → total price + KZT/m² + explanations |
| POST   | `/batch`          | multipart CSV/XLSX upload → row-wise predictions |
| GET    | `/template/csv`   | download CSV template                         |
| GET    | `/template/xlsx`  | download XLSX template                        |
| GET    | `/docs`           | Swagger UI                                    |

## Local verification

```bash
python scripts/webapp_local_test.py       # 20-assertion endpoint check
python scripts/smoke_local.py             # 5-city + CSV batch smoke
```

## Notes

* `data/distance_grid.parquet` (196 MB) is tracked via **Git LFS**.
  `git lfs install` is required before cloning.
* `YANDEX_MAPS_API_KEY` env var is optional; without it the UI falls
  back to Leaflet + OpenStreetMap tiles.
* Region calibration window `data/region_calibration.json` is dated
  Jan–Feb 2026; rebuild by re-running the training notebook.
