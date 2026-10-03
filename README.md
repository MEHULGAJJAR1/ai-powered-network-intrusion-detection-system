# AI-Powered Network Intrusion Detection System (NIDS)

A production-oriented, modular machine-learning application for classifying KDD Cup 1999 connection records into **Normal, DoS, Probe, R2L, and U2R**. It includes a Flask REST API, responsive SOC-style dashboard, reproducible training/evaluation scripts, a model registry, CSV inference, SQLite prediction history, and SHAP-capable explanations.

> **No trained weights or performance values are bundled.** The repository does not contain the KDD dataset. Add the authentic dataset and run training before inference is enabled. Dashboard metrics are loaded from the generated evaluation artifacts; the application does not invent demo scores. Generated records are for UI/demo walkthroughs only and are never used by `train.py`.

## What is implemented

- Canonical adapter for the KDD Cup 1999 41-feature schema, original attack labels, headered files, gzip files, and the common trailing difficulty-column variant.
- Explicit label grouping into five classes; unknown labels fail closed rather than being silently dropped or relabeled.
- Exact-duplicate detection/removal before splitting, deterministic stratified sampling, group-disjoint train/validation/test partitions by feature fingerprint, and group-aware stratified CV.
- Leakage-safe sklearn preprocessing inside each CV fold: numeric quantile clipping, median imputation, robust scaling, categorical imputation and one-hot encoding with unknown-category tolerance.
- Logistic Regression baseline, class-weighted Random Forest, XGBoost with balanced sample weights, and optional LightGBM when installed.
- `RandomizedSearchCV` optimization using macro-F1; production selection is by validation macro-F1, not test performance or accuracy.
- Held-out accuracy, macro precision/recall/F1, weighted F1, multiclass ROC-AUC OVR when computable, per-class support/recall/FPR, confusion matrices, search parameters and split metadata.
- Versioned model files plus the active joblib pipeline, separately persisted preprocessor, JSON registry, evaluation metrics and a training-only explainability background sample.
- Individual and JSON-batch REST inference, schema validation, CSV upload/download, confidence/probabilities, a transparent risk heuristic, prediction/alert history, CSV history export and health checks.
- Global model feature importance and per-record SHAP `TreeExplainer` attribution for supported tree models. If SHAP cannot run in the current runtime, the UI labels the global-importance fallback and does not claim it is local SHAP.
- Responsive dark/light dashboard with overview, synthetic simulation, individual prediction, CSV batch, history, alert filters, performance, real-dataset exploration, explainability and architecture pages.
- Pydantic request contracts, upload extension/size/row/schema checks, CSV formula-prefix mitigation, CORS allowlist, rate limiting, structured logs, secure error responses and security headers.
- Docker support and pytest coverage for data adaptation, preprocessing, group splitting, loading, validation and API behavior.

## Architecture

```text
KDD Cup 1999 CSV/GZIP
        │
        ▼
Data adapter ── schema/label checks, duplicate detection, KDD attack-family mapping
        │
        ▼
Preprocessing pipeline ── train-fitted clipping/imputation/scaling/one-hot encoding
        │
        ▼
Model training ── Logistic Regression / Random Forest / XGBoost / optional LightGBM
        │                class weighting · group-aware randomized stratified CV
        ▼
Evaluation ── validation macro-F1 selection · held-out test report
        │
        ▼
Versioned model registry ── joblib pipeline + JSON metadata/metrics + SHAP background
        │
        ▼
Flask REST API ── Pydantic validation · rate limits · prediction/alert history
        │
        ▼
Responsive SOC dashboard ── inference · CSV · performance · data exploration · explanations
```

See the in-app **Architecture** section and [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for design and methodology. API docs are available at `/api/docs`; the machine-readable OpenAPI document is at `/api/openapi.json`.

## Dataset

The default source is the **KDD Cup 1999 10% training subset**, available from the [UCI KDD Cup 1999 archive](https://kdd.ics.uci.edu/databases/kddcup99/). It contains 41 connection features and an original attack label. The repository does not redistribute the dataset. The downloader tries UCI first and, if that file endpoint blocks automated access, falls back to a **pinned public GitHub mirror** of the same named KDD file; it prints the actual source and computed SHA-256. For strict provenance, pass an approved `--url`, verify the checksum against your trusted source, or place an authorized copy manually. The training-time digest is recorded in the model registry.

Download to the default location:

```bash
python scripts/download_kdd.py
# or place the authentic file at data/raw/kddcup.data_10_percent.gz
```

You may also pass `--data /path/to/file` or set `NIDS_DATA_PATH`. Do not use `scripts/generate_demo_data.py` output for model training. That utility creates **synthetic unlabeled UI/demo records** only.

### Class mapping

| Dashboard class | Example KDD attack names (the adapter includes the reviewed mapping) |
|---|---|
| Normal | `normal` |
| DoS | `back`, `land`, `neptune`, `pod`, `smurf`, `teardrop`, `apache2`, `mailbomb`, `udpstorm`, `processtable`, `worm` |
| Probe | `ipsweep`, `nmap`, `portsweep`, `satan`, `mscan`, `saint` |
| R2L | `ftp_write`, `guess_passwd`, `imap`, `multihop`, `phf`, `spy`, `warezclient`, `warezmaster`, `named`, `sendmail`, `snmpguess`, `snmpgetattack`, `httptunnel`, `xlock`, `xsnoop` |
| U2R | `buffer_overflow`, `loadmodule`, `perl`, `rootkit`, `xterm`, `ps`, `sqlattack` |

Attack taxonomy variants are documented in `nids/schema.py`. Unknown labels raise a descriptive error and must be reviewed before adding a mapping. KDD is an old benchmark and is not a substitute for current, environment-specific network telemetry or a modern benchmark.

## Data preparation and leakage controls

1. `nids/data/adapter.py` reads the 42-column legacy file, detects optional headers and the common trailing `difficulty` field, normalizes category strings, converts numerical columns, maps attack labels and reports duplicates/coercions.
2. Exact duplicate feature+label rows are removed before model splitting. Dataset exploration can show duplicate counts without deduplicating. Feature-identical rows with conflicting labels remain detectable as a single feature group.
3. When requested, a deterministic **stratified row cap** is applied after adaptation. The split then groups identical feature fingerprints together to keep them from crossing partitions.
4. Nested `StratifiedGroupKFold` makes approximately 64% search-train, 16% validation and 20% held-out test partitions. The code rejects a split when any class cannot be represented safely; use more data rather than weakening the group boundary.
5. The preprocessing pipeline is inside each CV estimator, so quantile limits, imputers, scaler and one-hot categories are fitted only on the fold's training rows. The same persisted fitted transformer runs at inference.
6. KDD is highly imbalanced. Logistic Regression and Random Forest use class weights; XGBoost receives balanced per-row sample weights. No SMOTE/resampling is applied to validation/test data.
7. Search uses macro-F1. The validation macro-F1 selects the production candidate; after selection, each tuned candidate is refit on train+validation and evaluated on the untouched test partition. Test metrics are reported for comparison, not used to select the production winner.

## Installation and quick start

Requires Python **3.10–3.13** (Python 3.11 is the reference/Docker runtime), a working compatible wheel environment, and enough memory for the chosen dataset/model cap. XGBoost is in the standard requirements. LightGBM is optional and automatically included if installed.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env                 # optional for local defaults; edit before deployment
python scripts/download_kdd.py
python train.py --data data/raw/kddcup.data_10_percent.gz \
  --output artifacts --max-rows 100000 --cv-folds 3 --n-iter 2
python app.py
```

Open <http://localhost:5000>. For larger experiments, increase `--max-rows` or set it to `0` to use the complete adapted file, and increase `--n-iter`/`--cv-folds` for a broader search. Training multiple tree candidates on the full dataset can take substantial time and memory; tune resource limits for your host.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\download_kdd.py
python train.py --data data\raw\kddcup.data_10_percent.gz --output artifacts --max-rows 100000 --cv-folds 3 --n-iter 2
python app.py
```

### Docker

The web service can start without a model (health is `degraded`, predictions return HTTP 503). To build and start after training:

```bash
# Download the real dataset on the host and train into the mounted artifacts directory.
python scripts/download_kdd.py
docker compose build
docker compose run --rm nids python train.py \
  --data /app/data/raw/kddcup.data_10_percent.gz --output /app/artifacts \
  --max-rows 100000 --cv-folds 3 --n-iter 2
docker compose up -d nids
```

Open <http://localhost:5000>. `docker compose up --build -d nids` is the one-command runtime start when artifacts have already been trained. `docker compose run --rm nids python train.py ...` runs training in the same image. Data and artifacts are bind-mounted so they persist on the host. The container runs as a non-root numeric UID/GID (defaults 1000); set `NIDS_UID`/`NIDS_GID` in `.env` to match writable host directories on Linux. Change `.env`/Compose environment values before any exposed deployment; the sample secret is not production-safe.

## Train, evaluate and inspect artifacts

```bash
python train.py --help
python evaluate.py --data data/raw/kddcup.data_10_percent.gz --artifacts artifacts
python scripts/write_report_results.py --artifacts artifacts
```

`train.py` writes, atomically where supported:

```text
artifacts/
├── production_model.joblib                 # active end-to-end fitted pipeline
├── preprocessor.joblib                     # fitted preprocessor for separate inspection
├── explainability_background.joblib        # deterministic training-only sample
├── metrics.json                            # active measured CV/validation/test metrics
├── registry.json                           # active version, hash, features, packages, metrics
├── metrics_versions/<version>.json         # archived run metrics
├── registry_versions/<version>.json        # archived run registry
├── models/<name>-<version>.joblib           # versioned model candidates
└── evaluation/reevaluated_metrics.json      # optional evaluate.py output
```

The active service automatically loads the production pipeline and registry at process startup. Restart/redeploy the service after training a new version. Keep artifacts private if they contain operational or sensitive data; joblib files must only be loaded from trusted sources.

## Dashboard pages

- **Overview:** prediction-history KPIs, class/risk distribution, recent events and measured model quality.
- **Live simulation:** continuously classifies clearly marked synthetic examples using the actual loaded model. It is not a live packet feed.
- **Analyze traffic:** single-record form generated from the API schema; strict 41-feature validation; prediction probabilities and local explanation when available.
- **CSV batch scan:** secure CSV validation, in-pipeline batch prediction and downloadable output with class probabilities, policy risk and global feature context.
- **Prediction history / alerts:** search, pagination, sorting, class/severity/confidence/date filters and CSV export. History is local SQLite and bounded by `NIDS_MAX_HISTORY_ROWS`.
- **Model performance:** all candidate metrics, per-class recall and false-positive rate, confusion matrix, ROC-AUC when defined and split metadata.
- **Data exploration:** real source-file dimensions, class distribution, exact duplicates, numerical statistics, categorical distributions and correlation matrix. No KDD file means an explicit unavailable state.
- **Explainability:** global SHAP summary for supported tree models when available; otherwise model feature importance. Individual prediction view distinguishes local SHAP from global fallback.
- **Architecture:** data-to-dashboard flow and operating boundary.

### Screenshots

Screenshots are intentionally not committed: no dataset/model run is bundled, and screenshots would otherwise imply measured results that have not been generated. The dashboard pages above are the screenshot targets after you train with your own authentic dataset; capture Overview, Model Performance, Data Exploration, Explainability and Batch Results from the running app.

## REST API

All record inference requires all 41 canonical feature keys. Keys may be in any order; the service reorders them to the stored schema. Missing feature values may be `null`/blank and are handled by the fitted imputer. Unknown categorical values are accepted by the one-hot encoder's `handle_unknown=ignore`; unknown fields, negative values, non-finite numeric values and out-of-range rate fields are rejected.

| Method / path | Description |
|---|---|
| `GET /api/health` | Process/model/dataset availability. Healthy app start is possible without model; predictions then return 503. |
| `GET /api/schema` | Feature definitions and class order. |
| `POST /api/predict` | `{"record": { ...41 features... }}` → class, confidence, per-class probabilities, policy severity and explanation. |
| `POST /api/predict-batch` | `{"records": [{...}, ...], "include_local_explanations": false}`. JSON batch cap is configurable. |
| `POST /api/upload-csv` | Multipart field `file`, `.csv` only; validated schema; returns prediction CSV attachment. |
| `GET /api/model-info` | Model/version/dataset/feature registry. |
| `GET /api/metrics` | Actual evaluation JSON or `available:false` before training. |
| `GET /api/feature-importance?limit=25` | Global SHAP/feature-importance summary. |
| `GET /api/data-summary` | Actual configured dataset summary or `available:false`. |
| `GET /api/stats` | Aggregates calculated from retained inference history. |
| `GET /api/history` | Filtered/paginated inference history. Supports attack type, severity, confidence, date/time, search and sort parameters. |
| `GET /api/alerts` | Same filters, limited to non-Normal predictions. |
| `GET /api/history/export.csv?alerts_only=true` | Download stored history or alert history. |
| `GET /api/demo-record` | Synthetic unlabeled row for a form walkthrough. |
| `POST /api/demo/predict` | Classify one generated synthetic row with the active model; response is marked synthetic. |
| `GET /api/openapi.json`, `GET /api/docs` | OpenAPI 3.0 contract and browser-readable endpoint reference. |

### Example single-record prediction

A request must contain all 41 feature keys. The following command builds a schema-complete **synthetic demo** record and sends it to the active model; the record has no ground-truth label and is not training data:

```bash
python scripts/generate_demo_data.py --rows 1 --output /tmp/synthetic_traffic.csv
python - <<'PY' > /tmp/nids_payload.json
import csv, json
with open('/tmp/synthetic_traffic.csv', newline='') as stream:
    print(json.dumps({"record": next(csv.DictReader(stream))}))
PY
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/nids_payload.json
```

The browser's **Fill synthetic example** button creates the same schema-complete demo form. These generated features are for UI/API testing after a real model exists; the actual prediction is produced by the loaded model, with no simulated class label.

## Configuration

Copy `.env.example` to `.env`. Important settings:

| Variable | Default | Purpose |
|---|---|---|
| `NIDS_DATA_PATH` | `data/raw/kddcup.data_10_percent.gz` | Dataset used for training/exploration. |
| `NIDS_ARTIFACT_DIR` | `artifacts` | Model/metrics/registry location. |
| `NIDS_HISTORY_DB` | `data/nids_history.sqlite3` | Local history database. |
| `NIDS_MAX_UPLOAD_MB` | `10` | Flask maximum request size. |
| `NIDS_MAX_BATCH_ROWS` | `5000` | CSV batch row cap. |
| `NIDS_MAX_JSON_BATCH_ROWS` | `500` | JSON batch row cap. |
| `NIDS_MAX_HISTORY_ROWS` | `50000` | History retention bound. |
| `NIDS_CORS_ORIGINS` | local origins | Comma-separated allowed browser origins. |
| `NIDS_RATE_LIMIT_DEFAULT` | `300 per hour` | Flask-Limiter request rate. |
| `NIDS_RATE_LIMIT_STORAGE_URI` | `memory://` | Use Redis-backed storage for multiple workers. |
| `NIDS_ENV` | `development` | Set `production` for deployment behavior/documentation. |
| `NIDS_SECRET_KEY` | development-only sample | Change to a high-entropy value before deployment. |

## Testing

```bash
python -m pytest -q
```

The tests use small deterministic fixtures for software contracts only. They do not claim model performance and do not replace a training run on KDD data.

## Deployment and security boundary

The container runs Gunicorn as a non-root user, exposes a health check, sets response security headers, bounds uploads/history and supports rate limiting. For a real deployment:

1. Terminate TLS and place the app behind a network boundary and identity-aware proxy/SSO. This repository does **not** implement user authentication or authorization.
2. Set a unique secret, explicit CORS origins, strong rate-limit storage (Redis for multi-worker), and restrictive network access.
3. Use managed shared persistence (e.g. PostgreSQL) instead of local SQLite when running replicas; protect/expire retained traffic records according to policy.
4. Keep model artifacts trusted and immutable; joblib is a pickle-based format and can execute code on load.
5. Add a packet/flow collector and schema adapter only after security review. This app itself does not capture packets.
6. Monitor logs and model drift, validate against current authorized traffic, and review false positives/rare-class misses with an analyst before automated response.

## Limitations, ethics and future work

KDD Cup 1999 is a historical benchmark with known age, repeated examples, and class-distribution limitations. A high benchmark score would not establish production detection performance. Model confidence can be miscalibrated; a confidence-based severity is an operational heuristic, not incident severity. SHAP explains model output, not causality. The application is not a replacement for a SIEM, endpoint controls, packet sensor, analyst review, or current adversarial validation. Network records can contain sensitive metadata; minimize collection and retention, obtain authorization, and control access.

Future work: evaluate newer datasets (e.g. CIC-IDS or UNSW-NB15 under their terms), time-/host-based external validation, probability calibration, drift monitoring, authenticated RBAC, managed audit storage, analyst feedback, threshold calibration per environment, robust adversarial testing, streaming telemetry integration and a maintained packet/flow adapter.

## GitHub

Published repository: [MEHULGAJJAR1/ai-powered-network-intrusion-detection-system](https://github.com/MEHULGAJJAR1/ai-powered-network-intrusion-detection-system). See [`PUSH_TO_GITHUB.md`](PUSH_TO_GITHUB.md) for clone and follow-up push instructions.

## Project structure

```text
nids/                 Flask API, schemas, data adapter, ML, history service, dashboard assets
scripts/              KDD downloader, synthetic-only demo generator, report appendix renderer
artifacts/            Generated model registry, joblib pipelines and measured metrics (not committed)
data/                 Dataset input, processed area and runtime history (not committed)
docs/openapi.json      OpenAPI 3.0 specification
tests/                 pytest unit/API tests
train.py               Reproducible train/tune/select/persist CLI
evaluate.py            Re-run held-out evaluation from the recorded split
PROJECT_REPORT.md      Technical project report
RESUME_READY.md        Resume-ready summary and implemented-feature bullets
```
