# PROJECT REPORT — AI-Powered Network Intrusion Detection System

## Abstract

This project implements a modular, full-stack network-intrusion classification system for the five requested classes: **Normal, Denial of Service (DoS), Probe, Remote-to-Local (R2L), and User-to-Root (U2R)**. The backend adapts the legacy KDD Cup 1999 record format, validates the canonical 41 features, builds leakage-aware scikit-learn pipelines, compares multiple classifiers, persists a versioned production pipeline and exposes REST inference. A responsive security-operations dashboard presents history-derived activity, test-set model metrics, dataset exploration and model explanations. The code contains no pre-trained model or hand-entered performance claims: evaluation values are produced only after training on a supplied authentic dataset.

## 1. Problem statement

Network intrusion detection seeks to distinguish benign connection records from malicious behavior and to identify attack families early enough for analysts to investigate. Supervised classification is attractive because it can map structured connection features to a compact taxonomy. The central challenge is not just overall accuracy: attack types are imbalanced, rare R2L/U2R examples can be missed while common classes dominate, repeated records can inflate evaluation, and operational inference must use exactly the same feature transformations as training.

This implementation addresses those concerns as a research/engineering baseline. It is not a live packet sensor, SIEM, or autonomous response system. The interface accepts individual flow/connection records and CSV batches; a separate, explicitly synthetic simulation endpoint exists only for UI walkthroughs.

## 2. Objectives and scope

1. Adapt KDD Cup 1999 source files into a stable, ordered 41-feature contract.
2. Map original attack names into Normal, DoS, Probe, R2L and U2R, rejecting unknown labels.
3. Detect exact duplicates and keep identical feature fingerprints in one data partition.
4. Fit preprocessing only on training folds and apply the serialized transformer unchanged during inference.
5. Train a Logistic Regression baseline, Random Forest and XGBoost model; include LightGBM if the optional package is installed.
6. Tune candidates with randomized stratified group CV, address imbalance by class weights or balanced sample weights, select by validation macro-F1, and reserve a group-disjoint test set for final reporting.
7. Persist versioned model artifacts, feature metadata, source digest, package versions, parameters and measured metrics.
8. Provide input-validated single/batch/CSV prediction, confidence/probability output, transparent policy severity, history and explanation.
9. Deliver a responsive dashboard, API documentation, Docker packaging and automated tests.

## 3. Literature and technical context

The KDD Cup 1999 corpus was derived from the DARPA intrusion-detection evaluation and became a widely used benchmark for connection-level intrusion classification. The benchmark is useful for illustrating feature encoding, class imbalance and evaluation workflows, but it is now historical. Published analyses have cautioned that its repeated records and benchmark construction can affect the interpretation of classifier results. Accordingly, this project reports duplicate handling and uses feature-group-disjoint partitions; it does not assume that a KDD result transfers to a contemporary network.

Contextual references:

- UCI KDD Cup 1999 dataset archive: <https://kdd.ics.uci.edu/databases/kddcup99/>
- M. Tavallaee et al., “A Detailed Analysis of the KDD CUP 99 Data Set,” IEEE Symposium on Computational Intelligence for Security and Defense Applications, 2009.
- R. P. Lippmann et al., “The 1999 DARPA Off-Line Intrusion Detection Evaluation,” Computer Networks, 2000.

The project uses standard supervised-learning baselines, group-aware cross-validation, per-class measures and model explanation. It does not make a novelty claim.

## 4. Dataset and target taxonomy

The default downloader targets the original KDD Cup 1999 10% training file in gzip form. It tries the UCI archive first and falls back to a pinned public GitHub mirror if the UCI data endpoint blocks automated access; the actual source URL and locally computed digest are printed. For strict provenance, operators should supply an approved source/checksum or place an authorized copy manually. The repository does not bundle the dataset. A digest is calculated again at training time and written to the registry.

Each record contains 41 features describing connection duration, protocol, service, status flag, byte counts, login/privilege indicators and traffic-window statistics, plus a source attack label. The adapter accepts a headered canonical CSV, the traditional headerless 42-column file, and a common 43-column variation with a trailing difficulty field. It canonicalizes feature order, lowercases categorical values, coerces numeric fields and explicitly reports duplicate/coercion information.

Original attack names are mapped to five classes. Examples include `neptune`, `smurf`, and `back` for DoS; `satan`, `nmap`, and `ipsweep` for Probe; `guess_passwd`, `ftp_write`, and `httptunnel` for R2L; and `buffer_overflow`, `rootkit`, and `perl` for U2R. The reviewed mapping is in `nids/schema.py`. Unknown labels fail training until a mapping is reviewed.

No corpus dimensions, class frequencies or score values are embedded in the report because the actual source file is not part of this deliverable. The Data Exploration page and `registry.json` calculate and record values from the operator's dataset.

## 5. Preprocessing and feature engineering

The preprocessing contract is implemented once in `nids/ml/preprocessing.py` and included in each complete model pipeline:

- **Validation and ordering:** the adapter requires all 41 feature names, rejects unexpected columns, and restores canonical order before fitting or inference.
- **Missing values:** null/blank numeric values are passed to a median imputer; categorical values are imputed by the most-frequent training category. Inference accepts null/blank fields only when their feature keys are present.
- **Categorical variables:** `protocol_type`, `service`, and `flag` are case-normalized and one-hot encoded with `handle_unknown="ignore"` so a newly observed service does not alter feature order.
- **Numeric scaling:** a robust scaler is fitted on training folds. It is useful for the linear baseline; tree models share the same canonical end-to-end pipeline.
- **Outliers:** a custom quantile clipper learns the 0.5th and 99.5th percentile bounds from each fitting fold and clips held-out values to those learned limits. It does not inspect validation/test distributions.
- **Missingness indicators:** numeric imputation includes missing indicators available at fit time.
- **Serialization:** training writes the fitted pipeline and a separate preprocessor artifact. Inference loads and calls the same fitted pipeline rather than recreating encoders/scalers.

## 6. Class imbalance strategy

Normal and common DoS traffic can greatly outnumber rare U2R/R2L cases in this benchmark. The implementation therefore avoids treating raw accuracy as sufficient evidence. Logistic Regression uses `class_weight="balanced"`; Random Forest uses `class_weight="balanced_subsample"`; XGBoost receives `compute_sample_weight("balanced", y)` during CV and final refitting. Optional LightGBM uses balanced class weighting. These strategies are restricted to model fitting; no oversampling is performed on validation or test partitions.

The primary search and production-selection metric is macro-F1, which gives each class equal influence. The dashboard separately exposes macro precision/recall, weighted F1, per-class recall and one-vs-rest false-positive rate. Rare-class support is displayed next to recall so a small denominator is visible. Accuracy remains available as a secondary measure.

## 7. Algorithms and hyperparameter tuning

The trainer constructs sklearn `Pipeline` objects so preprocessing is refitted inside each CV training fold:

- **Logistic Regression:** class-weighted `saga` baseline with randomized regularization `C` candidates.
- **Random Forest:** balanced-subsample tree ensemble with randomized tree count, depth and leaf-size candidates.
- **XGBoost:** multiclass probability objective, histogram tree method, balanced sample weights and randomized tree/learning-rate/subsampling parameters.
- **LightGBM (optional):** automatically added only if importable; its presence is recorded by candidate artifacts.

The default command uses a row cap of 100,000, three maximum CV folds and two randomized candidates per model to bound the initial search. These are tunable, not asserted optimal. Group-aware `StratifiedGroupKFold` is used for tuning. The best CV estimator is assessed on the validation partition; the winner is selected by validation macro-F1, with a deterministic tie-break by model name. Tuned candidates are then refitted on train+validation. The untouched held-out test set is used to compute the comparison results only after that selection.

### Split and leakage controls

Exact feature+label duplicates are detected and removed by the adapter before the training split. The trainer also computes a feature fingerprint and keeps identical-feature records in a single outer train/validation/test partition. Nested group-stratified folds target approximately 64% search-train, 16% validation and 20% test. If the supplied subset cannot provide enough distinct groups to represent every class safely, training stops with a useful error instead of silently falling back to a leaky random split. The split indices, sizes, seed and group-disjoint policy are persisted in `metrics.json`.

## 8. Experimental setup and measured results

The experiment is reproducible with `train.py --seed 42`; the exact source path, SHA-256, rows after adaptation, sample cap, random seed, package versions, selected model, best parameters and artifact version are written to `artifacts/registry.json`. The measured model comparison and split record are written to `artifacts/metrics.json`.

**No full model-comparison benchmark or performance value is published or bundled.** Temporary smoke runs on an authentic public KDD Cup 1999 10% file verified the training/evaluation paths on a capped sample (including an all-candidate comparison with a reduced search); their artifacts were kept outside this project and their scores are intentionally not reported as project results because they are smoke-test runs, not the configured full experiment. The delivered workspace contains no dataset or trained weights. Therefore no accuracy, precision, recall, F1, ROC-AUC, confusion-matrix count, or best-model claim is stated here. After training, generate a measured appendix from the artifacts:

```bash
python train.py --data data/raw/kddcup.data_10_percent.gz --output artifacts
python evaluate.py --data data/raw/kddcup.data_10_percent.gz --artifacts artifacts
python scripts/write_report_results.py --artifacts artifacts
```

The report appendix `artifacts/evaluation/PROJECT_REPORT_RESULTS.md` contains only values read from the generated registry and metrics. Review the `test` entries in `metrics.json` and `validation` entries separately. Candidate test scores are not used for production selection. The model that wins validation macro-F1 is reported as the validation-selected candidate; no model is described as universally best.

Metrics calculated by the evaluator:

- Accuracy, macro precision, macro recall, macro-F1 and support-weighted F1.
- Macro one-vs-rest ROC-AUC when every class is present and the estimator returns probabilities; otherwise the artifact stores `null`.
- Per-class precision, recall, F1, support, true positives, false negatives, false positives and false-positive rate.
- Five-by-five confusion matrix in canonical class order.

The one-vs-rest false-positive rate is computed as `FP / (FP + TN)` for each class. A zero denominator is represented as zero; support and confusion counts remain available for interpretation.

## 9. Explainability

For supported tree estimators, `ExplainabilityService` attempts SHAP `TreeExplainer` values using a deterministic background sample drawn only from the training/development partition. Local explanations aggregate one-hot contributions back to the original KDD feature name and indicate whether a signed contribution supports or opposes the predicted class output. Global explanations aggregate mean absolute SHAP values when available. The model's built-in tree importance or mean absolute linear coefficients are used as a labeled fallback when SHAP is unavailable or incompatible.

Batch predictions skip per-row SHAP by default for latency and label their output as global feature context. This prevents a global ranking from being mislabeled as an individual record attribution. Explanations characterize the fitted model, are not causal claims, and do not prove that a connection is malicious.

## 10. System architecture and implementation

The system is split into distinct modules:

- `nids/data/adapter.py`: KDD-specific input adaptation, deduplication and feature coercion.
- `nids/ml/preprocessing.py`: serializable transformer graph and quantile clipper.
- `nids/ml/training.py`, `evaluation.py`, `explainability.py`, `inference.py`: fit, compare, explain, validate and serve models.
- `nids/api/app.py`, `routes.py`, `schemas.py`: Flask factory, Pydantic contracts, API endpoints and production-safe error responses.
- `nids/services/history.py`: bounded SQLite history, alert queries and aggregates.
- `nids/web/templates/index.html`, `static/app.css`, `static/app.js`: responsive dashboard without mandatory CDN resources.
- `train.py`, `evaluate.py`: operator-facing CLI entrypoints.

The operational path is:

```text
Data → Preprocessing → Feature Engineering → Model Training → Evaluation
     → Model Registry → Flask API → Frontend Dashboard
```

The active model, fitted preprocessor, versioned candidate joblibs, JSON registry, measured metrics and training-only explanation background are saved in `artifacts/`. Model artifacts are loaded automatically when the Flask application starts. Retraining should be done as an offline deployment step, followed by process restart/redeployment.

## 11. API design

The service publishes an OpenAPI 3.0 document at `/api/openapi.json` and a browser-readable endpoint reference at `/api/docs`.

| Endpoint | Function |
|---|---|
| `GET /api/health` | Service, dataset, model and registry availability. |
| `GET /api/schema` | Ordered feature metadata and class labels. |
| `POST /api/predict` | One record: prediction, class probabilities, confidence, risk policy and explanation. |
| `POST /api/predict-batch` | JSON record batch with configurable cap; optionally requests local explanations for a small batch. |
| `POST /api/upload-csv` | Validates extension, size, row count, schema and row feature values; returns a prediction CSV attachment. |
| `GET /api/model-info` | Active model/version and registry information. |
| `GET /api/metrics` | Measured candidate comparison and test results. |
| `GET /api/feature-importance` | Global model importance/SHAP summary. |
| `GET /api/data-summary` | Dataset dimensions, class distribution, numerical/categorical summaries and correlations. |
| `GET /api/history`, `/api/alerts` | Filtered, sorted, paginated local inference history. |
| `GET /api/stats` | Dashboard aggregates calculated from retained predictions. |
| `GET /api/history/export.csv` | History/alert CSV export. |
| `GET /api/demo-record`, `POST /api/demo/predict` | Clearly synthetic UI walkthrough feature generation/classification. |

Numeric inputs must be finite and non-negative for the KDD schema; rates must lie in `[0,1]`; binary features use 0/1; categorical strings are length-bounded. Validation errors return useful field messages without exposing raw stack traces. CSV output prefixes spreadsheet-formula-leading user-controlled text to reduce formula-injection risk.

## 12. Dashboard and monitoring behavior

Dashboard KPI values represent locally retained inference events, not dataset rows. “Detected attacks” counts non-Normal predictions; “high-risk” counts policy levels high/critical. Confidence-based severity is explicit and configurable in code; it is not a calibrated estimate of incident impact. The history includes source tags so synthetic simulation outputs remain identifiable.

The simulation generator produces feature-shaped examples without ground-truth labels. The actual loaded classifier supplies the displayed predictions. This is not streaming telemetry and does not access network interfaces. The dashboard disables prediction and simulation controls if no trained model is available. The data exploration page displays an unavailable state until a real configured dataset is present.

## 13. Testing and quality controls

`pytest` covers:

- Original label-family mapping, header/difficulty adaptation, unknown-label rejection and duplicate detection.
- Missing values, unknown categories, stable transform width, outlier clipping and group-disjoint partitions.
- Inference probability shape, prediction input validation, model artifact round-trip and explanations.
- Health/model availability, single and batch prediction, CSV download/schema checks, history writes and invalid input messages.
- Macro/weighted metrics, per-class recall/FPR, confusion matrix and undefined ROC-AUC behavior.

Test fixtures are small software-contract fixtures, not a claim of model quality. They never appear in production training or dashboard evaluation artifacts.

## 14. Security, ethics and operational limitations

This repository is a production-oriented baseline, not a fully managed security platform. The Flask service includes CORS configuration, rate limiting, bounded multipart uploads, secure filename handling, Pydantic input contracts, structured JSON logs, generic server errors, response security headers and bounded local history. It **does not implement authentication or authorization**. Expose it only behind TLS, an identity-aware proxy, network controls and appropriate access policy. Memory-backed rate limiting and local SQLite are single-node defaults; use Redis/shared persistence for multi-worker or replicated deployments.

KDD Cup 1999 is a historical dataset and may not reflect present protocols, applications, host behavior, encryption patterns, or adversarial adaptation. Duplicate removal and group splitting reduce one class of leakage but do not eliminate benchmark bias or time/host distribution shift. Class weighting cannot create missing rare-class information. Score calibration, drift, threshold selection and adversarial robustness require deployment-specific studies. No automated blocking is included.

Network feature records can contain sensitive metadata. Only process traffic with authorization, minimize collected fields, define retention/deletion, encrypt storage and restrict access. The system is decision support; an analyst must validate alerts before response.

## 15. Future scope

- External validation with current authorized flow telemetry and modern benchmark datasets.
- Time- and host-disjoint validation, calibration, drift alerts and uncertainty-aware abstention.
- Authenticated RBAC/SSO, managed audit storage, Postgres history and Redis rate limiting.
- Approved packet/flow ingestion service and protocol-versioned feature adapters.
- Analyst feedback loop, environment-specific threshold policy and monitored false-positive review.
- Privacy-preserving data minimization, retention jobs and secure artifact provenance/signing.
- Explainability latency optimization and model-specific explanation fidelity checks.

## Appendix A — run instructions

See [`README.md`](README.md) for Linux/macOS, Windows, Docker, training and test commands. The shortest local flow is:

```bash
python -m pip install -r requirements.txt
python scripts/download_kdd.py
python train.py --data data/raw/kddcup.data_10_percent.gz
python app.py
python -m pytest -q
```
