# Resume-ready project entry

**AI-Powered Network Intrusion Detection System (NIDS)** · Python, Flask, scikit-learn, XGBoost, Pandas, NumPy, Pydantic, SQLite, Docker, JavaScript

Designed and implemented a full-stack, five-class network-traffic classification application for KDD Cup 1999, with a leakage-aware training pipeline, versioned model artifacts, validated REST inference and a responsive cybersecurity dashboard. Model metrics are generated from the supplied dataset and are intentionally not stated until a real training run has been completed.

## ATS-friendly implementation bullets

- Built a modular KDD Cup 1999 adapter and leakage-safe scikit-learn pipeline for 41 network features, including duplicate detection, categorical one-hot encoding, numeric outlier clipping/imputation/scaling, and stable train/inference feature ordering.
- Implemented reproducible Logistic Regression, Random Forest and XGBoost model comparison with randomized stratified group cross-validation, class weighting, validation macro-F1 selection, and held-out per-class recall, false-positive rate, confusion-matrix and ROC-AUC reporting.
- Developed a Flask REST API with Pydantic contracts for single/batch inference, secure CSV upload/download, model/health/metrics endpoints, bounded prediction history, structured logging, rate limiting and OpenAPI documentation.
- Created a responsive SOC-style dashboard for prediction history, severity-filtered alerts, model performance, KDD data exploration, global feature importance and SHAP-based tree-model explanations with transparent fallbacks.
- Packaged the service with Docker/Gunicorn and added pytest coverage for data adaptation, preprocessing, group-disjoint splits, model loading, inference validation, API health and CSV batch behavior.

## Tailoring note

Keep these bullets only if you can discuss the corresponding code and trade-offs in an interview. After running training on the authentic KDD file, you may add the measured validation/test results from `artifacts/metrics.json`; do not copy unverified or synthetic metrics into a resume.
