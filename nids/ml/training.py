"""Reproducible training pipeline with group-aware held-out evaluation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version as package_version
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split

from nids import __version__
from nids.data.adapter import read_kdd_dataset
from nids.ml.evaluation import evaluate_predictions, validation_selection_score
from nids.ml.models import BalancedXGBClassifier
from nids.ml.preprocessing import build_pipeline
from nids.schema import (
    CLASS_ORDER,
    ID_TO_LABEL,
    LABEL_TO_ID,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    feature_metadata,
)

logger = logging.getLogger(__name__)


def _feature_groups(X: pd.DataFrame) -> np.ndarray:
    """Fingerprint feature rows so identical traffic records never cross splits."""
    canonical = X.astype("string").fillna("<MISSING>")
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)
    return hashes.astype(str)


def _class_group_counts(y: np.ndarray, groups: np.ndarray) -> dict[int, int]:
    return {
        class_id: len(set(groups[y == class_id]))
        for class_id in range(len(CLASS_ORDER))
    }


def _choose_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    target_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    overall = np.bincount(y, minlength=len(CLASS_ORDER)).astype(float)
    overall /= max(overall.sum(), 1)
    candidates = []
    for train_indices, heldout_indices in splitter.split(X, y, groups):
        if set(np.unique(y[heldout_indices])) != set(range(len(CLASS_ORDER))):
            continue
        heldout_share = np.bincount(y[heldout_indices], minlength=len(CLASS_ORDER)).astype(float)
        heldout_share /= max(heldout_share.sum(), 1)
        size_error = abs(len(heldout_indices) / len(y) - target_fraction)
        distribution_error = float(np.mean(np.abs(heldout_share - overall)))
        candidates.append((size_error * 2.0 + distribution_error, train_indices, heldout_indices))
    if not candidates:
        raise ValueError(
            "Could not construct a group-disjoint split containing all five classes. "
            "Use the full KDD dataset or a larger class-stratified sample; rare U2R/R2L "
            "records must be represented by enough distinct feature groups."
        )
    _, train_indices, heldout_indices = min(candidates, key=lambda item: item[0])
    return train_indices, heldout_indices


def make_group_aware_splits(
    X: pd.DataFrame, y: Iterable[int], *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/validation/test indices without identical-feature split leakage.

    Nested five-fold StratifiedGroupKFold gives approximately 64/16/20 percent
    partitions while preserving all classes in validation and test.
    """
    labels = np.asarray(list(y), dtype=int)
    groups = _feature_groups(X)
    support = _class_group_counts(labels, groups)
    minimum = min(support.values(), default=0)
    if minimum < 5:
        counts = {ID_TO_LABEL[key]: value for key, value in support.items()}
        raise ValueError(
            "At least five distinct feature groups per class are required for the "
            f"leakage-safe nested split; observed {counts}."
        )
    trainval, test = _choose_fold(
        X, labels, groups, n_splits=5, seed=seed, target_fraction=0.20
    )
    dev_X = X.iloc[trainval]
    dev_y = labels[trainval]
    dev_groups = groups[trainval]
    dev_support = _class_group_counts(dev_y, dev_groups)
    if min(dev_support.values(), default=0) < 5:
        raise ValueError("The development partition does not retain five groups for every class.")
    train_relative, validation_relative = _choose_fold(
        dev_X, dev_y, dev_groups, n_splits=5, seed=seed + 1, target_fraction=0.20
    )
    train = trainval[train_relative]
    validation = trainval[validation_relative]
    if not all(set(np.unique(labels[index])) == set(range(len(CLASS_ORDER))) for index in (train, validation, test)):
        raise ValueError("A split is missing one or more target classes; use more data.")
    return train, validation, test, groups


def _candidate_models(seed: int, requested: list[str] | None = None):
    candidates: dict[str, tuple[Any, dict[str, list[Any]], bool]] = {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced", solver="saga", max_iter=400,
                tol=1e-3, random_state=seed, n_jobs=1,
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=180, class_weight="balanced_subsample",
                n_jobs=1, random_state=seed, max_features="sqrt",
            ),
            {
                "classifier__n_estimators": [140, 220],
                "classifier__max_depth": [None, 24],
                "classifier__min_samples_leaf": [1, 2],
            },
            False,
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = (
            BalancedXGBClassifier(
                XGBClassifier(
                    objective="multi:softprob", num_class=len(CLASS_ORDER),
                    eval_metric="mlogloss", tree_method="hist", n_jobs=1,
                    random_state=seed, verbosity=0,
                )
            ),
            {
                "classifier__estimator__n_estimators": [120, 200],
                "classifier__estimator__max_depth": [4, 6],
                "classifier__estimator__learning_rate": [0.05, 0.1],
                "classifier__estimator__subsample": [0.8, 1.0],
                "classifier__estimator__colsample_bytree": [0.8, 1.0],
                "classifier__estimator__reg_lambda": [1.0, 5.0],
            },
            True,
        )
    except ImportError:
        logger.warning("XGBoost is not installed; skipping that candidate")

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = (
            LGBMClassifier(
                objective="multiclass", class_weight="balanced", n_jobs=1,
                random_state=seed, verbosity=-1,
            ),
            {
                "classifier__n_estimators": [120, 200],
                "classifier__num_leaves": [31, 63],
                "classifier__learning_rate": [0.05, 0.1],
            },
            False,
        )
    except ImportError:
        pass

    if requested:
        unknown = sorted(set(requested) - set(candidates))
        if unknown:
            available = ", ".join(sorted(candidates))
            raise ValueError(f"Unknown/unavailable model(s): {', '.join(unknown)}. Available: {available}")
        candidates = {key: candidates[key] for key in requested}
    return candidates


def _predict_probabilities(estimator, X):
    predictions = estimator.predict(X).astype(int)
    probabilities = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
    return predictions, probabilities


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp, compress=3)
    os.replace(temp, path)


def train_project(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_rows: int | None = 100_000,
    cv_folds: int = 3,
    n_iter: int = 2,
    search_jobs: int = 1,
    requested_models: list[str] | None = None,
) -> dict[str, Any]:
    """Tune, compare, refit, test, and persist candidates and the selected model.

    Hyperparameter CV and the outer held-out partitions are group-disjoint by
    feature fingerprint. The validation macro-F1 selects the production candidate;
    the test set is only reported after selection and is never used to choose it.
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    data_path = Path(data_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registry_versions").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_versions").mkdir(parents=True, exist_ok=True)

    X, y_text, load_report = read_kdd_dataset(data_path, deduplicate=True)
    y = y_text.map(LABEL_TO_ID).to_numpy(dtype=int)
    if max_rows and len(X) > max_rows:
        if max_rows < 10 or max_rows < len(CLASS_ORDER) * 5:
            raise ValueError("max_rows is too small for a five-class stratified sample")
        indices = np.arange(len(X))
        selected, _ = train_test_split(
            indices, train_size=max_rows, random_state=seed, stratify=y
        )
        selected.sort()
        X = X.iloc[selected].reset_index(drop=True)
        y = y[selected]
    sample_rows = len(X)
    if set(np.unique(y)) != set(range(len(CLASS_ORDER))):
        missing = [ID_TO_LABEL[i] for i in range(len(CLASS_ORDER)) if i not in set(np.unique(y))]
        raise ValueError(f"Training data does not contain every required class: {missing}")

    train_idx, validation_idx, test_idx, groups = make_group_aware_splits(X, y, seed=seed)
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_validation, y_validation = X.iloc[validation_idx], y[validation_idx]
    X_dev, y_dev = X.iloc[np.concatenate([train_idx, validation_idx])], y[np.concatenate([train_idx, validation_idx])]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    train_groups = groups[train_idx]
    support = _class_group_counts(y_train, train_groups)
    folds = min(cv_folds, min(support.values()))
    if folds < 2:
        raise ValueError("Not enough distinct feature groups per class for stratified CV")

    candidates = _candidate_models(seed, requested_models)
    if not candidates:
        raise RuntimeError("No model candidates are available")
    model_runs: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    val_scores: dict[str, float] = {}
    cv_splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    for name, (classifier, distributions, needs_weights) in candidates.items():
        logger.info("Starting model search", extra={"event": "training_started", "model": name,
                                                    "rows": len(X_train), "cv_folds": folds})
        pipeline = build_pipeline(classifier)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=distributions,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv_splitter,
            refit=True,
            n_jobs=search_jobs,
            random_state=seed,
            verbose=0,
            error_score="raise",
            return_train_score=False,
        )
        start = time.perf_counter()
        # XGBoost's adapter computes balanced weights inside each fold's own fit.
        search.fit(X_train, y_train, groups=train_groups)
        validation_pred, validation_probs = _predict_probabilities(search.best_estimator_, X_validation)
        validation_metrics = evaluate_predictions(
            y_validation, validation_pred, validation_probs,
            model_classes=search.best_estimator_.named_steps["classifier"].classes_,
        )
        val_scores[name] = validation_selection_score(validation_metrics)
        model_runs[name] = {
            "validation": validation_metrics,
            "cv_best_macro_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "search_seconds": round(time.perf_counter() - start, 3),
            "imbalance_strategy": "balanced sample weights in CV" if needs_weights else "class_weight=balanced",
        }
        fitted_models[name] = search.best_estimator_
        logger.info("Model search completed", extra={"event": "training_candidate_complete",
                                                       "model": name,
                                                       "validation_macro_f1": val_scores[name]})

    # Select by validation macro-F1; deterministic tie-break by name.
    selected_name = sorted(val_scores, key=lambda key: (-val_scores[key], key))[0]
    selected_params = model_runs[selected_name]["best_params"]
    _ = selected_params  # retained in the registry for auditability
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    model_version = f"1.0.0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Refit every tuned candidate on train+validation before final held-out test evaluation.
    # This uses no information from the test partition.
    test_models: dict[str, Any] = {}
    for name, estimator in fitted_models.items():
        refit_start = time.perf_counter()
        # BalancedXGBClassifier derives final-fit weights from y_dev in its own fit.
        estimator.fit(X_dev, y_dev)
        test_pred, test_probs = _predict_probabilities(estimator, X_test)
        test_metrics = evaluate_predictions(
            y_test, test_pred, test_probs,
            model_classes=estimator.named_steps["classifier"].classes_,
        )
        model_runs[name]["test"] = test_metrics
        model_runs[name]["refit_seconds"] = round(time.perf_counter() - refit_start, 3)
        model_runs[name]["final_training_rows"] = int(len(X_dev))
        test_models[name] = estimator
        versioned_artifact = f"models/{name}-{model_version}.joblib"
        model_runs[name]["artifact"] = versioned_artifact
        _atomic_joblib_dump(estimator, output_dir / versioned_artifact)

    # The production selection remains the validation winner. All test results are
    # retained for the comparison dashboard but did not influence the selection.
    production_pipeline = test_models[selected_name]
    production_path = output_dir / "production_model.joblib"
    _atomic_joblib_dump(production_pipeline, production_path)
    _atomic_joblib_dump(production_pipeline.named_steps["preprocessor"], output_dir / "preprocessor.joblib")
    # A deterministic training-only background sample supports global SHAP summaries.
    background = X_dev.sample(n=min(256, len(X_dev)), random_state=seed).copy()
    _atomic_joblib_dump(background, output_dir / "explainability_background.joblib")
    metrics_doc = {
        "available": True,
        "selection_criterion": "validation_macro_f1",
        "selected_model": selected_name,
        "selected_validation_macro_f1": val_scores[selected_name],
        "test_set_usage": "held out; reported after validation-based model selection",
        "class_order": list(CLASS_ORDER),
        "models": model_runs,
        "split": {
            "train_rows_for_search": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "development_rows_for_final_refit": int(len(X_dev)),
            "test_rows": int(len(test_idx)),
            "group_disjoint_by_feature_fingerprint": True,
            "class_distribution": {
                "search_train": {ID_TO_LABEL[index]: int(count) for index, count in enumerate(np.bincount(y_train, minlength=len(CLASS_ORDER)))},
                "validation": {ID_TO_LABEL[index]: int(count) for index, count in enumerate(np.bincount(y_validation, minlength=len(CLASS_ORDER)))},
                "test": {ID_TO_LABEL[index]: int(count) for index, count in enumerate(np.bincount(y_test, minlength=len(CLASS_ORDER)))},
            },
            "seed": seed,
        },
    }
    registry = {
        "registry_schema_version": 1,
        "model_name": selected_name,
        "active_model": selected_name,
        "version": model_version,
        "trained_at": created_at,
        "dataset": {
            "name": "KDD Cup 1999",
            "source_file": data_path.name,
            "subset": "10% training subset" if "10_percent" in data_path.name.lower() else "operator-supplied KDD-format file",
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "rows_before_sampling": int(load_report.rows_after_deduplication),
            "experiment_row_cap": int(max_rows) if max_rows else None,
            "rows_used_for_experiment": int(sample_rows),
            "duplicate_rows_removed": int(load_report.duplicate_rows_removed),
            "class_distribution_after_adapter": load_report.class_distribution,
            "class_distribution_used_for_experiment": {
                ID_TO_LABEL[index]: int(count) for index, count in enumerate(np.bincount(y, minlength=len(CLASS_ORDER)))
            },
            "load_report": load_report.to_dict(),
        },
        "feature_count": len(X.columns),
        "features": list(X.columns),
        "feature_metadata": feature_metadata(),
        "class_order": list(CLASS_ORDER),
        "preprocessing_version": PREPROCESSING_VERSION,
        "schema_version": SCHEMA_VERSION,
        "random_seed": seed,
        "packages": {
            "python": platform.python_version(),
            "nids": __version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": _installed_version("scipy"),
            "xgboost": _installed_version("xgboost"),
            "shap": _installed_version("shap"),
            "lightgbm": _installed_version("lightgbm"),
            "flask": _installed_version("Flask"),
        },
        "models": {
            name: {
                "artifact": model_runs[name]["artifact"],
                "validation_macro_f1": val_scores[name],
                "test_metrics": model_runs[name]["test"],
                "best_params": model_runs[name]["best_params"],
            }
            for name in model_runs
        },
        "metrics": model_runs[selected_name]["test"],
        "production_artifact": "production_model.joblib",
        "explainability_background_artifact": "explainability_background.joblib",
        "versioned_registry_artifact": f"registry_versions/{model_version}.json",
        "versioned_metrics_artifact": f"metrics_versions/{model_version}.json",
    }
    metrics_doc["model_version"] = model_version
    metrics_doc["trained_at"] = created_at
    metrics_doc["versioned_metrics_artifact"] = f"metrics_versions/{model_version}.json"
    _write_json(output_dir / registry["versioned_metrics_artifact"], metrics_doc)
    _write_json(output_dir / registry["versioned_registry_artifact"], registry)
    _write_json(output_dir / "metrics.json", metrics_doc)
    _write_json(output_dir / "registry.json", registry)
    logger.info("Training complete", extra={"event": "training_complete", "selected_model": selected_name,
                                             "validation_macro_f1": val_scores[selected_name],
                                             "artifact_dir": str(output_dir)})
    return {"registry": registry, "metrics": metrics_doc}
