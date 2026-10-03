#!/usr/bin/env python3
"""Re-evaluate persisted model candidates on the reproducible group-held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from nids.config import settings
from nids.data.adapter import read_kdd_dataset
from nids.ml.evaluation import evaluate_predictions
from nids.ml.training import make_group_aware_splits
from nids.schema import ID_TO_LABEL, LABEL_TO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.data_path)
    parser.add_argument("--artifacts", type=Path, default=settings.artifact_dir)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Use the recorded cap if omitted; 0 uses the full adapted file")
    parser.add_argument("--seed", type=int, default=None, help="Use recorded seed if omitted")
    args = parser.parse_args()
    artifacts = args.artifacts.expanduser().resolve()
    registry_path = artifacts / "registry.json"
    if not registry_path.is_file():
        raise SystemExit(f"Model registry not found: {registry_path}; train models first.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected_cap = args.max_rows
    if selected_cap is None:
        selected_cap = int(registry["dataset"].get("rows_used_for_experiment", 0))
    seed = args.seed if args.seed is not None else int(registry.get("random_seed", 42))

    X, y_labels, _ = read_kdd_dataset(args.data, deduplicate=True)
    y = y_labels.map(LABEL_TO_ID).to_numpy(dtype=int)
    if selected_cap and len(X) > selected_cap:
        selected, _ = train_test_split(
            np.arange(len(X)), train_size=selected_cap, random_state=seed, stratify=y
        )
        selected.sort()
        X = X.iloc[selected].reset_index(drop=True)
        y = y[selected]
    train_idx, validation_idx, test_idx, _ = make_group_aware_splits(X, y, seed=seed)
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    model_paths = [
        (artifacts / entry["artifact"], name)
        for name, entry in registry.get("models", {}).items()
        if entry.get("artifact")
    ]
    if not model_paths:
        model_paths = [(artifacts / "production_model.joblib", registry.get("active_model", "production_model"))]
    results = {}
    for path, model_name in model_paths:
        if not path.is_file():
            continue
        pipeline = joblib.load(path)
        predictions = pipeline.predict(X_test).astype(int)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        classes = pipeline.named_steps["classifier"].classes_
        results[model_name] = evaluate_predictions(
            y_test, predictions, probabilities, model_classes=classes
        )
    if not results:
        raise SystemExit("No model files were available for evaluation.")
    output = {
        "available": True,
        "dataset": str(args.data.resolve()),
        "rows_used": int(len(X)),
        "test_rows": int(len(test_idx)),
        "seed": seed,
        "group_disjoint_by_feature_fingerprint": True,
        "metrics": results,
    }
    evaluation_dir = artifacts / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    target = evaluation_dir / "reevaluated_metrics.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "models": list(results)}, indent=2))


if __name__ == "__main__":
    main()
