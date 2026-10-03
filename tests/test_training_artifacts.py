from __future__ import annotations

import json

from nids.ml.inference import InferenceEngine
from nids.ml.training import train_project
from tests.conftest import make_training_frame


def test_reproducible_training_writes_versioned_artifacts_and_registry(tmp_path):
    X, y = make_training_frame(rows_per_class=20)
    original_labels = ["normal.", "neptune.", "satan.", "guess_passwd.", "buffer_overflow."]
    frame = X.copy()
    frame["label"] = [original_labels[int(class_id)] for class_id in y]
    dataset = tmp_path / "tiny-kdd.csv"
    frame.to_csv(dataset, index=False)
    artifacts = tmp_path / "artifacts"

    result = train_project(
        dataset, artifacts, seed=17, max_rows=None, cv_folds=2, n_iter=1,
        requested_models=["logistic_regression"],
    )
    registry = result["registry"]
    metrics = result["metrics"]
    assert registry["feature_count"] == 41
    assert registry["version"].startswith("1.0.0-")
    assert registry["models"]["logistic_regression"]["artifact"].startswith("models/")
    assert (artifacts / registry["models"]["logistic_regression"]["artifact"]).is_file()
    assert (artifacts / registry["versioned_registry_artifact"]).is_file()
    assert (artifacts / registry["versioned_metrics_artifact"]).is_file()
    loaded_registry = json.loads((artifacts / "registry.json").read_text())
    assert loaded_registry["version"] == registry["version"]
    assert metrics["available"] is True
    assert metrics["selected_model"] == "logistic_regression"
    assert metrics["models"]["logistic_regression"]["test"]["overall"]["samples"] > 0
    engine = InferenceEngine.load(artifacts / "production_model.joblib", artifacts / "registry.json",
                                  artifacts / "explainability_background.joblib")
    assert engine.model_info()["available"] is True
