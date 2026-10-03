"""Validated inference engine with automatic artifact loading and explanations."""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nids.ml.explainability import ExplainabilityService
from nids.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    RATE_FEATURES,
)

logger = logging.getLogger(__name__)


class PredictionInputError(ValueError):
    """Raised when a user-supplied network record does not match the schema."""


def validate_record(record: Any) -> dict[str, Any]:
    """Validate, normalize, and order one record while preserving nullable values."""
    if not isinstance(record, dict):
        raise PredictionInputError("Each record must be a JSON object keyed by canonical KDD feature names.")
    missing = [name for name in FEATURE_COLUMNS if name not in record]
    unexpected = sorted(str(name) for name in record if name not in FEATURE_COLUMNS)
    if missing or unexpected:
        issues = []
        if missing:
            issues.append("missing required features: " + ", ".join(missing))
        if unexpected:
            issues.append("unexpected fields: " + ", ".join(unexpected))
        raise PredictionInputError("Invalid record schema: " + "; ".join(issues))

    clean: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        value = record[name]
        if name in CATEGORICAL_FEATURES:
            if value is None or (isinstance(value, str) and not value.strip()):
                clean[name] = None
                continue
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise PredictionInputError(f"Feature '{name}' must be a short text category or null.")
            category = str(value).strip().lower()
            if len(category) > 64:
                raise PredictionInputError(f"Feature '{name}' must be 64 characters or fewer.")
            clean[name] = category
            continue

        if value is None or (isinstance(value, str) and not value.strip()):
            clean[name] = None
            continue
        if isinstance(value, bool):
            raise PredictionInputError(f"Feature '{name}' must be numeric, not a boolean.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"Feature '{name}' must be numeric or null.") from exc
        if not math.isfinite(number):
            raise PredictionInputError(f"Feature '{name}' must be finite; NaN and infinity are not accepted.")
        if abs(number) > 1e12:
            raise PredictionInputError(f"Feature '{name}' is outside the accepted magnitude (±1e12).")
        if number < 0:
            raise PredictionInputError(f"Feature '{name}' cannot be negative for the KDD schema.")
        if name in RATE_FEATURES and number > 1:
            raise PredictionInputError(f"Feature '{name}' is a rate and must be between 0 and 1.")
        if name in BINARY_FEATURES and number not in (0, 1):
            raise PredictionInputError(f"Feature '{name}' must be 0 or 1.")
        if name == "su_attempted" and number not in (0, 1, 2):
            raise PredictionInputError("Feature 'su_attempted' must be 0, 1, or 2 in the legacy KDD schema.")
        clean[name] = number
    return clean


def risk_level(prediction: str, confidence: float) -> str:
    """Transparent policy heuristic; not a ground-truth severity label."""
    if prediction == "Normal":
        return "low"
    if confidence >= 0.95:
        return "critical"
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


class InferenceEngine:
    def __init__(self, pipeline, registry: dict[str, Any] | None = None,
                 background: pd.DataFrame | None = None):
        if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
            raise ValueError("The loaded artifact is not a compatible NIDS preprocessing/model pipeline")
        classifier = pipeline.named_steps["classifier"]
        class_ids = {int(value) for value in getattr(classifier, "classes_", [])}
        if class_ids and class_ids != set(range(len(CLASS_ORDER))):
            raise ValueError(f"Model class IDs {sorted(class_ids)} do not match the required five-class order")
        self.pipeline = pipeline
        self.registry = registry or {}
        self.model_version = str(self.registry.get("version", "unknown"))
        self.explainability = ExplainabilityService(pipeline, background)

    @classmethod
    def load(cls, model_path: str | Path, registry_path: str | Path,
             background_path: str | Path | None = None) -> "InferenceEngine":
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Production model artifact not found: {model_path}")
        pipeline = joblib.load(model_path)
        registry_path = Path(registry_path)
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
        background = None
        if background_path and Path(background_path).is_file():
            try:
                background = joblib.load(background_path)
            except Exception:
                logger.warning("Could not load explainability background sample", extra={"event": "background_load_failed"})
        return cls(pipeline, registry, background)

    @property
    def model_name(self) -> str:
        return str(self.registry.get("active_model", self.registry.get("model_name", "unregistered model")))

    def model_info(self) -> dict[str, Any]:
        dataset = dict(self.registry.get("dataset", {}))
        dataset.pop("path", None)  # Do not disclose the backend's absolute filesystem path.
        return {
            "available": True,
            "model_name": self.model_name,
            "version": self.model_version,
            "trained_at": self.registry.get("trained_at"),
            "feature_count": len(FEATURE_COLUMNS),
            "features": list(FEATURE_COLUMNS),
            "class_order": list(CLASS_ORDER),
            "dataset": dataset,
            "preprocessing_version": self.registry.get("preprocessing_version"),
            "metrics": self.registry.get("metrics", {}),
        }

    def feature_importance(self, limit: int = 25) -> dict[str, Any]:
        return self.explainability.global_importance(limit=limit)

    def _explanation(self, record: dict[str, Any], class_id: int, *, local: bool) -> dict[str, Any]:
        if local:
            return self.explainability.local_explanation(record, class_id)
        global_data = self.explainability.global_importance(limit=5)
        features = [
            {"feature": item["feature"], "contribution": None, "direction": "global context"}
            for item in global_data.get("features", [])
        ]
        return {
            "method": "global-importance context",
            "scope": "global_context_for_record",
            "summary": "Per-record SHAP was skipped for batch latency; these are global model features, not local contributions.",
            "top_features": features,
            "caveat": "Global importance is not a per-record explanation and is not causal.",
        }

    def predict_records(
        self,
        records: list[dict[str, Any]],
        *,
        source: str = "api",
        include_local_explanations: bool = False,
    ) -> list[dict[str, Any]]:
        if not records:
            raise PredictionInputError("At least one record is required.")
        normalized = [validate_record(record) for record in records]
        frame = pd.DataFrame(normalized, columns=FEATURE_COLUMNS)
        predicted_ids = self.pipeline.predict(frame).astype(int)
        probability_matrix = self.pipeline.predict_proba(frame)
        classifier = self.pipeline.named_steps["classifier"]
        model_classes = [int(value) for value in classifier.classes_]
        results = []
        for index, (record, class_id) in enumerate(zip(normalized, predicted_ids)):
            prediction = ID_TO_LABEL[int(class_id)]
            aligned = {label: 0.0 for label in CLASS_ORDER}
            for column_index, model_class in enumerate(model_classes):
                aligned[ID_TO_LABEL[model_class]] = float(probability_matrix[index, column_index])
            confidence = float(aligned[prediction])
            level = risk_level(prediction, confidence)
            local = include_local_explanations and (len(records) <= 25 or len(records) == 1)
            explanation = self._explanation(record, int(class_id), local=local)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            results.append({
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "prediction": prediction,
                "confidence": round(confidence, 8),
                "probabilities": {key: round(value, 8) for key, value in aligned.items()},
                "risk_level": level,
                "explanation": explanation,
                "model_version": self.model_version,
                "source": source,
                # Internal field used by the bounded local history store; the API strips it.
                "record": record,
            })
        logger.info(
            "Prediction batch completed",
            extra={"event": "prediction_batch", "count": len(results),
                   "model_name": self.model_name, "source": source},
        )
        return results

    def predict_record(self, record: dict[str, Any], *, source: str = "api") -> dict[str, Any]:
        return self.predict_records([record], source=source, include_local_explanations=True)[0]
