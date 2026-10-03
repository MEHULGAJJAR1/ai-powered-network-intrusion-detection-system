"""Optional SHAP explanations with honest, labeled fallbacks."""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from nids.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _source_feature(encoded_name: str) -> str:
    name = str(encoded_name)
    for feature in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_") or name.endswith("__" + feature):
            return feature
        if f"__{feature}_" in name:
            return feature
    # ColumnTransformer may prefix transformer names. Keep a safe readable label.
    return name.split("__")[-1]


def _array_values(raw_values: Any, *, class_index: int | None = None) -> np.ndarray:
    """Normalize SHAP's list and multi-output ndarray formats to (rows, features)."""
    if isinstance(raw_values, list):
        index = class_index or 0
        raw_values = raw_values[index]
    if hasattr(raw_values, "values") and not isinstance(raw_values, np.ndarray):
        raw_values = raw_values.values
    values = np.asarray(raw_values)
    if values.ndim == 3:
        # Current SHAP is often (samples, features, classes); older releases may
        # return (classes, samples, features).
        if values.shape[0] == 1 and class_index is not None and values.shape[-1] > class_index:
            values = values[0, :, class_index][None, :]
        elif class_index is not None and values.shape[0] > class_index and values.shape[1] == 1:
            values = values[class_index, :, :]
        elif class_index is not None and values.shape[-1] > class_index:
            values = values[:, :, class_index]
        else:
            values = values[..., 0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


class ExplainabilityService:
    """Serve global and local explanations for a fitted preprocessing/model pipeline."""

    def __init__(self, pipeline, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.background = background
        self._explainer = None
        self._shap_checked = False
        self._shap_module = None
        self._global_cache: dict[int, dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    @property
    def classifier(self):
        return self.pipeline.named_steps["classifier"]

    @property
    def preprocessor(self):
        return self.pipeline.named_steps["preprocessor"]

    def _feature_names(self, transformed_width: int | None = None) -> list[str]:
        try:
            names = list(map(str, self.preprocessor.get_feature_names_out()))
        except Exception:
            names = []
        if transformed_width is not None and len(names) != transformed_width:
            names = [f"feature_{index}" for index in range(transformed_width)]
        return names

    def _raw_feature_importance(self) -> tuple[dict[str, float], str]:
        classifier = self.classifier
        if hasattr(classifier, "feature_importances_"):
            values = np.asarray(classifier.feature_importances_, dtype=float)
            method = "tree feature_importances_"
        elif hasattr(classifier, "coef_"):
            values = np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
            method = "mean absolute linear coefficient"
        else:
            return {}, "unavailable"
        names = self._feature_names(len(values))
        grouped: dict[str, float] = defaultdict(float)
        for name, value in zip(names, values):
            if np.isfinite(value):
                grouped[_source_feature(name)] += float(abs(value))
        total = sum(grouped.values())
        if total > 0:
            grouped = {name: value / total for name, value in grouped.items()}
        return dict(grouped), method

    def _get_shap(self):
        if not self._shap_checked:
            self._shap_checked = True
            try:
                import shap
                self._shap_module = shap
            except Exception as exc:  # SHAP is optional at runtime and can fail on ABI mismatch.
                logger.info("SHAP unavailable; using model feature-importance fallback",
                            extra={"event": "shap_unavailable", "reason": str(exc)[:240]})
        return self._shap_module

    def _tree_explainer(self):
        shap = self._get_shap()
        if shap is None or not hasattr(self.classifier, "feature_importances_"):
            return None
        if self._explainer is None:
            try:
                tree_model = getattr(self.classifier, "estimator_", self.classifier)
                self._explainer = shap.TreeExplainer(tree_model)
            except Exception as exc:
                logger.warning("Could not initialize TreeExplainer", extra={"event": "shap_init_failed", "reason": str(exc)[:240]})
                return None
        return self._explainer

    @staticmethod
    def _dense(values):
        if sparse.issparse(values):
            values = values.toarray()
        return np.asarray(values)

    def global_importance(self, limit: int = 25) -> dict[str, Any]:
        cached = self._global_cache.get(limit)
        if cached is not None:
            return cached
        raw_values, fallback_method = self._raw_feature_importance()
        shap = self._get_shap()
        explainer = self._tree_explainer() if shap is not None else None
        method = fallback_method
        if explainer is not None and self.background is not None and len(self.background):
            try:
                transformed = self._dense(self.preprocessor.transform(self.background.loc[:, list(FEATURE_COLUMNS)]))
                # The saved background is a deterministic training-only sample.
                transformed = transformed[: min(len(transformed), 256)]
                raw_shap = explainer.shap_values(transformed)
                if isinstance(raw_shap, list):
                    stack = np.stack([_array_values(item) for item in raw_shap], axis=0)
                    mean_abs = np.mean(np.abs(stack), axis=(0, 1))
                else:
                    array = np.asarray(getattr(raw_shap, "values", raw_shap))
                    if array.ndim == 3:
                        mean_abs = np.mean(np.abs(array), axis=(0, 2))
                    else:
                        mean_abs = np.mean(np.abs(array), axis=0)
                names = self._feature_names(len(mean_abs))
                grouped: dict[str, float] = defaultdict(float)
                for name, value in zip(names, mean_abs):
                    if np.isfinite(value):
                        grouped[_source_feature(name)] += float(value)
                total = sum(grouped.values())
                if total > 0:
                    raw_values = {key: value / total for key, value in grouped.items()}
                    method = "mean absolute SHAP value (training background sample)"
            except Exception as exc:
                logger.warning("Global SHAP summary failed; using model feature importance",
                               extra={"event": "shap_global_failed", "reason": str(exc)[:240]})
        items = sorted(raw_values.items(), key=lambda item: item[1], reverse=True)[:limit]
        result = {
            "available": bool(items),
            "method": method,
            "scope": "global",
            "features": [{"feature": name, "importance": round(value, 8)} for name, value in items],
            "note": "Global importance is an association within this fitted model, not causal evidence.",
        }
        with self._cache_lock:
            self._global_cache[limit] = result
        return result

    def local_explanation(self, raw_record: dict[str, Any], class_id: int, *, limit: int = 5) -> dict[str, Any]:
        frame = pd.DataFrame([raw_record], columns=FEATURE_COLUMNS)
        names = self._feature_names()
        classifier = self.classifier
        class_values = list(getattr(classifier, "classes_", []))
        class_index = class_values.index(class_id) if class_id in class_values else 0
        explainer = self._tree_explainer()
        if explainer is not None:
            try:
                transformed = self._dense(self.preprocessor.transform(frame))
                shap_values = explainer.shap_values(transformed)
                values = _array_values(shap_values, class_index=class_index)[0]
                if len(names) != len(values):
                    names = [f"feature_{i}" for i in range(len(values))]
                grouped_signed: dict[str, float] = defaultdict(float)
                for name, value in zip(names, values):
                    if np.isfinite(value):
                        grouped_signed[_source_feature(name)] += float(value)
                items = sorted(grouped_signed.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
                top = [
                    {"feature": name, "contribution": round(value, 6),
                     "direction": "supports" if value >= 0 else "opposes"}
                    for name, value in items
                ]
                supporting = [item["feature"] for item in top if item["contribution"] > 0]
                opposing = [item["feature"] for item in top if item["contribution"] < 0]
                summary = ""
                if supporting:
                    summary += "Model contributions supporting this class include " + ", ".join(supporting[:3]) + ". "
                if opposing:
                    summary += "Contributions opposing it include " + ", ".join(opposing[:3]) + "."
                if not summary:
                    summary = "No non-zero local contribution was returned for this record."
                return {
                    "method": "SHAP TreeExplainer",
                    "scope": "local",
                    "summary": summary.strip(),
                    "top_features": top,
                    "caveat": "SHAP contributions explain this model output; they do not establish causation.",
                }
            except Exception as exc:
                logger.warning("Local SHAP explanation failed", extra={"event": "shap_local_failed", "reason": str(exc)[:240]})

        # For a linear classifier, coefficient × transformed value is an exact
        # additive contribution to that class's linear score (before softmax).
        if hasattr(classifier, "coef_"):
            try:
                transformed = self._dense(self.preprocessor.transform(frame))[0]
                coefficients = np.asarray(classifier.coef_, dtype=float)
                if coefficients.ndim == 2 and class_index < coefficients.shape[0]:
                    linear_terms = coefficients[class_index] * transformed
                    if len(names) != len(linear_terms):
                        names = [f"feature_{i}" for i in range(len(linear_terms))]
                    grouped_signed: dict[str, float] = defaultdict(float)
                    for name, value in zip(names, linear_terms):
                        if np.isfinite(value):
                            grouped_signed[_source_feature(name)] += float(value)
                    items = sorted(grouped_signed.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
                    top = [
                        {"feature": name, "contribution": round(value, 6),
                         "direction": "supports" if value >= 0 else "opposes"}
                        for name, value in items
                    ]
                    supporting = [item["feature"] for item in top if item["contribution"] > 0]
                    opposing = [item["feature"] for item in top if item["contribution"] < 0]
                    summary = ""
                    if supporting:
                        summary += "Linear score contributions supporting this class include " + ", ".join(supporting[:3]) + ". "
                    if opposing:
                        summary += "Contributions opposing it include " + ", ".join(opposing[:3]) + "."
                    return {
                        "method": "linear coefficient × transformed value",
                        "scope": "local",
                        "summary": summary.strip() or "No non-zero linear contribution was returned for this record.",
                        "top_features": top,
                        "caveat": "This is an additive contribution to a linear class score before probability normalization; it is not causal evidence.",
                    }
            except Exception as exc:
                logger.warning("Linear local explanation failed", extra={"event": "linear_explanation_failed", "reason": str(exc)[:240]})

        global_data = self.global_importance(limit=limit)
        top = [
            {"feature": item["feature"], "contribution": None, "direction": "global context"}
            for item in global_data.get("features", [])[:limit]
        ]
        return {
            "method": "global-importance fallback",
            "scope": "global_context_for_record",
            "summary": (
                "Local SHAP values are unavailable for this model/runtime. The listed fields are "
                "globally influential in this fitted model, not record-specific contributions."
            ),
            "top_features": top,
            "caveat": "Global feature importance is not a per-record explanation and is not causal.",
        }
