"""Metrics designed for an imbalanced five-class intrusion task."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from nids.schema import CLASS_ORDER, ID_TO_LABEL


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 6) if np.isfinite(result) else None


def evaluate_predictions(y_true, y_pred, probabilities=None, *, model_classes=None) -> dict[str, Any]:
    """Return accuracy, macro/weighted scores, per-class recall/FPR, and confusion matrix.

    All reported classes are retained even when a split lacks a class. ROC-AUC is
    omitted (``None``) when one-vs-rest AUC cannot be computed from the split.
    """
    true_ids = np.asarray(y_true, dtype=int)
    pred_ids = np.asarray(y_pred, dtype=int)
    if true_ids.size != pred_ids.size:
        raise ValueError("y_true and y_pred must have the same length")
    labels = list(CLASS_ORDER)
    true_labels = [ID_TO_LABEL.get(int(value), str(value)) for value in true_ids]
    pred_labels = [ID_TO_LABEL.get(int(value), str(value)) for value in pred_ids]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    report = classification_report(
        true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0
    )
    per_class: dict[str, dict[str, Any]] = {}
    total = int(cm.sum())
    for index, label in enumerate(labels):
        tp = int(cm[index, index])
        fn = int(cm[index, :].sum() - tp)
        fp = int(cm[:, index].sum() - tp)
        tn = int(total - tp - fn - fp)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        metric = report.get(label, {})
        per_class[label] = {
            "precision": _number(metric.get("precision", 0.0)),
            "recall": _number(metric.get("recall", 0.0)),
            "f1": _number(metric.get("f1-score", 0.0)),
            "support": int(metric.get("support", 0)),
            "false_positive_rate": _number(fpr),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
        }

    roc_auc: float | None = None
    if probabilities is not None:
        values = np.asarray(probabilities, dtype=float)
        classes = np.asarray(model_classes if model_classes is not None else range(values.shape[1]), dtype=int)
        aligned = np.zeros((len(values), len(labels)), dtype=float)
        for source_index, class_id in enumerate(classes):
            if 0 <= class_id < len(labels) and source_index < values.shape[1]:
                aligned[:, class_id] = values[:, source_index]
        if set(true_ids.tolist()) == set(range(len(labels))) and np.isfinite(aligned).all():
            try:
                roc_auc = _number(
                    roc_auc_score(
                        true_ids,
                        aligned,
                        labels=list(range(len(labels))),
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except (ValueError, TypeError):
                roc_auc = None

    return {
        "overall": {
            "accuracy": _number(accuracy_score(true_labels, pred_labels)) if total else None,
            "precision_macro": _number(precision_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)) if total else None,
            "recall_macro": _number(recall_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)) if total else None,
            "f1_macro": _number(f1_score(true_labels, pred_labels, labels=labels, average="macro", zero_division=0)) if total else None,
            "f1_weighted": _number(f1_score(true_labels, pred_labels, labels=labels, average="weighted", zero_division=0)) if total else None,
            "roc_auc_ovr_macro": roc_auc,
            "samples": total,
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "values": cm.astype(int).tolist()},
        "classification_report": {
            label: {
                "precision": _number(report.get(label, {}).get("precision", 0)),
                "recall": _number(report.get(label, {}).get("recall", 0)),
                "f1-score": _number(report.get(label, {}).get("f1-score", 0)),
                "support": int(report.get(label, {}).get("support", 0)),
            }
            for label in labels
        },
    }


def validation_selection_score(metrics: dict[str, Any]) -> float:
    """Macro-F1 is the primary model-selection criterion; accuracy is not used."""
    value = metrics.get("overall", {}).get("f1_macro")
    return float(value) if value is not None else float("-inf")
