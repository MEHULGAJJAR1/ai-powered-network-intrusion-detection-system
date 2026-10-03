from __future__ import annotations

import numpy as np

from nids.ml.evaluation import evaluate_predictions


def test_metrics_include_macro_weighted_per_class_recall_fpr_and_auc():
    y_true = np.tile(np.arange(5), 4)
    y_pred = y_true.copy()
    probabilities = np.eye(5)[y_true] * 0.9 + 0.02
    result = evaluate_predictions(y_true, y_pred, probabilities, model_classes=np.arange(5))
    assert result["overall"]["accuracy"] == 1.0
    assert result["overall"]["f1_macro"] == 1.0
    assert result["overall"]["f1_weighted"] == 1.0
    assert result["overall"]["roc_auc_ovr_macro"] == 1.0
    assert set(result["per_class"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert result["per_class"]["U2R"]["recall"] == 1.0
    assert result["per_class"]["R2L"]["false_positive_rate"] == 0.0
    assert len(result["confusion_matrix"]["values"]) == 5


def test_roc_auc_is_none_when_test_split_lacks_a_class():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    probabilities = np.eye(5)[y_true]
    result = evaluate_predictions(y_true, y_pred, probabilities, model_classes=np.arange(5))
    assert result["overall"]["roc_auc_ovr_macro"] is None
    assert result["per_class"]["U2R"]["support"] == 0
