from __future__ import annotations

import json

import joblib
import pytest

from nids.ml.inference import InferenceEngine, PredictionInputError, validate_record
from tests.conftest import make_record
from nids.schema import FEATURE_COLUMNS


def test_single_prediction_has_five_probabilities_and_explanation(trained_engine):
    result = trained_engine.predict_record(make_record(45), source="unit_test")
    assert result["prediction"] in {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert set(result["probabilities"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    assert 0 <= result["confidence"] <= 1
    assert result["risk_level"] in {"critical", "high", "medium", "low"}
    assert result["explanation"]["method"] == "linear coefficient × transformed value"
    assert result["explanation"]["scope"] == "local"
    assert len(result["record"]) == len(FEATURE_COLUMNS)


def test_input_validation_rejects_missing_extra_and_invalid_rates():
    record = make_record(46)
    missing = dict(record)
    missing.pop("duration")
    with pytest.raises(PredictionInputError, match="missing required features"):
        validate_record(missing)
    extra = {**record, "label": "DoS"}
    with pytest.raises(PredictionInputError, match="unexpected fields"):
        validate_record(extra)
    bad_rate = dict(record)
    bad_rate["serror_rate"] = 1.25
    with pytest.raises(PredictionInputError, match="between 0 and 1"):
        validate_record(bad_rate)
    negative = dict(record)
    negative["src_bytes"] = -1
    with pytest.raises(PredictionInputError, match="cannot be negative"):
        validate_record(negative)


def test_missing_numeric_values_are_passed_to_fitted_imputer(trained_engine):
    record = make_record(47)
    record["duration"] = None
    result = trained_engine.predict_record(record)
    assert result["prediction"] in {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_model_loading_round_trip(tmp_path, trained_engine):
    model_path = tmp_path / "production_model.joblib"
    registry_path = tmp_path / "registry.json"
    joblib.dump(trained_engine.pipeline, model_path)
    registry_path.write_text(json.dumps({"active_model": "test", "version": "roundtrip-1"}))
    loaded = InferenceEngine.load(model_path, registry_path)
    assert loaded.model_info()["version"] == "roundtrip-1"
    assert loaded.predict_record(make_record(48))["prediction"] in {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_prediction_rejects_long_categories():
    record = make_record(49)
    record["service"] = "x" * 65
    with pytest.raises(PredictionInputError, match="64 characters"):
        validate_record(record)
