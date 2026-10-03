from __future__ import annotations

import io

import pandas as pd

from nids.api.app import create_app
from nids.services.history import HistoryStore
from tests.conftest import make_record


def test_api_health_and_model_info(api_client):
    health = api_client.get("/api/health")
    assert health.status_code == 200
    assert health.json["model_available"] is True
    assert health.json["status"] == "healthy"
    info = api_client.get("/api/model-info")
    assert info.status_code == 200
    assert info.json["available"] is True
    assert info.json["feature_count"] == 41


def test_single_prediction_and_history(api_client):
    response = api_client.post("/api/predict", json={"record": make_record(12)})
    assert response.status_code == 200
    assert response.json["prediction"] in {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert len(response.json["probabilities"]) == 5
    assert "record" not in response.json
    history = api_client.get("/api/history?page=1&per_page=10")
    assert history.json["total"] == 1
    assert history.json["items"][0]["prediction"] == response.json["prediction"]
    stats = api_client.get("/api/stats").json
    assert stats["total_traffic_analyzed"] == 1


def test_batch_prediction_and_invalid_batch_inputs(api_client):
    records = [make_record(seed) for seed in (21, 22, 23)]
    response = api_client.post("/api/predict-batch", json={"records": records})
    assert response.status_code == 200
    assert response.json["count"] == 3
    assert len(response.json["results"]) == 3
    bad = dict(records[0])
    bad.pop("duration")
    response = api_client.post("/api/predict-batch", json={"records": [records[0], bad]})
    assert response.status_code == 400
    assert api_client.get("/api/stats").json["total_traffic_analyzed"] == 3
    too_many = api_client.post("/api/predict-batch", json={"records": records * 2})
    assert too_many.status_code == 413


def test_invalid_single_prediction_has_useful_error(api_client):
    response = api_client.post("/api/predict", json={"record": {"duration": 0}})
    assert response.status_code == 400
    assert "missing required features" in response.json["error"]
    invalid = make_record(31)
    invalid["serror_rate"] = 4
    response = api_client.post("/api/predict", json={"record": invalid})
    assert response.status_code == 400
    assert "between 0 and 1" in response.json["error"]
    assert "Traceback" not in response.get_data(as_text=True)


def test_csv_upload_validates_schema_predicts_and_downloads(api_client):
    frame = pd.DataFrame([make_record(51), make_record(52)])
    data = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    response = api_client.post("/api/upload-csv", data={"file": (data, "traffic.csv")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "predicted_category" in response.get_data(as_text=True)
    assert int(response.headers["X-NIDS-Summary"] and __import__("json").loads(response.headers["X-NIDS-Summary"])["count"]) == 2
    assert api_client.get("/api/stats").json["total_traffic_analyzed"] == 2


def test_csv_upload_rejects_wrong_extension_and_schema(api_client):
    bad_extension = api_client.post("/api/upload-csv", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "traffic.txt")}, content_type="multipart/form-data")
    assert bad_extension.status_code == 400
    bad_schema = api_client.post("/api/upload-csv", data={"file": (io.BytesIO(b"duration,label\n0,Normal\n"), "traffic.csv")}, content_type="multipart/form-data")
    assert bad_schema.status_code == 400
    assert "missing_features" in bad_schema.json["details"]


def test_health_is_available_without_model(tmp_path):
    app = create_app({
        "TESTING": True,
        "INFERENCE_ENGINE": None,
        "HISTORY_STORE": HistoryStore(tmp_path / "empty.sqlite3"),
        "NIDS_DATA_PATH": str(tmp_path / "no-data.csv"),
        "NIDS_HISTORY_DB": str(tmp_path / "empty.sqlite3"),
    })
    response = app.test_client().get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "degraded"
    assert response.json["model_available"] is False
    assert app.test_client().post("/api/predict", json={"record": make_record(1)}).status_code == 503
