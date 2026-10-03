"""REST endpoints for model inference, observability, history, and exploration."""
from __future__ import annotations

import io
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Blueprint, Response, current_app, jsonify, request, send_file
from pydantic import ValidationError
from werkzeug.utils import secure_filename

from nids.api.schemas import BatchPredictRequest, PredictRequest
from nids.data.exploration import build_dataset_summary
from nids.data.synthetic import SyntheticRecordFactory
from nids.ml.inference import InferenceEngine, PredictionInputError
from nids.schema import CLASS_ORDER, FEATURE_COLUMNS, feature_metadata

logger = logging.getLogger(__name__)
api = Blueprint("api", __name__, url_prefix="/api")
synthetic_factory = SyntheticRecordFactory()
ALLOWED_RISK = {"critical", "high", "medium", "low"}


def _engine() -> InferenceEngine | None:
    return current_app.extensions.get("nids_engine")


def _store():
    return current_app.extensions["nids_history"]


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"field": ".".join(str(part) for part in item.get("loc", [])),
         "message": item.get("msg", "Invalid value"), "type": item.get("type", "value_error")}
        for item in exc.errors()
    ]


def _bad_request(message: str, *, details: Any = None, status: int = 400):
    response: dict[str, Any] = {"error": message}
    if details is not None:
        response["details"] = details
    return jsonify(response), status


def _record_internal(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("record", {})


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "record"}


def _store_result(result: dict[str, Any]) -> None:
    _store().insert(result, _record_internal(result))
    logger.info(
        "Network record classified",
        extra={"event": "prediction", "prediction_id": result["id"],
               "predicted_category": result["prediction"], "confidence": result["confidence"],
               "risk_level": result["risk_level"], "source": result.get("source")},
    )


def _store_batch_results(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    _store().insert_many([(item, _record_internal(item)) for item in results])
    distribution: dict[str, int] = {}
    for item in results:
        distribution[item["prediction"]] = distribution.get(item["prediction"], 0) + 1
    logger.info(
        "Batch predictions persisted",
        extra={"event": "prediction_batch_persisted", "count": len(results),
               "prediction_distribution": distribution, "source": results[0].get("source")},
    )


def _parse_datetime(value: str | None, field: str) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"'{field}' must be an ISO-8601 date/time.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _history_query(*, alerts_only: bool = False):
    try:
        page = int(request.args.get("page", "1"))
        per_page = int(request.args.get("per_page", "25"))
        if page < 1 or not 1 <= per_page <= 100:
            raise ValueError("page must be >= 1 and per_page must be between 1 and 100")
        attack_type = request.args.get("attack_type") or None
        severity = request.args.get("severity") or None
        if attack_type and attack_type not in CLASS_ORDER:
            raise ValueError(f"attack_type must be one of: {', '.join(CLASS_ORDER)}")
        if severity and severity.lower() not in ALLOWED_RISK:
            raise ValueError("severity must be critical, high, medium, or low")
        min_conf = request.args.get("min_confidence")
        max_conf = request.args.get("max_confidence")
        minimum = float(min_conf) if min_conf is not None else None
        maximum = float(max_conf) if max_conf is not None else None
        if any(value is not None and (not math.isfinite(value) or value < 0 or value > 1) for value in (minimum, maximum)):
            raise ValueError("confidence filters must be between 0 and 1")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("min_confidence cannot exceed max_confidence")
        query = _store().query(
            page=page,
            per_page=per_page,
            attack_type=attack_type,
            severity=severity.lower() if severity else None,
            min_confidence=minimum,
            max_confidence=maximum,
            since=_parse_datetime(request.args.get("since"), "since"),
            until=_parse_datetime(request.args.get("until"), "until"),
            search=request.args.get("search"),
            alerts_only=alerts_only,
            sort_by=request.args.get("sort_by", "timestamp"),
            sort_order=request.args.get("sort_order", "desc"),
            include_record=False,
        )
        return jsonify(query)
    except (ValueError, TypeError) as exc:
        return _bad_request(str(exc))


@api.get("/health")
def health():
    engine = _engine()
    data_path = current_app.config["NIDS_DATA_PATH"]
    dataset_available = Path(data_path).is_file()
    registry_path = Path(current_app.config["NIDS_REGISTRY_PATH"])
    return jsonify({
        "status": "healthy" if engine else "degraded",
        "service": "ai-powered-nids",
        "model_available": engine is not None,
        "dataset_available": dataset_available,
        "model_name": engine.model_name if engine else None,
        "model_version": engine.model_version if engine else None,
        "registry_available": registry_path.is_file(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


@api.get("/schema")
def schema():
    return jsonify({"features": feature_metadata(), "feature_count": len(FEATURE_COLUMNS),
                    "class_order": list(CLASS_ORDER),
                    "missing_values": "Accepted as null/blank and imputed by the fitted pipeline."})


@api.get("/model-info")
def model_info():
    engine = _engine()
    if not engine:
        return jsonify({"available": False, "message": "No trained production model is loaded. Run train.py with the KDD Cup 1999 dataset."})
    return jsonify(engine.model_info())


@api.get("/metrics")
def metrics():
    path = Path(current_app.config["NIDS_METRICS_PATH"])
    if not path.is_file():
        return jsonify({"available": False, "message": "No measured evaluation metrics are available. Train and evaluate using the real KDD dataset."})
    try:
        return jsonify(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read evaluation metrics", extra={"event": "metrics_load_failed"})
        return jsonify({"available": False, "message": "Evaluation metrics could not be loaded."}), 503


@api.get("/feature-importance")
def feature_importance():
    engine = _engine()
    if not engine:
        return jsonify({"available": False, "message": "Train a model to compute feature importance."})
    try:
        limit = min(50, max(1, int(request.args.get("limit", "25"))))
        return jsonify(engine.feature_importance(limit=limit))
    except (ValueError, TypeError) as exc:
        return _bad_request(f"Invalid feature-importance limit: {exc}")


@api.get("/data-summary")
def data_summary():
    try:
        return jsonify(build_dataset_summary(
            current_app.config["NIDS_DATA_PATH"],
            max_rows=current_app.config["NIDS_EXPLORATION_MAX_ROWS"],
        ))
    except Exception:
        logger.exception("Dataset exploration failed", extra={"event": "dataset_summary_failed"})
        return jsonify({"available": False, "message": "Dataset exploration failed. Check the configured KDD file and server logs."}), 500


@api.post("/predict")
def predict():
    engine = _engine()
    if not engine:
        return _bad_request("No trained model is available. Train a production model before requesting predictions.", status=503)
    try:
        payload = PredictRequest.model_validate(request.get_json(silent=True))
    except ValidationError as exc:
        return _bad_request("Request body does not match the prediction contract.", details=_validation_detail(exc))
    except Exception:
        return _bad_request("A JSON object with a 'record' field is required.")
    try:
        result = engine.predict_record(payload.record, source="api")
        _store_result(result)
        return jsonify(_public_result(result))
    except PredictionInputError as exc:
        return _bad_request(str(exc))
    except Exception:
        logger.exception("Prediction failed", extra={"event": "prediction_failed"})
        return jsonify({"error": "Prediction could not be completed. Check the request schema and model logs."}), 500


@api.post("/predict-batch")
def predict_batch():
    engine = _engine()
    if not engine:
        return _bad_request("No trained model is available. Train a production model before requesting predictions.", status=503)
    try:
        payload = BatchPredictRequest.model_validate(request.get_json(silent=True))
    except ValidationError as exc:
        return _bad_request("Request body does not match the batch prediction contract.", details=_validation_detail(exc))
    except Exception:
        return _bad_request("A JSON object containing a non-empty 'records' array is required.")
    configured_limit = current_app.config["NIDS_MAX_JSON_BATCH_ROWS"]
    if len(payload.records) > configured_limit:
        return _bad_request(f"JSON batch exceeds the configured limit of {configured_limit} records.", status=413)
    try:
        results = engine.predict_records(
            payload.records, source="api_batch",
            include_local_explanations=payload.include_local_explanations,
        )
        _store_batch_results(results)
        prediction_counts = {}
        for result in results:
            prediction_counts[result["prediction"]] = prediction_counts.get(result["prediction"], 0) + 1
        response = {
            "count": len(results),
            "attack_count": sum(count for label, count in prediction_counts.items() if label != "Normal"),
            "prediction_distribution": prediction_counts,
            "synthetic": False,
            "results": [_public_result(result) for result in results],
        }
        return jsonify(response)
    except PredictionInputError as exc:
        return _bad_request(str(exc))
    except Exception:
        logger.exception("Batch prediction failed", extra={"event": "batch_prediction_failed"})
        return jsonify({"error": "Batch prediction could not be completed. Check the request schema and model logs."}), 500


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


@api.post("/upload-csv")
def upload_csv():
    engine = _engine()
    if not engine:
        return _bad_request("No trained model is available. Train a production model before uploading traffic.", status=503)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _bad_request("Attach a CSV file using the multipart field named 'file'.")
    safe_name = secure_filename(upload.filename)
    if not safe_name or Path(safe_name).suffix.lower() != ".csv":
        return _bad_request("Only files with a .csv extension are accepted.")
    try:
        frame = pd.read_csv(upload.stream, dtype=object, keep_default_na=True)
    except Exception:
        return _bad_request("The uploaded file is not a readable CSV.")
    if frame.empty:
        return _bad_request("The CSV contains no traffic records.")
    if len(frame) > current_app.config["NIDS_MAX_BATCH_ROWS"]:
        return _bad_request(
            f"CSV has {len(frame)} rows; the configured maximum is {current_app.config['NIDS_MAX_BATCH_ROWS']}.",
            status=413,
        )
    columns = [str(column).strip() for column in frame.columns]
    if len(columns) != len(set(columns)):
        return _bad_request("CSV contains duplicate column names after whitespace normalization.")
    frame.columns = columns
    missing = [name for name in FEATURE_COLUMNS if name not in columns]
    unexpected = [name for name in columns if name not in FEATURE_COLUMNS]
    if missing or unexpected:
        details = {}
        if missing:
            details["missing_features"] = missing
        if unexpected:
            details["unexpected_columns"] = unexpected
        return _bad_request("CSV columns do not match the canonical 41-feature inference schema.", details=details)
    frame = frame.loc[:, list(FEATURE_COLUMNS)].astype(object).where(pd.notna(frame.loc[:, list(FEATURE_COLUMNS)]), None)
    records = frame.to_dict(orient="records")
    try:
        results = engine.predict_records(records, source="csv", include_local_explanations=False)
        _store_batch_results(results)
    except PredictionInputError as exc:
        return _bad_request(str(exc))
    except Exception:
        logger.exception("CSV batch prediction failed", extra={"event": "csv_prediction_failed", "rows": len(records)})
        return jsonify({"error": "CSV predictions could not be completed. Check feature values and server logs."}), 500

    output_rows = []
    for result in results:
        row = {**result["record"]}
        row.update({
            "predicted_category": result["prediction"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "prediction_id": result["id"],
            "timestamp": result["timestamp"],
            "model_version": result["model_version"],
            "explanation_method": result["explanation"].get("method"),
            "top_features_global_context": "; ".join(
                item["feature"] for item in result["explanation"].get("top_features", [])
            ),
        })
        for label, probability in result["probabilities"].items():
            row[f"probability_{label.lower()}"] = probability
        output_rows.append(row)
    output = pd.DataFrame(output_rows)
    for column in output.select_dtypes(include="object").columns:
        output[column] = output[column].map(_csv_safe)
    text = output.to_csv(index=False, lineterminator="\n")
    buffer = io.BytesIO(text.encode("utf-8-sig"))
    summary = {
        "count": len(results),
        "attack_count": sum(item["prediction"] != "Normal" for item in results),
        "distribution": {label: sum(item["prediction"] == label for item in results) for label in CLASS_ORDER},
        "synthetic": False,
    }
    response = send_file(
        buffer, mimetype="text/csv", as_attachment=True,
        download_name=f"nids_predictions_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv",
    )
    response.headers["X-NIDS-Summary"] = json.dumps(summary, separators=(",", ":"))
    return response


@api.get("/history/export.csv")
def history_export():
    alerts_only = request.args.get("alerts_only", "false").lower() == "true"
    rows = _store().export_rows(alerts_only=alerts_only)
    frame = pd.DataFrame([
        {key: item.get(key) for key in ("id", "timestamp", "prediction", "confidence", "risk_level", "source", "model_version")}
        for item in rows
    ])
    if frame.empty:
        frame = pd.DataFrame(columns=["id", "timestamp", "prediction", "confidence", "risk_level", "source", "model_version"])
    text = frame.to_csv(index=False, lineterminator="\n")
    return send_file(io.BytesIO(text.encode("utf-8-sig")), mimetype="text/csv", as_attachment=True,
                     download_name="nids_alert_history.csv" if alerts_only else "nids_prediction_history.csv")


@api.get("/history")
def history():
    return _history_query(alerts_only=False)


@api.get("/alerts")
def alerts():
    return _history_query(alerts_only=True)


@api.get("/stats")
def stats():
    return jsonify(_store().stats())


@api.get("/demo-record")
def demo_record():
    return jsonify({
        "synthetic": True,
        "warning": "Synthetic UI/demo features only; not captured telemetry and no ground-truth class is assigned.",
        "record": synthetic_factory.next_record(),
    })


@api.post("/demo/predict")
def demo_predict():
    engine = _engine()
    if not engine:
        return _bad_request("No trained model is available. Synthetic records are only classified by a trained model.", status=503)
    record = synthetic_factory.next_record()
    try:
        result = engine.predict_record(record, source="synthetic_demo")
        _store_result(result)
        public = _public_result(result)
        public["synthetic"] = True
        public["warning"] = "Synthetic UI/demo traffic; prediction is from the loaded model, not observed live network telemetry."
        return jsonify(public)
    except Exception:
        logger.exception("Synthetic simulation prediction failed", extra={"event": "demo_prediction_failed"})
        return jsonify({"error": "Synthetic simulation prediction failed. Check the model and server logs."}), 500
