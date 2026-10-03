"""Flask application factory, safe error handling, and serving configuration."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from nids import __version__
from nids.config import PACKAGE_ROOT, settings
from nids.logging_config import configure_logging
from nids.ml.inference import InferenceEngine
from nids.services.history import HistoryStore
from nids.api.routes import api

logger = logging.getLogger(__name__)


def create_app(config_override: dict | None = None) -> Flask:
    """Create a Flask app; inference is automatically loaded if artifacts exist."""
    configure_logging(settings.log_level)
    app = Flask(
        __name__,
        template_folder="../web/templates",
        static_folder="../web/static",
        static_url_path="/static",
    )
    app.config.from_mapping(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024,
        MAX_BATCH_ROWS=settings.max_batch_rows,
        NIDS_MAX_BATCH_ROWS=settings.max_batch_rows,
        NIDS_MAX_JSON_BATCH_ROWS=settings.max_json_batch_rows,
        NIDS_MAX_HISTORY_ROWS=settings.max_history_rows,
        NIDS_EXPLORATION_MAX_ROWS=settings.exploration_max_rows,
        NIDS_DATA_PATH=str(settings.data_path),
        NIDS_ARTIFACT_DIR=str(settings.artifact_dir),
        NIDS_MODEL_PATH=str(settings.model_path),
        NIDS_REGISTRY_PATH=str(settings.registry_path),
        NIDS_METRICS_PATH=str(settings.metrics_path),
        NIDS_HISTORY_DB=str(settings.history_db),
        NIDS_ENV=settings.environment,
        APP_VERSION=__version__,
        TESTING=False,
        JSON_SORT_KEYS=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    overrides = config_override or {}
    app.config.update(overrides)
    app.config["MAX_CONTENT_LENGTH"] = int(app.config.get("MAX_CONTENT_LENGTH", settings.max_upload_mb * 1024 * 1024))

    origins = app.config.get("NIDS_CORS_ORIGINS", settings.cors_origins)
    if isinstance(origins, str):
        origins = [value.strip() for value in origins.split(",") if value.strip()]
    CORS(app, resources={r"/api/*": {"origins": list(origins), "max_age": 600}})

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[] if app.config.get("TESTING") else [settings.rate_limit_default],
        storage_uri=settings.rate_limit_storage_uri,
        headers_enabled=True,
        strategy="fixed-window",
    )
    limiter.init_app(app)
    app.extensions["nids_limiter"] = limiter

    injected_engine = overrides.get("INFERENCE_ENGINE") if "INFERENCE_ENGINE" in overrides else None
    if "INFERENCE_ENGINE" in overrides:
        engine = injected_engine
    else:
        try:
            engine = InferenceEngine.load(
                app.config["NIDS_MODEL_PATH"],
                app.config["NIDS_REGISTRY_PATH"],
                Path(app.config["NIDS_ARTIFACT_DIR"]) / "explainability_background.joblib",
            )
            logger.info("Production model loaded", extra={"event": "model_loaded", "model": engine.model_name,
                                                           "version": engine.model_version})
        except FileNotFoundError:
            engine = None
            logger.warning("Production model not found; inference endpoints are disabled until training",
                           extra={"event": "model_unavailable", "artifact": app.config["NIDS_MODEL_PATH"]})
        except Exception:
            engine = None
            logger.exception("Production model could not be loaded", extra={"event": "model_load_failed"})
    app.extensions["nids_engine"] = engine

    injected_store = overrides.get("HISTORY_STORE") if "HISTORY_STORE" in overrides else None
    history_store = injected_store or HistoryStore(
        app.config["NIDS_HISTORY_DB"], app.config["NIDS_MAX_HISTORY_ROWS"]
    )
    app.extensions["nids_history"] = history_store
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return render_template("index.html", app_version=__version__)

    @app.get("/api/openapi.json")
    def openapi_spec():
        spec_path = PACKAGE_ROOT / "docs" / "openapi.json"
        return app.response_class(spec_path.read_text(encoding="utf-8"), mimetype="application/json")

    @app.get("/api/docs")
    def api_docs():
        return render_template("api_docs.html", app_version=__version__)

    @app.before_request
    def assign_request_id():
        g.request_id = uuid.uuid4().hex[:16]

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )
        response.headers.setdefault("X-Request-ID", getattr(g, "request_id", "unknown"))
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": f"Request exceeds the configured upload limit of {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."}), 413

    @app.errorhandler(429)
    def rate_limited(_error):
        return jsonify({"error": "Rate limit exceeded. Retry after the reset interval."}), 429

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "API endpoint not found."}), 404
        return render_template("not_found.html"), 404

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": error.description or "HTTP request failed."}), error.code or 400
        return render_template("error.html", request_id=getattr(g, "request_id", None)), error.code or 400

    @app.errorhandler(Exception)
    def unexpected_error(error):
        logger.exception("Unhandled request error", extra={"event": "unhandled_error", "path": request.path})
        if request.path.startswith("/api/"):
            return jsonify({"error": "An unexpected server error occurred.", "request_id": getattr(g, "request_id", None)}), 500
        return render_template("error.html", request_id=getattr(g, "request_id", None)), 500

    logger.info("Application initialized", extra={"event": "app_initialized", "version": __version__,
                                                    "model_available": engine is not None,
                                                    "environment": app.config["NIDS_ENV"]})
    return app
