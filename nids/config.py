"""Environment-based configuration for the NIDS service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PACKAGE_ROOT / ".env", override=False)


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Secrets are never logged."""

    project_root: Path = PACKAGE_ROOT
    artifact_dir: Path = _path_env("NIDS_ARTIFACT_DIR", PACKAGE_ROOT / "artifacts")
    data_path: Path = _path_env(
        "NIDS_DATA_PATH", PACKAGE_ROOT / "data" / "raw" / "kddcup.data_10_percent.gz"
    )
    history_db: Path = _path_env(
        "NIDS_HISTORY_DB", PACKAGE_ROOT / "data" / "nids_history.sqlite3"
    )
    max_upload_mb: int = int(os.getenv("NIDS_MAX_UPLOAD_MB", "10"))
    max_batch_rows: int = int(os.getenv("NIDS_MAX_BATCH_ROWS", "5000"))
    max_json_batch_rows: int = int(os.getenv("NIDS_MAX_JSON_BATCH_ROWS", "500"))
    max_history_rows: int = int(os.getenv("NIDS_MAX_HISTORY_ROWS", "50000"))
    exploration_max_rows: int = int(os.getenv("NIDS_EXPLORATION_MAX_ROWS", "100000"))
    cors_origins: tuple[str, ...] = _csv_env(
        "NIDS_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
    )
    rate_limit_storage_uri: str = os.getenv("NIDS_RATE_LIMIT_STORAGE_URI", "memory://")
    rate_limit_default: str = os.getenv("NIDS_RATE_LIMIT_DEFAULT", "300 per hour")
    log_level: str = os.getenv("NIDS_LOG_LEVEL", "INFO").upper()
    secret_key: str = os.getenv("NIDS_SECRET_KEY", "local-development-only-change-me")
    environment: str = os.getenv("NIDS_ENV", "development").lower()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "production_model.joblib"

    @property
    def registry_path(self) -> Path:
        return self.artifact_dir / "registry.json"

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "metrics.json"


settings = Settings()
