from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nids.ml.inference import InferenceEngine
from nids.ml.preprocessing import build_pipeline
from nids.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from nids.services.history import HistoryStore


def make_record(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    record = {}
    for name in FEATURE_COLUMNS:
        if name in CATEGORICAL_FEATURES:
            record[name] = {
                "protocol_type": ["tcp", "udp", "icmp"][seed % 3],
                "service": ["http", "private", "smtp"][seed % 3],
                "flag": ["sf", "s0", "rej"][seed % 3],
            }[name]
        elif name.endswith("_rate"):
            record[name] = float(rng.uniform(0.0, 1.0))
        elif name in {"land", "logged_in", "root_shell", "is_host_login", "is_guest_login"}:
            record[name] = int(rng.integers(0, 2))
        elif name == "su_attempted":
            record[name] = int(rng.integers(0, 3))
        else:
            record[name] = float(rng.integers(0, 100))
    record["src_bytes"] = float(seed * 100 + 1)
    return record


def make_training_frame(rows_per_class: int = 12):
    rows = []
    labels = []
    for class_id in range(5):
        for index in range(rows_per_class):
            row = make_record(seed=class_id * 1000 + index + 1)
            # Distinct, class-correlated feature patterns make the fixture deterministic.
            row["src_bytes"] = float(class_id * 10_000 + index * 31 + 2)
            row["dst_bytes"] = float((4 - class_id) * 1000 + index * 7)
            row["protocol_type"] = ["tcp", "udp", "icmp"][class_id % 3]
            row["service"] = ["http", "private", "smtp", "ftp", "telnet"][class_id]
            row["flag"] = ["sf", "s0", "rej", "rsto", "sh"][class_id]
            rows.append(row)
            labels.append(class_id)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), np.asarray(labels, dtype=int)


@pytest.fixture()
def trained_engine():
    frame, labels = make_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=300, class_weight="balanced", random_state=42))
    pipeline.fit(frame, labels)
    return InferenceEngine(pipeline, {"active_model": "test_logistic", "version": "test-1"})


@pytest.fixture()
def api_client(tmp_path: Path, trained_engine):
    from nids.api.app import create_app

    history = HistoryStore(tmp_path / "history.sqlite3", max_rows=500)
    app = create_app({
        "TESTING": True,
        "INFERENCE_ENGINE": trained_engine,
        "HISTORY_STORE": history,
        "NIDS_HISTORY_DB": str(tmp_path / "history.sqlite3"),
        "NIDS_DATA_PATH": str(tmp_path / "missing-kdd.csv"),
        "NIDS_MAX_JSON_BATCH_ROWS": 5,
        "NIDS_MAX_BATCH_ROWS": 20,
        "MAX_CONTENT_LENGTH": 1024 * 1024,
    })
    return app.test_client()
