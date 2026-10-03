"""SQLite-backed, bounded prediction and alert history for a single-node deployment."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from nids.schema import CLASS_ORDER


class HistoryStore:
    """Thread-safe-by-connection history store; use a managed SQL service at scale."""

    def __init__(self, path: str | Path, max_rows: int = 50_000):
        self.path = str(path)
        self.max_rows = max(100, int(max_rows))
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        if self.path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model_version TEXT,
                    probabilities_json TEXT NOT NULL,
                    explanation_json TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_type_risk ON predictions(prediction, risk_level)"
            )

    def insert(self, item: dict[str, Any], record: dict[str, Any]) -> None:
        self.insert_many([(item, record)])

    def insert_many(self, items: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        if not items:
            return
        values = []
        for item, record in items:
            values.append((
                item["id"], item["timestamp"], item["prediction"], float(item["confidence"]),
                item["risk_level"], item.get("source", "api"), item.get("model_version"),
                json.dumps(item.get("probabilities", {}), ensure_ascii=False),
                json.dumps(item.get("explanation", {}), ensure_ascii=False),
                json.dumps(record, ensure_ascii=False, allow_nan=False),
            ))
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO predictions
                (id, timestamp, prediction, confidence, risk_level, source, model_version,
                 probabilities_json, explanation_json, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
            connection.execute(
                """DELETE FROM predictions WHERE id IN (
                    SELECT id FROM predictions ORDER BY timestamp DESC LIMIT -1 OFFSET ?
                )""",
                (self.max_rows,),
            )

    @staticmethod
    def _decode(row: sqlite3.Row, include_record: bool = False) -> dict[str, Any]:
        result = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "prediction": row["prediction"],
            "confidence": float(row["confidence"]),
            "risk_level": row["risk_level"],
            "source": row["source"],
            "model_version": row["model_version"],
            "probabilities": json.loads(row["probabilities_json"]),
            "explanation": json.loads(row["explanation_json"]),
        }
        if include_record:
            result["record"] = json.loads(row["record_json"])
        return result

    def query(
        self,
        *,
        page: int = 1,
        per_page: int = 25,
        attack_type: str | None = None,
        severity: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        since: str | None = None,
        until: str | None = None,
        search: str | None = None,
        alerts_only: bool = False,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        include_record: bool = False,
    ) -> dict[str, Any]:
        where: list[str] = []
        parameters: list[Any] = []
        if attack_type:
            where.append("prediction = ?")
            parameters.append(attack_type)
        if severity:
            where.append("risk_level = ?")
            parameters.append(severity)
        if min_confidence is not None:
            where.append("confidence >= ?")
            parameters.append(min_confidence)
        if max_confidence is not None:
            where.append("confidence <= ?")
            parameters.append(max_confidence)
        if since:
            where.append("timestamp >= ?")
            parameters.append(since)
        if until:
            where.append("timestamp <= ?")
            parameters.append(until)
        if search:
            where.append("(id LIKE ? OR prediction LIKE ? OR source LIKE ?)")
            match = f"%{search[:100]}%"
            parameters.extend([match, match, match])
        if alerts_only:
            where.append("prediction != 'Normal'")
        clause = " WHERE " + " AND ".join(where) if where else ""
        order_columns = {"timestamp": "timestamp", "confidence": "confidence", "prediction": "prediction", "risk_level": "risk_level"}
        order_column = order_columns.get(sort_by, "timestamp")
        order = "ASC" if sort_order.lower() == "asc" else "DESC"
        safe_page = max(1, int(page))
        safe_size = min(100, max(1, int(per_page)))
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM predictions" + clause, parameters).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM predictions{clause} ORDER BY {order_column} {order}, id {order} LIMIT ? OFFSET ?",
                [*parameters, safe_size, (safe_page - 1) * safe_size],
            ).fetchall()
        return {
            "items": [self._decode(row, include_record=include_record) for row in rows],
            "total": total,
            "page": safe_page,
            "per_page": safe_size,
            "pages": (total + safe_size - 1) // safe_size,
        }

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            normal = int(connection.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'").fetchone()[0])
            attack = total - normal
            high_risk = int(connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE risk_level IN ('critical', 'high')"
            ).fetchone()[0])
            type_rows = connection.execute(
                "SELECT prediction, COUNT(*) AS amount FROM predictions GROUP BY prediction"
            ).fetchall()
            risk_rows = connection.execute(
                "SELECT risk_level, COUNT(*) AS amount FROM predictions GROUP BY risk_level"
            ).fetchall()
            confidence_rows = connection.execute(
                "SELECT confidence FROM predictions ORDER BY timestamp DESC LIMIT 10000"
            ).fetchall()
            latest = connection.execute(
                "SELECT timestamp FROM predictions ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        attack_distribution = {label: 0 for label in CLASS_ORDER}
        attack_distribution.update({row["prediction"]: int(row["amount"]) for row in type_rows})
        risk_distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        risk_distribution.update({row["risk_level"]: int(row["amount"]) for row in risk_rows})
        confidence_bins = {"0–20%": 0, "20–40%": 0, "40–60%": 0, "60–80%": 0, "80–100%": 0}
        for row in confidence_rows:
            value = float(row["confidence"])
            index = min(4, int(value * 5))
            key = list(confidence_bins)[index]
            confidence_bins[key] += 1
        return {
            "total_traffic_analyzed": total,
            "normal_traffic": normal,
            "detected_attacks": attack,
            "attack_percentage": round(100 * attack / total, 3) if total else 0.0,
            "high_risk_events": high_risk,
            "attack_distribution": attack_distribution,
            "risk_distribution": risk_distribution,
            "confidence_distribution_recent_10000": confidence_bins,
            "latest_prediction_at": latest["timestamp"] if latest else None,
            "history_retention_limit": self.max_rows,
        }

    def export_rows(self, *, alerts_only: bool = False) -> list[dict[str, Any]]:
        clause = " WHERE prediction != 'Normal'" if alerts_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM predictions{clause} ORDER BY timestamp DESC"
            ).fetchall()
        return [self._decode(row, include_record=False) for row in rows]
