"""Memory-bounded, real-dataset exploration summaries cached by file modification time."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nids.data.adapter import DatasetLoadReport, _header_aliases, _header_candidate, normalize_feature_frame
from nids.schema import ATTACK_TO_CATEGORY, CATEGORICAL_FEATURES, CLASS_ORDER, FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURES

_CACHE: dict[tuple[str, int, int, int], dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _make_layout(first_chunk: pd.DataFrame) -> tuple[dict[str, int], int, int | None, bool]:
    """Map a headered or headerless KDD chunk to original CSV column positions."""
    first_row = first_chunk.iloc[0]
    header = _header_candidate(first_row)
    width = first_chunk.shape[1]
    if header:
        raw_names = _header_aliases(header)
        difficulty_index = raw_names.index("difficulty") if "difficulty" in raw_names else None
        name_indices = {name: index for index, name in enumerate(raw_names) if name != "difficulty"}
        if len(name_indices) != 42 or LABEL_COLUMN not in name_indices or not set(FEATURE_COLUMNS).issubset(name_indices):
            raise ValueError("Header does not match the canonical KDD feature/label schema")
        return name_indices, width, difficulty_index, True
    if width == 43:
        label_values = first_chunk.iloc[:, 41].astype(str).str.strip().str.lower().str.rstrip(".")
        if label_values.isin(ATTACK_TO_CATEGORY).mean() < 0.95:
            raise ValueError("43-column file does not have recognized KDD labels before its trailing difficulty field")
        return {**{name: index for index, name in enumerate(FEATURE_COLUMNS)}, LABEL_COLUMN: 41}, 43, 42, False
    if width == 42:
        return {**{name: index for index, name in enumerate(FEATURE_COLUMNS)}, LABEL_COLUMN: 41}, 42, None, False
    raise ValueError(f"Expected 42 KDD columns or 43 including difficulty; found {width}")


def _reservoir_update(reservoir: pd.DataFrame, chunk: pd.DataFrame, seen_before: int,
                      sample_limit: int, rng: np.random.Generator) -> pd.DataFrame:
    """Merge a uniform sample of a new chunk with an existing uniform reservoir."""
    if chunk.empty:
        return reservoir
    if reservoir.empty:
        if len(chunk) <= sample_limit:
            return chunk.reset_index(drop=True).copy()
        chosen = rng.choice(len(chunk), size=sample_limit, replace=False)
        return chunk.iloc[chosen].reset_index(drop=True)
    total_after = seen_before + len(chunk)
    if total_after <= sample_limit:
        return pd.concat([reservoir, chunk], ignore_index=True)
    take_from_new = int(rng.hypergeometric(len(chunk), seen_before, sample_limit))
    keep_from_old = sample_limit - take_from_new
    old_sample = reservoir.iloc[rng.choice(len(reservoir), size=keep_from_old, replace=False)] if keep_from_old else reservoir.iloc[0:0]
    new_sample = chunk.iloc[rng.choice(len(chunk), size=take_from_new, replace=False)] if take_from_new else chunk.iloc[0:0]
    return pd.concat([old_sample, new_sample], ignore_index=True)


def build_dataset_summary(path: str | Path, *, max_rows: int = 100_000) -> dict[str, Any]:
    """Stream the configured file, retaining a deterministic uniform statistics sample.

    Full-file class counts and normalized duplicate fingerprints are computed in
    chunks. This avoids materializing a 494k-row, 41-feature table independently
    in each dashboard worker. Numeric/categorical summaries use a uniform
    reservoir sample and expose the sample size.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {"available": False, "message": "KDD dataset is not present. Add the authentic dataset and refresh this page."}
    stat = source.stat()
    sample_limit = max(1, int(max_rows))
    cache_key = (str(source), stat.st_mtime_ns, stat.st_size, sample_limit)
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    class_counts = {label: 0 for label in CLASS_ORDER}
    numeric_coercions: dict[str, int] = {}
    seen_fingerprints: set[int] = set()
    duplicate_count = 0
    total_rows = 0
    dropped_difficulty = False
    reservoir = pd.DataFrame(columns=FEATURE_COLUMNS)
    rng = np.random.default_rng(42)
    reader = pd.read_csv(source, header=None, compression="infer", chunksize=20_000, low_memory=False)
    layout: dict[str, int] | None = None
    expected_width = 0
    header_pending = False
    try:
        for raw_chunk in reader:
            if layout is None:
                layout, expected_width, difficulty_index, header_pending = _make_layout(raw_chunk)
                dropped_difficulty = difficulty_index is not None
            if raw_chunk.shape[1] != expected_width:
                raise ValueError(f"Dataset row chunk has {raw_chunk.shape[1]} columns; expected {expected_width}.")
            if header_pending:
                raw_chunk = raw_chunk.iloc[1:].copy()
                header_pending = False
            if raw_chunk.empty:
                continue
            feature_indices = [layout[name] for name in FEATURE_COLUMNS]
            feature_input = raw_chunk.iloc[:, feature_indices].copy()
            feature_input.columns = list(FEATURE_COLUMNS)
            features, coercions = normalize_feature_frame(feature_input)
            label_index = layout[LABEL_COLUMN]
            tokens = raw_chunk.iloc[:, label_index].astype(str).str.strip().str.lower().str.rstrip(".").str.strip()
            unknown = sorted(set(tokens) - set(ATTACK_TO_CATEGORY))
            if unknown:
                raise ValueError(f"Unsupported KDD label(s): {', '.join(unknown[:12])}")
            labels = tokens.map(ATTACK_TO_CATEGORY)
            counts = labels.value_counts()
            for label, count in counts.items():
                class_counts[label] += int(count)
            for name, count in coercions.items():
                numeric_coercions[name] = numeric_coercions.get(name, 0) + int(count)

            fingerprint_frame = features.copy()
            fingerprint_frame["__target__"] = labels.to_numpy()
            hashes = pd.util.hash_pandas_object(fingerprint_frame, index=False).to_numpy(dtype=np.uint64)
            for value in hashes:
                key = int(value)
                if key in seen_fingerprints:
                    duplicate_count += 1
                else:
                    seen_fingerprints.add(key)

            reservoir = _reservoir_update(reservoir, features, total_rows, sample_limit, rng)
            total_rows += len(features)
    except pd.errors.EmptyDataError:
        return {"available": False, "message": "Configured KDD file is empty."}
    finally:
        reader.close()

    if layout is None or total_rows == 0:
        return {"available": False, "message": "Configured KDD file contains no traffic rows."}
    numeric = reservoir.loc[:, list(NUMERIC_FEATURES)]
    stats_frame = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    numeric_stats: dict[str, dict[str, Any]] = {}
    for name, row in stats_frame.iterrows():
        numeric_stats[name] = {
            key_name: (None if pd.isna(row[key_name]) else round(float(row[key_name]), 6))
            for key_name in ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
        }
    categorical: dict[str, list[dict[str, Any]]] = {}
    for name in CATEGORICAL_FEATURES:
        counts = reservoir[name].fillna("<missing>").value_counts().head(12)
        categorical[name] = [{"value": str(value), "count": int(count)} for value, count in counts.items()]

    variance = numeric.var(numeric_only=True).sort_values(ascending=False)
    correlation_features = list(variance.head(12).index)
    corr = numeric[correlation_features].corr().replace([np.inf, -np.inf], np.nan)
    correlation = {
        "features": correlation_features,
        "values": [[None if pd.isna(value) else round(float(value), 5) for value in row] for row in corr.to_numpy()],
    }
    report = DatasetLoadReport(
        source=str(source), rows_loaded=total_rows, duplicate_rows_removed=0,
        rows_after_deduplication=total_rows,
        numeric_coercions={key: value for key, value in numeric_coercions.items() if value},
        class_distribution=class_counts, dropped_difficulty_column=dropped_difficulty,
    )
    result = {
        "available": True,
        "dataset_name": "KDD Cup 1999",
        "source_file": source.name,
        "dimensions": {"rows": int(total_rows), "features": len(FEATURE_COLUMNS), "columns_including_label": len(FEATURE_COLUMNS) + 1},
        "sampled_rows_for_statistics": int(len(reservoir)),
        "duplicate_rows_detected": duplicate_count,
        "class_distribution": class_counts,
        "numeric_statistics": numeric_stats,
        "categorical_distributions": categorical,
        "correlation": correlation,
        "adapter_report": report.to_dict(),
        "note": "Class counts and normalized duplicate fingerprints are calculated across the full file. Numerical, categorical and correlation summaries use a fixed-seed uniform reservoir sample.",
    }
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE[cache_key] = result
    return result
