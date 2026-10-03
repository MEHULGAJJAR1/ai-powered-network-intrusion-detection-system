"""KDD Cup 1999 adapter with explicit schema checks and label normalization."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nids.schema import (
    ATTACK_TO_CATEGORY,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)
LABEL_ALIASES = {"label", "class", "attack", "attack_type", "target", "type"}
DIFFICULTY_ALIASES = {"difficulty", "difficulty_level", "difficulty level"}


@dataclass
class DatasetLoadReport:
    source: str
    rows_loaded: int
    duplicate_rows_removed: int
    rows_after_deduplication: int
    numeric_coercions: dict[str, int]
    class_distribution: dict[str, int]
    dropped_difficulty_column: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_header(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _header_candidate(row: pd.Series) -> list[str] | None:
    values = [_normalized_header(value) for value in row.tolist()]
    canonical = set(FEATURE_COLUMNS)
    if len(values) not in (42, 43):
        return None
    label_positions = [i for i, value in enumerate(values) if value in LABEL_ALIASES]
    if len(label_positions) != 1:
        return None
    label_position = label_positions[0]
    feature_names = [value for i, value in enumerate(values) if i != label_position]
    if len(feature_names) == 42 and feature_names[-1] in DIFFICULTY_ALIASES:
        feature_names = feature_names[:-1]
    if len(feature_names) == 41 and set(feature_names) == canonical:
        return values
    return None


def _header_aliases(values: list[str]) -> list[str]:
    mapped = []
    for value in values:
        if value in LABEL_ALIASES:
            mapped.append(LABEL_COLUMN)
        elif value in DIFFICULTY_ALIASES:
            mapped.append("difficulty")
        else:
            mapped.append(value)
    return mapped


def normalize_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and coerce raw feature columns; missing values are retained for imputation."""
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    extra = [name for name in frame.columns if name not in FEATURE_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing features: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(map(str, extra))}")
        raise ValueError("Invalid feature schema (" + "; ".join(details) + ")")

    clean = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    coercions: dict[str, int] = {}
    for name in NUMERIC_FEATURES:
        before = clean[name].notna()
        converted = pd.to_numeric(clean[name], errors="coerce")
        coercions[name] = int((before & converted.isna()).sum())
        # Infinite values are treated as missing at fit time; inference validates them separately.
        converted = converted.replace([np.inf, -np.inf], np.nan)
        clean[name] = converted.astype("float64")
    for name in ("protocol_type", "service", "flag"):
        clean[name] = clean[name].map(
            lambda value: np.nan if pd.isna(value) or str(value).strip() == "" else str(value).strip().lower()
        )
    return clean, coercions


def read_kdd_dataset(
    path: str | Path,
    *,
    nrows: int | None = None,
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, pd.Series, DatasetLoadReport]:
    """Read raw KDD Cup CSV (optionally gzip-compressed) and map labels to 5 classes.

    Supports the original 41 features + attack label, named CSV headers, and the
    common 43-column variant with a trailing difficulty field. It intentionally
    rejects unknown attack names rather than silently dropping or relabeling them.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"KDD dataset not found at {source}. Download it with `python scripts/download_kdd.py` "
            "or set NIDS_DATA_PATH to the dataset file."
        )
    # Read one extra record so headered files still supply the requested number of data rows.
    raw = pd.read_csv(source, header=None, compression="infer", low_memory=False,
                      nrows=(nrows + 1 if nrows is not None else None))
    if raw.empty:
        raise ValueError(f"Dataset file is empty: {source}")

    dropped_difficulty = False
    header = _header_candidate(raw.iloc[0])
    if header:
        raw = raw.iloc[1:].reset_index(drop=True)
        raw.columns = _header_aliases(header)
        if nrows is not None:
            raw = raw.head(nrows)
        if "difficulty" in raw.columns:
            raw = raw.drop(columns=["difficulty"])
            dropped_difficulty = True
    else:
        if raw.shape[1] == 43:
            # NSL-KDD-style records occasionally carry a trailing difficulty value.
            candidate_labels = raw.iloc[:, 41].astype(str).str.strip().str.lower().str.rstrip(".")
            recognized = candidate_labels.isin(ATTACK_TO_CATEGORY).mean()
            if recognized < 0.95:
                raise ValueError(
                    f"Expected 42 KDD columns or 43 columns with a trailing difficulty field; got {raw.shape[1]} columns."
                )
            raw = raw.iloc[:, :42]
            dropped_difficulty = True
        else:
            dropped_difficulty = False
        if raw.shape[1] != 42:
            raise ValueError(f"Expected 42 KDD columns (41 features + label); got {raw.shape[1]}.")
        raw.columns = [*FEATURE_COLUMNS, LABEL_COLUMN]
        if nrows is not None:
            raw = raw.head(nrows)

    # Header-path can contain a difficulty column; handle it consistently.
    if "difficulty" in raw.columns:
        raw = raw.drop(columns=["difficulty"])
        dropped_difficulty = True
    if LABEL_COLUMN not in raw.columns:
        label_columns = [name for name in raw.columns if _normalized_header(name) in LABEL_ALIASES]
        if len(label_columns) == 1:
            raw = raw.rename(columns={label_columns[0]: LABEL_COLUMN})
    if set(FEATURE_COLUMNS).issubset(raw.columns) and LABEL_COLUMN in raw.columns:
        raw = raw.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]]
    else:
        raise ValueError("Could not adapt the input file to the canonical KDD feature schema.")

    loaded_rows = len(raw)
    original_labels = raw[LABEL_COLUMN].astype(str).str.strip()
    normalized_tokens = original_labels.map(lambda value: value.lower().rstrip(".").strip())
    unknown = sorted(set(normalized_tokens) - set(ATTACK_TO_CATEGORY))
    if unknown:
        preview = ", ".join(repr(value) for value in unknown[:12])
        raise ValueError(f"Unsupported label(s) in KDD data: {preview}. Add a reviewed mapping before training.")
    labels = normalized_tokens.map(ATTACK_TO_CATEGORY)

    features, coercions = normalize_feature_frame(raw.loc[:, list(FEATURE_COLUMNS)])
    normalized = features.copy()
    normalized[LABEL_COLUMN] = labels.to_numpy()
    duplicate_rows = int(normalized.duplicated(keep="first").sum()) if deduplicate else 0
    if deduplicate and duplicate_rows:
        keep = ~normalized.duplicated(keep="first")
        normalized = normalized.loc[keep].reset_index(drop=True)
        labels = normalized[LABEL_COLUMN].copy()
        normalized = normalized.drop(columns=[LABEL_COLUMN])
    else:
        normalized = normalized.drop(columns=[LABEL_COLUMN])
        labels = labels.reset_index(drop=True)

    labels = labels.reset_index(drop=True)
    distribution = labels.value_counts().reindex(
        ["Normal", "DoS", "Probe", "R2L", "U2R"], fill_value=0
    )
    report = DatasetLoadReport(
        source=str(source),
        rows_loaded=loaded_rows,
        duplicate_rows_removed=duplicate_rows,
        rows_after_deduplication=len(normalized),
        numeric_coercions={key: value for key, value in coercions.items() if value},
        class_distribution={key: int(value) for key, value in distribution.items()},
        dropped_difficulty_column=bool(dropped_difficulty),
    )
    logger.info(
        "KDD dataset adapted",
        extra={"event": "dataset_loaded", "rows_loaded": loaded_rows,
               "rows_after_deduplication": len(normalized), "duplicate_rows_removed": duplicate_rows},
    )
    return normalized.reset_index(drop=True), labels, report
