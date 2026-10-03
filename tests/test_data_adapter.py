from __future__ import annotations

import csv

import pytest

from nids.data.adapter import read_kdd_dataset
from nids.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def _row(label: str, offset: int = 0):
    row = []
    for name in FEATURE_COLUMNS:
        if name == "protocol_type":
            row.append("tcp")
        elif name == "service":
            row.append("http")
        elif name == "flag":
            row.append("SF")
        else:
            row.append(str(offset + 1))
    return row + [label]


def test_maps_original_kdd_attack_families_and_deduplicates(tmp_path):
    path = tmp_path / "kdd.csv"
    rows = [
        _row("normal.", 0), _row("neptune.", 1), _row("satan.", 2),
        _row("guess_passwd.", 3), _row("buffer_overflow.", 4), _row("normal.", 0),
    ]
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    X, y, report = read_kdd_dataset(path)
    assert len(X) == 5
    assert y.tolist() == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert report.rows_loaded == 6
    assert report.duplicate_rows_removed == 1
    assert report.rows_after_deduplication == 5
    assert list(X.columns) == list(FEATURE_COLUMNS)


def test_headered_file_and_trailing_difficulty_are_adapted(tmp_path):
    path = tmp_path / "headered.csv"
    headers = [*FEATURE_COLUMNS, "class", "difficulty"]
    row = _row("normal.", 5) + ["9"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerow(row)
    X, y, report = read_kdd_dataset(path)
    assert len(X) == 1
    assert y.iloc[0] == "Normal"
    assert report.dropped_difficulty_column is True
    assert not any(name in X.columns for name in ("difficulty", "class", "label"))


def test_unknown_label_is_rejected_instead_of_silently_relabelled(tmp_path):
    path = tmp_path / "unknown.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerow(_row("unexpected_attack", 1))
    with pytest.raises(ValueError, match="Unsupported label"):
        read_kdd_dataset(path)
