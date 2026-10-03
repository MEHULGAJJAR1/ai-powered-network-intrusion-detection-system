from __future__ import annotations

import csv

from nids.data.exploration import build_dataset_summary
from nids.schema import FEATURE_COLUMNS


def _row(label: str, offset: int):
    values = []
    for name in FEATURE_COLUMNS:
        if name == "protocol_type":
            values.append("TCP")
        elif name == "service":
            values.append("http")
        elif name == "flag":
            values.append("SF")
        else:
            values.append(str(offset + 1))
    return values + [label]


def test_streamed_exploration_uses_real_file_counts_and_uniform_sample(tmp_path):
    path = tmp_path / "tiny.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*FEATURE_COLUMNS, "class"])
        writer.writerows([
            _row("normal.", 0), _row("neptune.", 1), _row("satan.", 2),
            _row("normal.", 0),
        ])
    summary = build_dataset_summary(path, max_rows=2)
    assert summary["available"] is True
    assert summary["dimensions"]["rows"] == 4
    assert summary["sampled_rows_for_statistics"] == 2
    assert summary["duplicate_rows_detected"] == 1
    assert summary["class_distribution"]["Normal"] == 2
    assert summary["class_distribution"]["DoS"] == 1
    assert summary["class_distribution"]["Probe"] == 1
    assert len(summary["numeric_statistics"]) == 38
