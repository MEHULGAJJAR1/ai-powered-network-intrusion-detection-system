#!/usr/bin/env python3
"""Render a concise measured-results appendix from the generated model artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(value):
    return "not reported" if value is None else f"{float(value) * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.artifacts.expanduser().resolve()
    metrics_path = root / "metrics.json"
    registry_path = root / "registry.json"
    if not metrics_path.is_file() or not registry_path.is_file():
        raise SystemExit("Measured artifacts are missing. Run train.py on authentic KDD data first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve() if args.output else root / "evaluation" / "PROJECT_REPORT_RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Measured Results Appendix",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This appendix is rendered from this training run's `metrics.json` and `registry.json`; it contains no hand-entered or synthetic performance values.",
        "",
        f"- Dataset: {registry.get('dataset', {}).get('name', 'not recorded')}",
        f"- SHA-256: `{registry.get('dataset', {}).get('sha256', 'not recorded')}`",
        f"- Model version: `{registry.get('version', 'not recorded')}`",
        f"- Validation-selected model: `{metrics.get('selected_model', 'not recorded')}`",
        f"- Selection criterion: `{metrics.get('selection_criterion', 'not recorded')}`",
        "",
        "## Held-out test comparison",
        "",
        "| Candidate | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC OVR (macro) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics.get("models", {}).items():
        overall = item.get("test", {}).get("overall", {})
        auc = pct(overall.get("roc_auc_ovr_macro"))
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, pct(overall.get("accuracy")), pct(overall.get("precision_macro")),
            pct(overall.get("recall_macro")), pct(overall.get("f1_macro")),
            pct(overall.get("f1_weighted")), auc,
        ))
    selected = metrics.get("models", {}).get(metrics.get("selected_model"), {}).get("test", {})
    lines += ["", "## Selected model per-class results", "", "| Class | Support | Precision | Recall | F1 | False-positive rate |", "|---|---:|---:|---:|---:|---:|"]
    for label, item in selected.get("per_class", {}).items():
        lines.append(f"| {label} | {item.get('support', 0)} | {pct(item.get('precision'))} | {pct(item.get('recall'))} | {pct(item.get('f1'))} | {pct(item.get('false_positive_rate'))} |")
    lines += [
        "", "## Evaluation record", "",
        f"- Test rows: {selected.get('overall', {}).get('samples', 'not recorded')}",
        f"- Split record: `{json.dumps(metrics.get('split', {}), sort_keys=True)}`",
        "- Confusion matrix and candidate search parameters are retained in `metrics.json` and `registry.json`.",
        "- Interpret these results in light of KDD's age, class imbalance, exact-duplicate removal, and the group's feature-fingerprint split policy.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote measured-results appendix: {output}")


if __name__ == "__main__":
    main()
