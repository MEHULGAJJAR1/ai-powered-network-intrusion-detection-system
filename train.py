#!/usr/bin/env python3
"""Train/tune/evaluate NIDS models on an authentic KDD Cup 1999 file."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nids.config import settings
from nids.logging_config import configure_logging
from nids.ml.training import train_project


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.data_path,
                        help="KDD Cup 1999 CSV or .gz file (default: NIDS_DATA_PATH)")
    parser.add_argument("--output", type=Path, default=settings.artifact_dir,
                        help="Output artifact directory (default: NIDS_ARTIFACT_DIR)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=100_000,
                        help="Stratified cap after exact-duplicate removal; 0 uses all rows")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Maximum group-stratified CV folds (reduced if rare-class support requires it)")
    parser.add_argument("--n-iter", type=int, default=2,
                        help="RandomizedSearchCV candidates per model; raise for a broader search")
    parser.add_argument("--search-jobs", type=int, default=1,
                        help="Parallel CV jobs; keep low on memory-limited systems")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated subset: logistic_regression,random_forest,xgboost,lightgbm")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(settings.log_level)
    requested = [item.strip() for item in args.models.split(",") if item.strip()] or None
    result = train_project(
        args.data,
        args.output,
        seed=args.seed,
        max_rows=args.max_rows or None,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        search_jobs=args.search_jobs,
        requested_models=requested,
    )
    print(json.dumps({
        "status": "trained",
        "selected_model": result["registry"]["active_model"],
        "model_version": result["registry"]["version"],
        "validation_macro_f1": result["metrics"]["selected_validation_macro_f1"],
        "test_metrics_file": str(args.output.resolve() / "metrics.json"),
        "registry_file": str(args.output.resolve() / "registry.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
