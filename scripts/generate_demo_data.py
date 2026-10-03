#!/usr/bin/env python3
"""Generate synthetic, unlabeled CSV records for dashboard/upload walkthroughs only."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nids.data.synthetic import SyntheticRecordFactory
from nids.schema import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/demo/synthetic_traffic.csv"))
    args = parser.parse_args()
    if not 1 <= args.rows <= 5000:
        raise SystemExit("--rows must be between 1 and 5000")
    factory = SyntheticRecordFactory(args.seed)
    rows = [factory.next_record() for _ in range(args.rows)]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_csv(output, index=False)
    print(f"Wrote {args.rows} synthetic, unlabeled UI/demo records to {output}")
    print("This file is not authentic KDD training data and must not be used to train models.")


if __name__ == "__main__":
    main()
