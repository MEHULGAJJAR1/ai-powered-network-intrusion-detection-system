#!/usr/bin/env python3
"""Download authentic KDD Cup 1999 10% data and validate the gzip/record shape."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

OFFICIAL_URL = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
# Pinned public mirror used only when UCI blocks automated downloads (for example, HTTP 403).
# Operators requiring official-source provenance can pass --url and verify the printed SHA-256.
PINNED_MIRROR_URL = (
    "https://raw.githubusercontent.com/baonq-me/kdd-cup-1999/"
    "3158b52e174f051ff69a4f0758c4426544fdf245/kddcup.data_10_percent.gz"
)


def _download(url: str, temporary: Path, timeout: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "NIDS-dataset-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as target:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count == 0:
        raise RuntimeError("The server returned an empty dataset file")
    return byte_count, digest.hexdigest()


def _validate_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as stream:
            row = next(csv.reader(stream))
    except Exception as exc:
        raise RuntimeError("Downloaded file is not a readable gzip-compressed CSV") from exc
    if len(row) not in (42, 43):
        raise RuntimeError(f"Downloaded first record has {len(row)} columns; expected 42 or 43")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/kddcup.data_10_percent.gz"))
    parser.add_argument("--url", default=None, help="Explicit single source URL; default tries UCI then a pinned public mirror")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    sources = [args.url] if args.url else [OFFICIAL_URL, PINNED_MIRROR_URL]
    last_error = None
    for url in sources:
        try:
            byte_count, checksum = _download(url, temporary, args.timeout)
            _validate_gzip(temporary)
            os.replace(temporary, output)
            print(f"Downloaded {byte_count:,} bytes to {output}")
            print(f"SHA-256 (computed locally): {checksum}")
            print(f"Source: {url}")
            if url != OFFICIAL_URL:
                print("Note: a pinned public mirror was used because the official UCI file endpoint was unavailable.")
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            print(f"Source unavailable/invalid ({url}): {exc}", file=sys.stderr)
    print(f"All download sources failed: {last_error}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
