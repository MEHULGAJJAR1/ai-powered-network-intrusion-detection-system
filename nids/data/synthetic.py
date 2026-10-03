"""Synthetic-only UI monitor input generator; never used by the training pipeline."""
from __future__ import annotations

import threading

import numpy as np

from nids.schema import FEATURE_COLUMNS


class SyntheticRecordFactory:
    """Generate plausible-shaped records for UI walkthroughs, without ground truth."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def next_record(self) -> dict[str, object]:
        with self._lock:
            rng = self._rng
            mode = int(rng.integers(0, 5))
            protocol = str(rng.choice(["tcp", "udp", "icmp"], p=[0.72, 0.18, 0.10]))
            services = ["http", "private", "domain_u", "smtp", "ftp_data", "eco_i", "other", "telnet", "ssh"]
            service = str(rng.choice(services))
            flag = str(rng.choice(["SF", "S0", "REJ", "RSTR", "RSTO", "S1"], p=[0.69, 0.12, 0.08, 0.04, 0.04, 0.03]))
            row: dict[str, object] = {name: 0 for name in FEATURE_COLUMNS}
            row.update({
                "duration": round(float(rng.exponential(4.0)), 3),
                "protocol_type": protocol,
                "service": service,
                "flag": flag,
                "src_bytes": int(min(rng.lognormal(7.0, 1.25), 1_000_000)),
                "dst_bytes": int(min(rng.lognormal(6.5, 1.5), 1_000_000)),
                "land": int(rng.random() < 0.002),
                "wrong_fragment": int(rng.poisson(0.04)),
                "urgent": int(rng.random() < 0.001),
                "hot": int(rng.poisson(0.15)),
                "num_failed_logins": int(rng.poisson(0.1)),
                "logged_in": int(rng.random() < 0.77),
                "num_compromised": int(rng.poisson(0.08)),
                "root_shell": int(rng.random() < 0.002),
                "su_attempted": int(rng.choice([0, 1, 2], p=[0.99, 0.009, 0.001])),
                "num_root": int(rng.poisson(0.05)),
                "num_file_creations": int(rng.poisson(0.07)),
                "num_shells": int(rng.random() < 0.003),
                "num_access_files": int(rng.poisson(0.08)),
                "num_outbound_cmds": 0,
                "is_host_login": int(rng.random() < 0.002),
                "is_guest_login": int(rng.random() < 0.015),
                "count": int(rng.integers(1, 60)),
                "srv_count": int(rng.integers(1, 55)),
            })
            for name in (
                "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
            ):
                row[name] = round(float(rng.beta(1.1, 5.0)), 3)
            row.update({
                "dst_host_count": int(rng.integers(1, 256)),
                "dst_host_srv_count": int(rng.integers(1, 256)),
            })
            # These are feature-pattern variations only. No synthetic class label or
            # predicted outcome is assigned here; the loaded model makes the prediction.
            if mode == 1:  # burst-like transport telemetry
                row.update({"count": int(rng.integers(80, 256)), "serror_rate": 0.9,
                            "srv_serror_rate": 0.85, "dst_host_serror_rate": 0.8})
            elif mode == 2:  # varied-service scan-like telemetry
                row.update({"count": int(rng.integers(30, 150)), "srv_count": int(rng.integers(1, 12)),
                            "diff_srv_rate": 0.75, "srv_diff_host_rate": 0.7, "src_bytes": int(rng.integers(0, 500))})
            elif mode == 3:  # authentication-heavy telemetry
                row.update({"service": str(rng.choice(["ftp", "telnet", "ssh", "imap"])),
                            "num_failed_logins": int(rng.integers(1, 8)), "is_guest_login": 1,
                            "logged_in": 0})
            elif mode == 4:  # privilege / shell-related telemetry
                row.update({"num_compromised": int(rng.integers(1, 8)), "num_root": int(rng.integers(1, 10)),
                            "root_shell": 1, "num_shells": 1})
            # Canonical ordering is useful for CSV upload demos and deterministic validation.
            return {name: row[name] for name in FEATURE_COLUMNS}
