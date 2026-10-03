"""Canonical KDD Cup 1999 feature schema and label mapping."""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
PREPROCESSING_VERSION = "1.0.0"
DATASET_NAME = "KDD Cup 1999 (10% training subset by default)"
CLASS_ORDER = ("Normal", "DoS", "Probe", "R2L", "U2R")
LABEL_TO_ID = {label: index for index, label in enumerate(CLASS_ORDER)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

FEATURE_COLUMNS = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
)
LABEL_COLUMN = "label"
CATEGORICAL_FEATURES = ("protocol_type", "service", "flag")
NUMERIC_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES)
RATE_FEATURES = tuple(c for c in NUMERIC_FEATURES if c.endswith("_rate"))
BINARY_FEATURES = ("land", "logged_in", "root_shell", "is_host_login", "is_guest_login")

# KDD Cup 1999 attack names are mapped into the requested five-class taxonomy.
_ATTACK_FAMILIES = {
    "dos": (
        "apache2", "back", "land", "mailbomb", "neptune", "pod", "smurf", "teardrop",
        "udpstorm", "processtable", "worm",
    ),
    "probe": ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"),
    "r2l": (
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named",
        "phf", "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient",
        "warezmaster", "xlock", "xsnoop",
    ),
    "u2r": (
        "buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm",
    ),
}
ATTACK_TO_CATEGORY: dict[str, str] = {"normal": "Normal"}
_FAMILY_LABELS = {"dos": "DoS", "probe": "Probe", "r2l": "R2L", "u2r": "U2R"}
for _family, _names in _ATTACK_FAMILIES.items():
    ATTACK_TO_CATEGORY.update({name: _FAMILY_LABELS[_family] for name in _names})
ATTACK_TO_CATEGORY.update(_FAMILY_LABELS)

# Short, display-friendly metadata used by the form and API schema endpoint.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "duration": "Connection duration in seconds",
    "protocol_type": "Transport protocol",
    "service": "Destination network service",
    "flag": "Connection status flag",
    "src_bytes": "Bytes sent from source to destination",
    "dst_bytes": "Bytes sent from destination to source",
    "count": "Connections to the same host in the recent window",
    "srv_count": "Connections to the same service in the recent window",
    "logged_in": "Successful login indicator",
    "is_guest_login": "Guest-login indicator",
}


def normalize_attack_label(value: Any) -> str:
    """Map an original KDD label or already-grouped class to a canonical class.

    A trailing period is part of the legacy KDD label format and is ignored.
    Unknown labels raise instead of silently becoming a fabricated class.
    """
    key = str(value).strip().lower().rstrip(".").strip()
    try:
        return ATTACK_TO_CATEGORY[key]
    except KeyError as exc:
        raise ValueError(f"Unrecognized KDD label {value!r}; cannot map it to the five-class taxonomy") from exc


def feature_metadata() -> list[dict[str, Any]]:
    """Return ordered, JSON-safe feature metadata for clients."""
    category_options = {
        "protocol_type": ["tcp", "udp", "icmp"],
        "flag": ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"],
    }
    result = []
    for name in FEATURE_COLUMNS:
        kind = "categorical" if name in CATEGORICAL_FEATURES else "numeric"
        result.append(
            {
                "name": name,
                "type": kind,
                "description": FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ").capitalize()),
                "options": category_options.get(name),
                "nullable": True,
            }
        )
    return result
