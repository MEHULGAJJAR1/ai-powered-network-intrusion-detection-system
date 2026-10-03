from __future__ import annotations

import numpy as np

from nids.ml.preprocessing import QuantileClipper, build_preprocessor
from nids.ml.training import _feature_groups, make_group_aware_splits
from nids.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from tests.conftest import make_training_frame, make_record


def test_preprocessor_handles_missing_unknown_categories_and_order():
    X, _ = make_training_frame(rows_per_class=8)
    X.loc[0, "duration"] = np.nan
    X.loc[1, "service"] = None
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
    assert len(preprocessor.get_feature_names_out()) == transformed.shape[1]

    changed = make_record(999)
    changed["protocol_type"] = "future-protocol"
    changed["service"] = "unknown-service"
    changed["flag"] = "future-flag"
    changed["duration"] = None
    new_matrix = preprocessor.transform(__import__("pandas").DataFrame([changed], columns=FEATURE_COLUMNS))
    assert new_matrix.shape[1] == transformed.shape[1]
    dense = new_matrix.toarray() if hasattr(new_matrix, "toarray") else np.asarray(new_matrix)
    assert np.isfinite(dense).all()


def test_quantile_clipper_fits_limits_and_clips_only_at_transform():
    clipper = QuantileClipper(lower_quantile=0.1, upper_quantile=0.9)
    train = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    clipper.fit(train)
    transformed = clipper.transform(np.array([[-500.0], [500.0]]))
    assert transformed[0, 0] == clipper.lower_bounds_[0]
    assert transformed[1, 0] == clipper.upper_bounds_[0]
    assert clipper.upper_bounds_[0] < 100.0


def test_group_aware_split_keeps_duplicate_features_together_and_all_classes():
    X, y = make_training_frame(rows_per_class=15)
    # Add an exact feature duplicate; it must be assigned to the same partition.
    X = __import__("pandas").concat([X, X.iloc[[0]]], ignore_index=True)
    y = np.concatenate([y, y[[0]]])
    train, validation, test, groups = make_group_aware_splits(X, y, seed=23)
    group_sets = [set(groups[index]) for index in (train, validation, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    for indices in (train, validation, test):
        assert set(np.unique(y[indices])) == set(range(5))
