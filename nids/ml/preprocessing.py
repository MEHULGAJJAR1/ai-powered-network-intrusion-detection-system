"""Leakage-safe, serializable preprocessing for canonical KDD records."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from nids.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers to quantiles learned only from the current fit split.

    NaNs are intentionally passed through so the following imputer can handle them.
    The fitted limits serialize with the sklearn pipeline and are not recomputed at
    inference time.
    """

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):  # noqa: D401 - sklearn estimator protocol
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("QuantileClipper expects a two-dimensional feature matrix")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Quantile bounds must satisfy 0 <= lower < upper <= 1")
        lower = np.empty(values.shape[1], dtype=np.float64)
        upper = np.empty(values.shape[1], dtype=np.float64)
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            if finite.size:
                lower[index], upper[index] = np.quantile(
                    finite, [self.lower_quantile, self.upper_quantile]
                )
            else:
                lower[index], upper[index] = -np.inf, np.inf
        self.lower_bounds_ = lower
        self.upper_bounds_ = upper
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("QuantileClipper received an unexpected number of features")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def _one_hot_encoder() -> OneHotEncoder:
    # Keep compatibility with scikit-learn versions before sparse_output was renamed.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:  # pragma: no cover - exercised only on older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def _imputer(strategy: str, *, add_indicator: bool = False) -> SimpleImputer:
    try:
        return SimpleImputer(strategy=strategy, add_indicator=add_indicator, keep_empty_features=True)
    except TypeError:  # pragma: no cover - legacy sklearn fallback
        return SimpleImputer(strategy=strategy, add_indicator=add_indicator)


def build_preprocessor() -> ColumnTransformer:
    """Create the single canonical transformer used in CV, training, and serving."""
    numeric = Pipeline(
        steps=[
            ("outlier_clip", QuantileClipper()),
            ("imputer", _imputer("median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", _imputer("most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the canonical preprocessing steps."""
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
    )
