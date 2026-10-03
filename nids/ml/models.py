"""Classifier adapters used when an estimator needs fold-local class weighting."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Wrap XGBClassifier and compute balanced sample weights inside every fit.

    Because RandomizedSearchCV clones and fits this adapter independently for
    each fold, class weights are derived only from that fold's training labels.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=int)
        weights = compute_sample_weight(class_weight="balanced", y=y_array)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_array, sample_weight=weights)
        self.classes_ = np.asarray(self.estimator_.classes_)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    def get_booster(self):
        return self.estimator_.get_booster()
