
# -*- coding: utf-8 -*-


import numpy as np

from xgboost import XGBClassifier
from hyperopt import hp
from hyperopt.pyll import scope

from base import get_full_name, RANDOM_STATE
from models import AbstractModel


###############################################################################
def get_xgboost_model(name="xgboost_common"):
    return scope.get_xgb_model(
        n_estimators=scope.int(
            hp.quniform(
                get_full_name(name, "n_estimators"),
                1, 200, 1,
            ),
        ),
        max_depth=scope.int(
            hp.quniform(
                get_full_name(name, 'max_depth'),
                1, 13, 1,
            ),
        ),
        min_child_weight=scope.int(
            hp.quniform(
                get_full_name(name, 'min_child_weight'),
                1, 6, 1,
            ),
        ),
        subsample=scope.int(
            hp.uniform(
                get_full_name(name, 'subsample'),
                0.5, 1,
            ),
        ),
        gamma=hp.uniform(
            get_full_name(name, 'gamma'),
            0.5, 1,
        ),
        nthread=1,
        seed=RANDOM_STATE,
    )


@scope.define
def get_xgb_model_wrapper(*args, **kwargs):
    xgb = XGBoostWrapper(current_layer=XGBClassifier(*args, **kwargs))

    return xgb


class XGBoostWrapper(AbstractModel):
    """Wrapper for scikit-learn xgb classifier wrapper."""

    def fit(self, X, y):
        self._features_count = X.shape[1]
        # create (n,1)-vector of zeros if X is empty
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1))
        # fit XGBClassifier
        self.current_layer.fit(X, y)
        # get weights from fitted scikit-learn model
        self.feature_importances_ = self.current_layer.feature_importances_

        return self

    def predict(self, X):
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1))

        return self.current_layer.predict(X)

    def transform(self, X):
        """Transform dataset X by deleting columns that correspond to less important features."""
        weights = np.asarray(self.feature_importances_)
        indexes = np.where(np.abs(weights) > 0.0)[0]

        X_tr = X[:, indexes]

        return X_tr

    def get_hyperparams(self, deep=True):
        return self.get_params(deep)

    def get_feature_importances(self):
        # model must be fitted first to return something
        return self.feature_importances_

    def get_support(self, indices=True):
        if indices:
            return np.arange(self._features_count)
        else:
            return np.ones(self._features_count, dtype=np.bool)

