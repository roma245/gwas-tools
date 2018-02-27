
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from hyperopt.pyll import scope

from models import AbstractModel


###############################################################################
def get_rf_model(name='rf'):
    """Set up hyperparameters and ranges for random forest model."""
    return scope.get_rf_model(n_estimators=100, n_jobs=1)


@scope.define
def get_rf_model(*args, **kwargs):
    """Get random forest model initialized with parameters from the calling function."""
    rf = RFWrapper(current_layer=RandomForestClassifier(*args, **kwargs))

    return rf


class RFWrapper(AbstractModel):
    """Wrapper for scikit-learn random forest model."""

    def fit(self, X, y):
        self._features_count = X.shape[1]
        # create (n,1)-vector of zeros if X is empty
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1), dtype=X.dtype)
        # fit scikit-lear random forest
        self.current_layer.fit(X, y)

        self.feature_importances_ = self.current_layer.feature_importances_

        return self

    def predict(self, X):
        # create (n,1)-vector of zeros if X is empty
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 1), dtype=X.dtype)

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
        return self.feature_importances_

    def get_support(self, indices=True):
        if indices:
            return np.arange(self._features_count)
        else:
            return np.ones(self._features_count, dtype=np.bool)




