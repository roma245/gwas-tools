
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from sklearn.linear_model import LogisticRegression
from hyperopt import hp
from hyperopt.pyll import scope

from base import get_full_name, RANDOM_STATE
from models import AbstractModel


###############################################################################
def get_linear_model(name="line12ar_common"):
    """Set up hyperparameters and ranges for logistic regression model."""
    return scope.get_lr_model_wrapper(
        C=hp.uniform(get_full_name(name, 'C'), 0, 1),
        penalty=hp.choice(
            get_full_name(name, 'penalty'),
            ('l1', 'l2')
        ),
        class_weight=hp.choice(
            get_full_name(name, 'class_weight'),
            (defaultdict(lambda: 1.0), 'balanced')
        ),
        fit_intercept=hp.choice(
            get_full_name(name, 'fit_intercept'),
            (False, True)
        ),
        random_state=RANDOM_STATE
    )


@scope.define
def get_lr_model_wrapper(*args, **kwargs):
    """Get logistic regression model initialized with parameters from the calling function."""
    lr = LogisticRegressionWrapper(current_layer=LogisticRegression(*args, **kwargs))

    return lr


class LogisticRegressionWrapper(AbstractModel):
    """Wrapper for scikit-learn logistic regression model."""

    def fit(self, X, y):
        self._features_count = X.shape[1]
        # create (n,1)-vector of zeros if X is empty
        if self._features_count == 0:
            X = np.zeros((X.shape[0], 1), dtype=X.dtype)
        # fit scikit-lear logistic regression
        self.current_layer.fit(X, y)
        # get weights from fitted scikit-learn model
        self.feature_importances_ = self.current_layer.coef_.ravel()

        return self

    def predict(self, X):
        # create (n,1)-vector of zeros if X is empty
        if self._features_count == 0:
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
        # model must be fitted first to return something
        return self.feature_importances_

    def get_support(self, as_indices=True):
        if as_indices:
            # get indexes of the features
            return np.arange(self._features_count)
        else:
            # tells which feature to show
            return np.ones(self._features_count, dtype=np.bool)
