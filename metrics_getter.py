
# -*- coding: utf-8 -*-

from copy import deepcopy
from scipy.stats import norm
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from base import RANDOM_STATE


###############################################################################
ACCURACY = 'accuracy'
F1 = 'f1_score'
ROC_AUC = 'roc_auc_score'
CONFUSION_MATRIX = 'confusion_matrix'
VALUES_TRUE = 'y_valid'
VALUES_PRED = 'y_pred'
TEST_PREDICTIONS = 'y_test'
HYPERPARAMS = 'hyperparams'
FEATURES = 'features'
OBJECTS = 'objects'

ALL_METRICS = [
    ACCURACY, F1,
    ROC_AUC,
    CONFUSION_MATRIX,
    VALUES_TRUE,
    VALUES_PRED,
    TEST_PREDICTIONS,
    HYPERPARAMS,
    FEATURES,
    OBJECTS
]

ALL_Y_TRUE_Y_PRED_BASED_METRICS = [
    ACCURACY, F1,
    ROC_AUC,
    CONFUSION_MATRIX,
    VALUES_TRUE,
    VALUES_PRED
]

PLOT_METRICS = [
    ACCURACY, F1,
    ROC_AUC
]


###############################################################################
class AccuracyLossGetter:
    """Calculate loss function."""
    def __call__(self, metrics):
        return 1.0 - metrics[ROC_AUC]


class MetricsGetter:
    """Calculate metrics."""

    def __init__(self, metrics, loss_func, n_folds):
        self._metrics = metrics
        self._loss_func = loss_func
        self._n_folds = n_folds

    def __call__(self, model, X, y, features, objects, X_test=None):
        model = deepcopy(model)
        metrics = self.get_cv_metrics(
            model,
            X,
            y,
            features,
            objects,
            self._metrics,
            self._n_folds,
            X_test=X_test,
        )
        loss = self._loss_func(metrics)

        return metrics, loss

    def set_folds_count(self, n_folds):
        self._n_folds = n_folds

    def get_cv_metrics(self, model, X, y, features, objects, metrics, n_folds, X_test=None):
        """Calculate metrics for the model on (X, y) dataset using cross-validation."""
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=RANDOM_STATE
            )
        )
        # get metrics from training set
        result = self.get_y_true_y_pred_based_metrics(y, y_pred, metrics)
        # fit model to get features and predictions
        model.fit(X, y)

        if HYPERPARAMS in metrics:
            result[HYPERPARAMS] = model.get_hyperparams()
        if FEATURES in metrics:
            result[FEATURES] =features[model.get_support(as_indices=True)]
        if OBJECTS in metrics:
            result[OBJECTS] = objects
        if TEST_PREDICTIONS in metrics:
            # predictions for X_test
            result[TEST_PREDICTIONS] = model.predict(X_test)

        return result

    def get_y_true_y_pred_based_metrics(self, y_true, y_pred, metrics):
        """Calculate metrics for y_pred, y_true arrays."""
        result = dict()
        if ACCURACY in metrics:
            result[ACCURACY] = accuracy_score(y_true, y_pred)
        if F1 in metrics:
            result[F1] = f1_score(y_true, y_pred)
        if ROC_AUC in metrics:
            result[ROC_AUC] = roc_auc_score(y_true, y_pred)
        if CONFUSION_MATRIX in metrics:
            result[CONFUSION_MATRIX] = confusion_matrix(y_true, y_pred)
        if VALUES_TRUE in metrics:
            result[VALUES_TRUE] = y_true
        if VALUES_PRED in metrics:
            result[VALUES_PRED] = y_pred

        return result

    # .. compare predictions ..

    def results_differ_p_value(self, y_true, y1, y2):
        y1 = (np.array(y1) == np.array(y_true)).astype(np.float64)
        y2 = (np.array(y2) == np.array(y_true)).astype(np.float64)
        diff = y1 - y2
        norm_stat = diff.mean() / diff.std() * np.sqrt(diff.shape[0])
        quantile = norm.cdf(norm_stat)

        return min(quantile, 1.0 - quantile)
