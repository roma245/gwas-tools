
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from hyperopt import hp
from hyperopt.pyll import scope


###############################################################################
@scope.define
def get_model(model_instance):
    """This function adds existing model instance to hyperopt score."""
    if not isinstance(model_instance, AbstractModel):
        raise TypeError, "Model is not instance of AbstractModel class."

    return model_instance


class AbstractModel(BaseEstimator):
    def __init__(self, current_layer=None):
        self.current_layer = current_layer
        self.lower_layer = None
        self.feature_importances_ = None

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def get_hyperparams(self, deep=True):
        raise NotImplementedError()

    def get_feature_importances(self):
        raise NotImplementedError()

    def get_support(self, as_indices=False):
        raise NotImplementedError()


###############################################################################
class ModelSelector(AbstractModel):
    """Base class for constructing multi-layer models."""

    def __init__(self, name, children=list(), lower_layer=None):
        super(ModelSelector, self).__init__(
            current_layer=None,
            lower_layer=lower_layer
        )
        self.name = name
        self.children = children

    def add(self, child_model):
        #if isinstance(child_model_getter, AbstractModel) and not child_model_getter in self._children:
        if not child_model in self.children:
            self.children.append(child_model)

    def remove(self, child_model):
        index = self.children.index(child_model)
        del self.children[index]

    def get_child(self, index):
        return self.children[index]

    def fit(self, X, y):
        if not self.children:
            raise ValueError, "Current level not defined."

        if len(self.children) == 1:
            self.current_layer = self.children[0]
        else:
            self.current_layer = hp.choice(self.name, self.children)

        # get amount of features in the initial dataset X
        self._features_count = X.shape[1]

        if self.lower_layer is not None:
            X = self.lower_layer.transform(X)

        self.current_layer.fit(X, y)

        return self

    def predict(self, X):
        if self.lower_layer is not None:
            X, features = self.lower_layer.transform(X)

        return self.current_layer.predict(X)

    def transform(self, X):
        """Transform dataset X by deleting columns that correspond to less important features."""
        if self.lower_layer is not None:
            X = self.lower_layer.transform(X)

        weights = np.asarray(self.feature_importances_)
        indexes = np.where(np.abs(weights) > 0.0)[0]

        X_tr = X[:, indexes]

        return X_tr

    def get_hyperparams(self, deep=True):
        """Get all hyperparameters."""
        hp_params = dict()
        if self.lower_layer:
            # if this is a leaf and no lower-level models exist
            hp_params['selector_lower'] = self.lower_layer.get_hyperparams(deep)

        hp_params['selector_current'] = self.get_hyperparams(deep)

        return hp_params

    def get_support(self, as_indices=True):
        """Get indexes of features to restore feature names.

        This method should be invoked only after model is fitted.
        """
        if not self._features_count:
            raise ValueError, "Fit the model first."

        if as_indices:
            # get indexes of the features in the initial dataset X
            features = np.arange(self._features_count)
        else:
            # this will tell us which features to show
            features = np.ones(self._features_count, dtype=np.bool)

        return self.transform(features)


###############################################################################
class SequentialModel(AbstractModel):
        """Base class for constructing multi-layer models."""

        def __init__(self, name, layers=None):
            super(SequentialModel, self).__init__()
            self.name = name
            self.layers = layers

        def get_configuration(self):
            return self.layers

        def get_name(self):
            return self.name

        def add(self, layer):
            """Add layer to the model."""
            if not self.layers:
                self.layers = list()
            #if isinstance(layer, AbstractModel):
            self.layers.append(layer)

        def remove(self, layer):
            index = self.layers.index(layer)
            del self.layers[index]

        def merge_previous(self, prev_layers=2):
            """Merge given number of previous layers using np.choice() operation.
            Assume that the first element in layers list is the lowest model layer and
            the last is on top.

            Example: [x1, x2, x3] -> [x1, np.choice(x2, x3)]."""
            if len(self.layers) < prev_layers:
                raise ValueError, "Total number of layers is less than layers to merge."

            merged_index = len(self.layers)-prev_layers+1
            merged_layer = self.layers[-prev_layers:]
            for _ in range(prev_layers):
                self.layers.pop()
            self.layers.append(
                hp.choice("layer_{}".format(merged_index), merged_layer))

        def get_layer(self, index):
            return self.layers[index]

        def fit(self, X, y):
            if not self.layers:
                raise ValueError, "Model not defined."

            # get amount of features in the initial dataset X
            self.features_count = X.shape[1]

            # fit models and transform data on previous layers
            for layer in self.layers[:-1]:
                layer.fit(X, y)
                X = layer.transform(X)
            # fit top layer
            self.layers[-1].fit(X, y)
            # get feature importances
            self.feature_importances_ = self.layers[-1].get_feature_importances()

            return self

        def predict(self, X):
            for layer in self.layers[:-1]:
                X = layer.transform(X)

            return self.layers[-1].predict(X)

        def transform(self, X):
            """Transform dataset X by deleting columns that correspond to less important features."""
            for layer in self.layers:
                X = layer.transform(X)

            return X

        def get_hyperparams(self, deep=True):
            """Get all hyperparameters."""
            hp_params = dict()
            for i, layer in enumerate(self.layers):
                hp_params['layer_{}'.format(i)] = layer.get_hyperparams(deep)

            return hp_params

        def get_feature_importances(self):
            return self.feature_importances_

        def get_support(self, as_indices=True):
            """Get indexes of features to restore feature names.

            This method should be invoked only after model is fitted.
            """
            if not self.features_count:
                raise ValueError, "Fit the model first."

            if as_indices:
                # get indexes of the features in the initial dataset X
                features = np.arange(self.features_count)
            else:
                # this will tell us which features to show
                features = np.ones(self.features_count, dtype=np.bool)

            features = features.reshape(1, self.features_count)
            for layer in self.layers:
                features = layer.transform(features)

            return features.ravel()


###############################################################################
# Models for constructing and checking features based on feature1_AND_feature2, feature1_OR_feature2 rules

def and_arrays(arrays):
    arrays_sum = arrays.sum(axis=0)
    return (arrays_sum == len(arrays)).astype(arrays.dtype)


def test_features_combimation(combination, X, y):
    combination_feature = and_arrays(X[:, combination].T)
    matr = np.zeros((2, 2), dtype=np.int32)
    for y_true, y_pred in izip(y, combination_feature):
        matr[y_true, y_pred] += 1

    return chi2_contingency(matr)[1]