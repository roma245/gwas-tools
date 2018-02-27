import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.base import BaseEstimator
from common import and_arrays
from hyperopt import hp, STATUS_OK, STATUS_FAIL, Trials, fmin, tpe
from hyperopt.pyll import scope
from generate_subsets import SubsetGenerator
from data_keeper import get_data_keeper

import pandas as pd
import time
import sys
from os.path import isfile, join



RAW_X_BEFORE_SUBSET_GENERATION_PATH = "raw_X_before_subsets_generation.csv"
POSSIBLE_COMPLEX_FEATURES_PATH = "possible_complex_features.txt"
get_generator_result = None



@scope.define   #scope.get_complex_features_adder_wrapper
def get_model_space_generator(*args, **kwargs):
    matrix_before_generating = get_ready_generator()[1] # get generated complex features
    kwargs['matrix_before_generating'] = matrix_before_generating.as_matrix()
    kwargs['features_names'] = list(matrix_before_generating.columns.values)

    return ModelSpaceGenerator(*args, **kwargs)


class ModelSpaceGenerator(BaseEstimator):   #ComplexFeaturesAdderWrapper
    def __init__(self, inner_model, matrix_before_generating, features_names, extender_strategy):
        self.inner_model = inner_model
        self.matrix_before_generating = matrix_before_generating
        self.features_names = features_names
        self.extender_strategy = extender_strategy

    def _get_simple_features(self, indexes):
        return self.matrix_before_generating[indexes]

    def _fit_feature_extender(self, simple_features, y, indexes):
        self.extender_strategy.fit(simple_features, y, indexes)

    def _extend_features(self, simple_features):
        return self.extender_strategy.transform(simple_features)

    def fit(self, indexes=None, y=None, simple_features=None):
        indexes = indexes.ravel()
        if y is None:
            raise KeyError("y should be setted")
        if simple_features is None:
            simple_features = self._get_simple_features(indexes)
        else:
            simple_features = simple_features[self.features_names]
        self._fit_feature_extender(simple_features, y, indexes)
        extended_features = self._extend_features(simple_features)
        self.inner_model.fit(extended_features, y) # Possily should save fitted coefficients into file
        return self

    def predict(self, indexes=None, simple_features=None):
        if simple_features is None:
            if indexes is None:
                raise KeyError("either simple_features or indexes should be setted")
            indexes = indexes.ravel()
            simple_features = self._get_simple_features(indexes)
        extended_features = self._extend_features(simple_features)
        return self.inner_model.predict(extended_features)

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        extender_support = self.extender_strategy.get_support(indices=True, features_names=self.features_names)
        #extender_support = self.features_names
        return [extender_support[i] for i in self.inner_model.get_support(indices=True)]


    def get_feature_importances(self):
        return self.inner_model.get_feature_importances()







""""""




def get_simple_feature_adder_wrapper_params(
        inner_model_params,
        max_features=None,
        pre_filter=None,
        features_indexes_getter=None,
        priority_getter=None,
        name='feature_adder_common'
    ):
    # which way to estimate complex feature importance to use: Bayes approach or Simple approach
    priority_getter = priority_getter if priority_getter is not None \
        else get_priority_getter_params(get_full_name(name, 'priority_getter'))
    pre_filter = pre_filter if pre_filter is not None \
        else get_min_size_prefilter_params(get_full_name(name, 'pre_filter'))
    features_indexes_getter = features_indexes_getter if features_indexes_getter is not None \
        else get_index_getter_params(get_full_name(name, 'indexes_getter'))
    max_features = max_features if max_features is not None \
        else hp.qloguniform(
                get_full_name(name, 'max_features'),
                -1, 10, 1,
            )

    # exetender stratagy - which features to use? simple features or complex features selected by importance
    extender_strategy = hp.choice(
        get_full_name(name, 'extender_strategy'),
        (
            scope.get_extender_strategy(
                max_features=max_features,
                priority_getter=priority_getter,
                pre_filter=pre_filter,
                simple_features_indexes_getter=features_indexes_getter,
            ),
            scope.get_nothing_doing_extender_strategy(),
        ),
    )
    return scope.get_complex_features_adder_wrapper(
        inner_model=inner_model_params,
        extender_strategy=extender_strategy,
    )




""""""

