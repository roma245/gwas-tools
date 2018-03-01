
# -*- coding: utf-8 -*-

import time
import traceback

from hyperopt import hp, STATUS_OK, STATUS_FAIL, Trials, fmin, tpe
from hyperopt.pyll import scope

from models import *


###############################################################################
class MetamodelFactory(object):
    """Base class for metamodel creation and hyperparameters search."""

    def __init__(self, max_evals=100):
        self.param_space = None
        self.results_dumper = None
        self.metrics_getter = None
        self.trials = Trials()
        self.max_evals = max_evals

    def configure_params(self, inner_model, feature_selector=None):
        """Factory method for configuration metamodel structure."""
        raise NotImplementedError()

    def set_result_dumper(self, result_dumper):
        self.results_dumper = result_dumper

    def set_metrics_getter(self, metrics_getter):
        self.metrics_getter = metrics_getter

    def get_objective_function(self, X, y,  features, objects, X_test, callback=None):
        """Objective function for calculating loss for checked hyperparameters."""
        def objective_function(model):

            # create estimator with concrete values of hyperparameters
            model_layers = list(model)
            model = SequentialModel(name=self.name, layers=model_layers)

            if not model:
                raise ValueError, "Model can't be 'None' in objective function."

            start_time = time.time()
            try:
                # get in 'metrics' dictionary all possible metrics and predictions on X_test dataset
                metrics, loss = self.metrics_getter(model, X, y, features, objects, X_test)
                result = {
                    'status': STATUS_OK,
                    'loss': loss,
                    'full_metrics': metrics,
                    'time_calculated': time.time(),
                    'time_spent': time.time() - start_time,
                    'model': model,
                }
            except:
                with open(self.results_dumper.get_errors_log_filename(), "a") as f:
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.write(repr(model))
                    f.write("\n")
                result = {
                    'status': STATUS_FAIL,
                    'traceback': traceback.format_exc(),
                    'time_calculated': time.time(),
                    'time_spent': time.time() - start_time,
                    'model': model,
                }

            if callback is not None:
                callback(result)

            return result

        return objective_function

    def fit(self, X, y,  features, objects, X_test=None):
        """Search best hyperparameters for (X, y) dataset."""
        try:
            best = fmin(
                fn=self.get_objective_function(
                    X,
                    y,
                    features,
                    objects,
                    X_test,
                    callback=lambda result: self.results_dumper.add_result(result),
                ),
                space=self.param_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=self.trials
            )
            #print best
            min_loss = 1.0
            for el in self.trials.results:
                if el['status'] == STATUS_OK:
                    if el['loss'] < min_loss:
                        min_loss = el['loss']
                        self._result_model = el['model']

            self._result_model.fit(X, y)

        finally:
            self.results_dumper.flush()

        return self

    def predict(self, X):
        return self._result_model.predict(X)

    def get_feature_importances(self):
        """Return importance weights of all features in fitted model."""
        return self._result_model.get_feature_importances()

    def get_support(self, as_indices=False):
        """Return feature names or feature indices. In case of indices=False returns a list of TRUE/FALSE values
        that indicate which positions in feature names array should be shown.
        """
        return self._result_model.get_support(as_indices=as_indices)

    def get_hyperparams(self, deep=True):
        return self._result_model.get_hyperparams(deep)

    def load_model(self, filename):
        """Read model from file."""
        pass

    def save_model(self, filename):
        """Save result model with fitted hyperparameters into file."""
        pass

    def load_trials(self, filename):
        """Load trials database from file to continue search."""
        self.trials = self.results_dumper.load_hp_trials(filename)

    def save_trials(self, filename):
        """Save current trials in one file."""
        self.results_dumper.save_hp_trials(self.trials, filename)


###############################################################################
class SimpleFeaturesMetamodel(MetamodelFactory):
    """Initialize space of metamodel parameters with model
    and do nothing with features (simple features).
    """

    def configure_params(self, inner_model, feature_selector=None):
        """Initialize space of metamodel parameters with model and do not transform initial features.

        It is supposed that 'inner_model' parameter of class 'AbstractModel'.
        """
        self.name = inner_model.name
        self.param_space = inner_model.get_configuration()


class ComplexFeaturesMetamodel(MetamodelFactory):
    """Initialize metamodel parameter space with model and configure rules
    for transforming initial features into combinations of features (complex features).
    """

    def configure_params(self, inner_model, feature_selector=None):
        """Initialize space of metamodel parameters with model and feature selector that will
        transform initial features to something different, e.g. combinations of initial features.
        """
        self.param_space = scope.get_model_with_feature_selector(
            inner_model=inner_model,
            extender_strategy=feature_selector
        )

