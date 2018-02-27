

# -*- coding: utf-8 -*-


from base import get_full_name

from hyperopt import hp, STATUS_OK, STATUS_FAIL, Trials, fmin, tpe
from hyperopt.pyll import scope

from wrappers_xgboost import get_xgboost_params
from wrappers_logistic import get_linear_model
from wrappers_forest import get_rf_model_params

from sklearn.feature_selection import f_classif, chi2





# Select from XGB, Regression or RandomForest

@scope.define
def get_basic_model_selector_primitive(*args, **kwargs):
    return BasicModelSelectorPrimitive(*args, **kwargs)


class BasicModelSelectorPrimitive(Model):
    def __init__(self, lower_level_model = None):
        super(BasicModelSelectorPrimitive, self).__init__()

        # set lower-level model
        self._lower_level_model = lower_level_model

        # set model name for logging
        name = "model_common"; xgb_name = 'xgb'; lr_name = 'lr'; rf_name = 'rf'

        # set model structure
        self._model_structure = hp.choice(
            name,
            (
                get_xgboost_params(name = get_full_name(name, xgb_name)),
                get_linear_model(name = get_full_name(name, lr_name)),
                get_rf_model_params(name = get_full_name(name, rf_name)),
            )
        )

    def fit(self, X, y):
        if self._lower_level_model is not None:
            X = self._lower_level_model.fit(X.copy(), y.copy()).transform(X.copy())

        self._model_structure.fit(X.copy(), y.copy())

        return self

    def transform(self, X):
        if self._lower_level_model is not None:
            X = self._lower_level_model().transform(X.copy())

        return self._model_structure.predict(X.copy())

    def predict(self, X):
        if self._lower_level_model is not None:
            X = self._lower_level_model.transform(X.copy())

        return self._model_structure.predict(X.copy())

    def get_feature_importances(self):

        return self._model_structure.feature_importances_



# Run one of {LR, RF or XGB} model on features selected by RF or LR using existing class from previous version

@scope.define
def get_multimarker_feature_selector_primitive(*args, **kwargs):
    return MultimarkerFeatureSelectorPrimitive(*args, **kwargs)


class MultimarkerFeatureSelectorPrimitive(Model):
    def __init__(self):
        super(MultimarkerFeatureSelectorPrimitive, self).__init__()

        # range for feature selectino threshold
        l_threshold = -5; u_threshold = 5

        name = 'multimarker_feature_selector'; xgb_name = 'xgb'; lr_name = 'lr'; rf_name = 'rf'

        # structure of the current lever - we will use existing class from previous version from this
        self._model_structure = scope.get_model_based_feature_selection_model(
            # lower layer model for estimating features importance
            estimator=hp.choice(
                name, (
                    get_rf_model_params(name=get_full_name(name, rf_name)),
                    get_linear_model(name=get_full_name(name, lr_name)),
                )
            ),
            # current layer model that works on filtered features
            inner_model=hp.choice(
                name, (
                    get_xgboost_params(name = get_full_name(name, xgb_name)),
                    get_linear_model(name = get_full_name(name, lr_name)),
                    get_rf_model_params(name = get_full_name(name, rf_name)),
                )
            ),
            # threshold for filtering important features
            feature_selection_threshold_coef=hp.loguniform(
                get_full_name(name, "threshold"), l_threshold, u_threshold
            ),
        )

    def fit(self, X, y):
        self._model_structure.fit(X.copy(), y.copy())
        return self

    def transform(self, X):
        return self._model_structure.predict(X.copy())

    def predict(self, X):
        return self._model_structure.predict(X.copy())



# Run one of {LR, RF or XGB} model on features selected by Chi2 or ANOVA test using existing class from previous version

@scope.define
def get_singleimarker_feature_selector_primitive(*args, **kwargs):
    return MultimarkerFeatureSelectorPrimitive(*args, **kwargs)


class SingelmarkerFeatureSelectorPrimitive(Model):
    def  __init__(self, lower_level_model):
        super(SingelmarkerFeatureSelectorPrimitive, self).__init__()

        name = 'single_marker_k_best_selector'; xgb_name = 'xgb'; lr_name = 'lr'; rf_name = 'rf'

        self._lower_level_model = lower_level_model

        self._model_structure = scope.get_k_best_wrapper(
            # current layer model that works on filtered features
            inner_model=hp.choice(
                name,
                (
                    get_xgboost_params(name = get_full_name(name, xgb_name)),
                    get_linear_model_params(name = get_full_name(name, lr_name)),
                    get_rf_model_params(name = get_full_name(name, rf_name)),
                )
            ),
            # lower layer model for estimating features importance
            k_best=scope.get_k_best(
                k=hp.qloguniform(
                    get_full_name(name, 'k'),
                    0, 5, 1
                ),
                score_func=hp.choice(
                    get_full_name(name, 'score_func'),
                    (
                        chi2,
                        f_classif
                    )
                ),
            ),
        )


########################################


def get_feature_selector_params(inner_model_params, name='feature_selector_params'):
    model_based_estimator = get_feature_selector_estimator_params()
    return hp.choice(
            name, (
                get_k_best_params(
                    inner_model_params=inner_model_params,
                    name=get_full_name(name, 'k_best'),
                ),
                #get_boruta_feature_selector_params(
                #    name=get_full_name(name, 'boruta'),
                #    inner_model_params=inner_model_params,
                #),
                get_model_based_feature_selector_params(
                    name=get_full_name(name, 'model_based_selector'),
                    estimator=model_based_estimator,
                    inner_model_params=inner_model_params,
                ),
            ),
        )







class MetamodelPrimitive1(Metamodel):
    def set_model_structure(self):
        self.model_structure = InnerModel1().get_model_structure()

    def set_feature_selector(self):
        self.feature_selector = SimpleFeatureSelector()



class MetamodelPrimitive2(Metamodel):
    def set_model_structure(self):
        self.model_structure = InnerModel2().get_model_structure()

    def set_feature_selector(self):
        self.feature_selector = SimpleFeatureSelector()







#####  model experiments


def get_model_params(name="model_common", xgb_name=None, lr_name=None):
    xgb_result_name = xgb_name if xgb_name is not None else get_full_name(name, 'xgb')
    lr_result_name = lr_name if lr_name is not None else get_full_name(name, 'lr')
    return hp.choice(name, (
                    get_xgboost_params(xgb_result_name),
                    get_linear_model_params(lr_result_name),
                    get_rf_model_params('rf'),
                )
           )

def get_all_params():
    inner_model_params = get_model_params()
    result_params = scope.get_complex_features_adder_wrapper(
        inner_model=inner_model_params,
        extender_strategy=scope.get_nothing_doing_extender_strategy(),
    )
    return result_params


#########  selector model experimetn


def get_all_params():
    inner_model_params = get_model_params()
    feature_selection_params = get_feature_selector_params(
        inner_model_params=inner_model_params,
    )
    result_params = scope.get_complex_features_adder_wrapper(
        inner_model=feature_selection_params,
        extender_strategy=scope.get_nothing_doing_extender_strategy(),
    )
    return result_params



########## frn model experimetn



def get_all_params():
    inner_model_params = get_model_params()
    frn_params = get_frn_params(inner_model_params)
    result_params = scope.get_complex_features_adder_wrapper(
        inner_model=frn_params,
        extender_strategy=scope.get_nothing_doing_extender_strategy(),
    )
    return result_params



######### extender selector model


def get_feature_selector_estimator_params(name='feature_selector_estimator'):
    return hp.choice(
        name, (
            get_rf_model_params(name=get_full_name(name, 'rf')),
            get_linear_model_params(name=get_full_name(name, 'lr')),
        )
    )


def get_feature_selector_params(inner_model_params, name='feature_selector_params'):
    model_based_estimator = get_feature_selector_estimator_params()
    return hp.choice(
            name, (
                get_k_best_params(
                    inner_model_params=inner_model_params,
                    name=get_full_name(name, 'k_best'),
                ),
                #get_boruta_feature_selector_params(
                #    name=get_full_name(name, 'boruta'),
                #    inner_model_params=inner_model_params,
                #),
                get_model_based_feature_selector_params(
                    name=get_full_name(name, 'model_based_selector'),
                    estimator=model_based_estimator,
                    inner_model_params=inner_model_params,
                ),
            ),
        )

def get_all_params():
    inner_model_params = get_model_params() # choose between parameters of model: RF, XGB or Log regression
    feature_selection_params = get_feature_selector_params( # choose between models for feature selection: Chi-squared or RF/XGB/LogRegr k most important features
        inner_model_params=inner_model_params,
    )
    result_params = get_simple_feature_adder_wrapper_params(
        inner_model_params=feature_selection_params,
    )
    return result_params



####### extender robust model


def get_feature_selector_estimator_params(name='feature_selector_estimator'):
    return hp.choice(
        name, (
            get_rf_model_params(name=get_full_name(name, 'rf')),
            get_linear_model_params(name=get_full_name(name, 'lr')),
        )
    )



def get_all_params():
    inner_model_params = get_feature_selector_estimator_params()
    result_params = get_simple_feature_adder_wrapper_params(
        inner_model_params=inner_model_params,
    )
    return result_params



##########  extender frn model


def get_all_params():
    inner_model_params = get_model_params()
    frn_params = get_frn_params(inner_model_params)
    result_params = get_simple_feature_adder_wrapper_params(
        inner_model_params=frn_params,
    )
    return result_params



#########  complex features experiments



def get_all_params():
        # which estimator for feature importance to use and other params
    result_params = get_simple_feature_adder_wrapper_params(
                                    features_indexes_getter_opt = 'and',
                                    priority_getter_opt='simple')

    return result_params


