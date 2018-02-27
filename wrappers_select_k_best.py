
from sklearn.feature_selection import SelectKBest



def get_support_for_feature_selection_wrapper(
    feature_selector_indices,
    inner_indices,
    indices,
    ):
    result_support_indices = feature_selector_indices[inner_indices] if sum(feature_selector_indices.shape) > 0 \
                                else feature_selector_indices
    if indices:
        return result_support_indices
    else:
        raise KeyError("indices should be true")



####################################################################################################


## 1. Select k best features based on single-marker statistical tests


@scope.define
def get_k_best(*args, **kwargs):
    return SelectKBest(*args, **kwargs)



@scope.define
def get_k_best_wrapper(*args, **kwargs):
    return SelectKBestWrapper(*args, **kwargs)


class SelectKBestWrapper(BaseEstimator):
    def __init__(self, inner_model, k_best):
        self.inner_model = inner_model
        self.k_best = k_best

    def _fix_params(self, X, y):
        self.k_best.set_params(k=int(min(self.k_best.get_params()['k'], X.shape[1])))  #### correction int()

    def fit(self, X, y):
        self._fix_params(X, y)
        X = self.k_best.fit_transform(X, y)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        X = self.k_best.transform(X)
        return self.inner_model.predict(X)

    def get_support(self, indices=False):
        feature_selector_support = self.k_best.get_support(indices=True)
        inner_model_support = self.inner_model.get_support(indices=True)
        return get_support_for_feature_selection_wrapper(
            feature_selector_support,
            inner_model_support,
            indices,
        )


    def get_feature_importances(self):
        return self.inner_model.feature_importances_



####################################################################################################

## 2. Select k best features based on importance weights in regression and random forest models



@scope.define
def get_model_based_feature_selection_model(*args, **kwargs):
    return ModelFeatureSelectionWrapper(*args, **kwargs)


def get_model_based_feature_selector_params(
        inner_model_params,
        name='model_based_feature_selector',
        estimator=None,
    ):
    if estimator is None:
        estimator = get_feature_selector_estimator_params()
    return scope.get_model_based_feature_selection_model(
        estimator=estimator,
        inner_model=inner_model_params,
        feature_selection_threshold_coef=hp.loguniform(get_full_name(name, "threshold"), -5, 5),
    )



class ModelFeatureSelectionWrapper(BaseEstimator):
    def __init__(self, estimator, inner_model, feature_selection_threshold_coef=3):
        self.estimator=estimator
        self.inner_model = inner_model
        self.feature_selector = None
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_feature_selector(self):
        if self.feature_selector is None:
            self.feature_selector = SelectFromModel(self.estimator,
                                                    threshold='{}*mean'.format(float(self.feature_selection_threshold_coef)))
        return self.feature_selector

    def get_support(self, indices=False):
        feature_selector_support = self.feature_selector.get_support(indices=True)
        inner_support = self.inner_model.get_support(indices=True)
        return get_support_for_feature_selection_wrapper(
            feature_selector_support,
            inner_support,
            indices,
        )

    def fit(self, X, y):
        print X, X.shape
        X = self._get_feature_selector().fit(X.copy(), y.copy()).transform(X.copy())
        self.inner_model.fit(X.copy(), y)
        return self

    def predict(self, X):
        X = self._get_feature_selector().transform(X.copy())
        return self.inner_model.predict(X.copy())

    def get_feature_importances(self):
        return self.inner_model.feature_importances_



class ModelBasedFeatureImportanceGetter(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def get_feature_importances(self, X, y):
        return self.inner_model.fit(X, y).feature_importances_

    def get_support(self, *args, **kwargs):
        return self.inner_model.get_support(*args, **kwargs)

