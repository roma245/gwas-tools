import numpy as np
from sklearn.base import BaseEstimator
#from joblib import Memory
from pygco import pygco
from common import forward_out
from wrappers import ModelBasedFeatureImportanceGetter
#from numba import jit


#mem_frn = Memory(cachedir='cache/frn')

#@forward_out("logs/frn.log")
#@mem_frn.cache



@scope.define
def get_frn(*args, **kwargs):
    return FeatureRelevanceNetworkWrapper(*args, **kwargs)


def get_frn_params(
    inner_model,
    feature_importances_getter=None,
    name='frn_common'
    ):
    if feature_importances_getter is None:
        feature_importances_getter = get_feature_selector_estimator_params()

    return scope.get_frn(
        inner_model=inner_model,
        feature_importances_getter=feature_importances_getter,
        lmbda=hp.loguniform(
            get_full_name(name, 'lmbda'),
            -5, 15,
        ),
        feature_selection_threshold_coef=hp.uniform(
            get_full_name(name, 'threshold_coef'),
            0.0, 1.0,
        ),
    )





def get_rel_feat_frn(params, X, feature_importances):
    frn = FeatureRelevanceNetwork(**params)
    return frn.get_relevant_features(X, feature_importances)


#@jit(nopython=True)
def do_params_estimation_heavy_job(X):
    correlation_coefficient = np.corrcoef(np.array(X.T, dtype=np.float64))
    indices = np.triu_indices(correlation_coefficient.shape[0], 1)
    edges = np.concatenate(
        (indices[0].reshape((1, indices[0].shape[0])), 
         indices[1].reshape((1, indices[1].shape[0]))), 
    axis=0).T
    edge_weights = np.abs(correlation_coefficient[indices])
    edges = edges[edge_weights >= 0.75]
    edge_weights = edge_weights[edge_weights >= 0.75]
    return edges, edge_weights


class FeatureRelevanceNetwork(BaseEstimator):
    def __init__(self,
                 lmbda,
                 feature_selection_threshold_coef):
        self.lmbda = lmbda
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_model_parameters(self, X, feature_importances):
        real_importances = np.array(feature_importances)
        real_importances = real_importances - real_importances.min()
        real_importances = real_importances / real_importances.max()
        
        unary_cost = np.zeros((X.shape[1], 2), dtype=np.float64)
        unary_cost[:,0] = real_importances
        unary_cost[:,1] = 2 * self.feature_selection_threshold_coef - real_importances
        pairwise_cost = np.array([[0.0, 1.0],
                                  [1.0, 0.0]])
        edges, edge_weights = do_params_estimation_heavy_job(X)
        edge_weights = edge_weights * self.lmbda
        return edges, edge_weights, unary_cost, pairwise_cost

    def get_relevant_features(self, X, feature_importances):
        print "get_relevant_features_call"
        edges, edge_weights, unary_cost, pairwise_cost = self._get_model_parameters(X, feature_importances)
        result = pygco.cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, n_iter=1, algorithm="expansion")
        print "features count:", result.sum()
        print result, result.shape
        return result



class FeatureRelevanceNetworkWrapper(BaseEstimator):
    def __init__(self,
                 inner_model,
                 feature_importances_getter,
                 lmbda = 10000.0,
                 feature_selection_threshold_coef=0.5):
        self.inner_model = inner_model
        self.feature_importances_getter = feature_importances_getter
        self.lmbda = lmbda
        self.feature_selection_threshold_coef = feature_selection_threshold_coef

    def _get_reduced_params(self):
        result = self.get_params(deep=False).copy()
        del result['inner_model']
        del result['feature_importances_getter']
        return result

    def _filter_X(self, X):
        result = np.array(X)[:,self._relevant_feature_mask.astype(np.bool)]
        return result

    def _set_relevant_feature_mask(self, X, y):
        feature_importances = ModelBasedFeatureImportanceGetter(
            self.feature_importances_getter,
        ).get_feature_importances(X, y)
        self._relevant_feature_mask = get_rel_feat_frn(self._get_reduced_params(), X, feature_importances)

    def get_support(self, indices=False):
        if not indices:
            return self._relevant_feature_mask
        else:
            return np.array([i for i in xrange(len(self._relevant_feature_mask)) if self._relevant_feature_mask[i]])

    def fit(self, X, y):
        self._set_relevant_feature_mask(X, y)
        X = self._filter_X(X)
        self.inner_model.fit(X, y)
        return self

    def predict(self, X):
        X = self._filter_X(X)
        return self.inner_model.predict(X)

####
    def get_feature_importances(self):
        return self.inner_model.get_feature_importances()