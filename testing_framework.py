from data_keeper import get_data_keeper
from sklearn.model_selection import GridSearchCV

from testing import test_models_with_drugs

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

get_data_keeper().get_possible_drugs()

import wrappers
from wrappers import GridSearchCVWrapper
from wrappers import XGBoostClassifierFeatureImportances as XGB
from wrappers import MatrixCleaningWrapper
from wrappers import SparseWrapper
from wrappers import ModelFeatureSelectionWrapper
from wrappers import ModelBasedFeatureImportanceGetter
from wrappers import AsMatrixWrapper

from frn import FeatureRelevanceNetworkWrapper

def get_complete_linear_model():
    inner_model = LogisticRegressionCV(Cs=[10 ** i for i in xrange(-4, 4)], solver='liblinear')
    outer_model = GridSearchCV(inner_model, {'penalty': ['l1', 'l2']})
    return MatrixCleaningWrapper(SparseWrapper(outer_model))

def get_complete_tree_based_model():
    cv_params = {'inner_model__inner_model__n_estimators': [1],#, 5, 10, 20, 50, 100],
                 'feature_selection_threshold_coef': [0.1]}#, 1, 3, 10, 30, 100, 300]}
    return MatrixCleaningWrapper(FeatureRelevanceNetworkWrapper(XGB(n_estimators=100), ModelBasedFeatureImportanceGetter(XGB())))


from generate_subsets_for_common_x import get_ready_generator
#GENERATOR_FOLDER = '/media/vlad/01D198E892261920/vlad/vlad/diploma'
GENERATOR_FOLDER = '/home/roma/Documents/CurrentWork/tb_analysis/experiments_results'


import complex_features_inserting
complex_features_inserting = reload(complex_features_inserting)
from complex_features_inserting import ExtenderStrategy, \
                                       MinSizePreFilter, \
                                       SimplePriorityGetter, \
                                       BayesBasedPriorityGetter, \
                                       ComplexFeaturesAdderWrapper, \
                                       AndBasedSimpleFeaturesIndexGetter, \
                                       MinSimpleFeaturesIndexGetter

def get_simple_feature_adder_wrapper(inner_model, max_features_to_add):
    #generator, matrix_before_generating = get_ready_generator()
    #priority_getter = SimplePriorityGetter()
    #pre_filter = MinSizePreFilter(min_size=1)
    #simple_features_indexes_getter = MinSimpleFeaturesIndexGetter(generator, 1000)
    #extender_strategy = ExtenderStrategy(max_features=1000,
    #                                     priority_getter=priority_getter,
    #                                     pre_filter=pre_filter,
    #                                     generator=get_ready_generator()[0],
    #                                     simple_features_indexes_getter=simple_features_indexes_getter)
    feature_selector = AsMatrixWrapper(ModelFeatureSelectionWrapper(inner_model))
    return feature_selector
    return ComplexFeaturesAdderWrapper(inner_model=feature_selector,
                                       matrix_before_generating=matrix_before_generating.as_matrix(),
                                       features_names=list(matrix_before_generating.columns.values),
                                       extender_strategy=extender_strategy)


from sklearn.linear_model import LogisticRegression
simple_feature_adder_wrapper = get_simple_feature_adder_wrapper(XGB(), 1000)

from testing import test_models_with_drugs
test_models_with_drugs([('model', simple_feature_adder_wrapper)], ['ETHI: Ethionamide/ Prothionamide '], as_indexes=False)

