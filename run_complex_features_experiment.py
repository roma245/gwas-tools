import numpy as np
import math
import sys
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.base import BaseEstimator

from common import and_arrays
from common import get_experiment_name_for_drug
from data_keeper import get_data_keeper
from saving_results import ResultsDumper

from generate_subsets_for_common_x import get_ready_generator



def init_common():
    get_ready_generator()



def run_experiment(
                params,
                experiment_name,
                drug,
    ):
    experiment_name_for_drug = get_experiment_name_for_drug(experiment_name, drug)
    results_dumper = ResultsDumper(experiment_name_for_drug)
    results_dumper.set_subdir(str(0))

    X, y = get_data_keeper().get_train_data(drug, as_indexes=True)

    init_common()

    model = params

    model.fit(indexes=X, y=y)

    # save model.extender_strategy._result_feature_sets
    results_dumper.save_tuple(model.extender_strategy._result_feature_sets)


def get_all_params():
        # which estimator for feature importance to use and other params
    result_params = get_simple_feature_adder_wrapper_params(
                                    features_indexes_getter_opt = 'and',
                                    priority_getter_opt='simple')

    return result_params


def run_copmlex_features_estimator(drug):
    params = get_all_params()
    return run_experiment(
                params=params,
                experiment_name='complex_features',
                drug=drug,
            )


if __name__ == '__main__':
    run_copmlex_features_estimator(get_data_keeper().get_possible_second_level_drugs()[int(sys.argv[1])])
