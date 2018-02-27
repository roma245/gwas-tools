import sys
from hyperopt.pyll import scope
from run_experiment import run_experiment
from data_keeper import get_data_keeper
from hyperparameter_search import get_simple_feature_adder_wrapper_params,\
                                  get_feature_selector_params, \
                                  get_model_params
from common import MAX_EVALS


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


def run_selector_model(drug):
    params = get_all_params()
    return run_experiment(
        params=params,
        experiment_name='selector_model',
        drug=drug,
        as_indexes=True,
        max_evals=MAX_EVALS,
    )


if __name__ == '__main__':
    run_selector_model(get_data_keeper().get_possible_second_level_drugs()[int(sys.argv[1])])
    #run_selector_model(get_data_keeper().get_possible_second_level_drugs()[int(2)])
