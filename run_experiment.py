
# -*- coding: utf-8 -*-


import sys
from multiprocessing import Process
from sklearn.cross_validation import StratifiedKFold
from base import get_experiment_name_for_drug, PROCESSORS_COUNT, RANDOM_STATE, MAX_EVALS
from data_keeper import get_data_keeper
from metamodel import MetamodelFactory
from results_dumper import ResultsDumper
from metrics_getter import MetricsGetter, ALL_METRICS, ACCURACY, TEST_PREDICTIONS
from generate_subsets_for_common_x import get_ready_generator

from wrappers_logistic import get_linear_model_params


def init_common():
    get_ready_generator()


class AccuracyLossGetter:
    def __call__(self, metrics):
        return 1.0 - metrics[ACCURACY]


class ExperimentForDrugCaller:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, drug):
        kwargs = self._kwargs.copy()
        kwargs['drug'] = drug

        return run_experiment(*self._args, **kwargs)


def run_experiment_fold(model, X, y, train_index, test_index, fold_index):

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    model.results_dumper.set_subdir(str(fold_index))
    model.results_dumper.set_test_true(y_test)

    sys.stdout = sys.stderr = model.results_dumper.get_logs_file()

    model.fit(X_train, y_train, X_test)

    model_features = model.get_support(True)
    model_feature_importance = model.get_feature_importances()

    print model_features
    print model_feature_importance

    #df = pd.DataFrame({
     #   'imp': np.asarray(model_feature_importance),
    #    'pos': np.asarray(model_features)})

    #df.columns = [str(result.inner_model), 'pos']
   # df.to_csv(join('/home/roma/Documents/CurrentWork/tb_analysis/experiments_results', "final_model_features{}.csv".format(fold_index)))

    model.results_dumper.dump_final_result(model._result_model, model._result_metrics)

    ### calculate and plot model performance metrics
    ### read all files with models, read metrics for train data, calculate and plot metrics for test data

    model.results_dumper.plot_all_metrics()

######################


def run_experiment(
        params,
        experiment_name,
        drug,
        max_evals=100,
        as_indexes=True
    ):

    experiment_name_for_drug = get_experiment_name_for_drug(experiment_name, drug)
    results_dumper = ResultsDumper(experiment_name_for_drug)
    loss_getter = AccuracyLossGetter()

    inner_metrics_getter = MetricsGetter(
        metrics=ALL_METRICS,
        as_indexes=as_indexes,
        loss_func=loss_getter,
        n_folds=5,
    )

    model = MetamodelFactory(
        metamodel_structure=params,
        feature_selector=None,
        results_dumper=results_dumper,
        metrics_getter=inner_metrics_getter,
        max_evals=max_evals
    )

    X, y = get_data_keeper().get_train_data(drug, as_indexes=as_indexes)

    n_folds = 5

    if len(y) < 50:
        n_folds = 10

    init_common()
    
    processes = list()
    
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=RANDOM_STATE)):
        process = Process(
            target=run_experiment_fold,
            args=(model, X, y, train_index, test_index, i)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

########


def run_model(drug):
    params = get_linear_model_params()

    return run_experiment(
        params=params,
        experiment_name='model',
        drug=drug,
        as_indexes=True,
        #as_indexes=False,
        max_evals=MAX_EVALS,
    )


if __name__ == '__main__':
    #run_model(get_data_keeper().get_possible_second_level_drugs()[int(sys.argv[1])])
    run_model(get_data_keeper().get_possible_second_level_drugs()[int(2)])
    #run_model(get_data_keeper().get_possible_first_level_drugs()[int(2)])

