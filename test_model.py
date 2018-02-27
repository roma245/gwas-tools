# -*- coding: utf-8 -*-

import sys
from os.path import isfile, isdir, join

import pandas as pd
import numpy as np

from multiprocessing import Process

from sklearn.model_selection import StratifiedKFold
from base import get_experiment_name_for_drug, PROCESSORS_COUNT, RANDOM_STATE, MAX_EVALS

from metamodel import MetamodelFactory

from metrics_getter import AccuracyLossGetter, MetricsGetter, ALL_METRICS, ACCURACY, TEST_PREDICTIONS
from results_dumper import ResultsDumper

from data_keeper import get_data_keeper

from wrappers_logistic import get_linear_model


class SimpleMetamodel(MetamodelFactory):
    def __init__(self, experiment_name):
        super(SimpleMetamodel, self).__init__()

        self.metamodel_structure = get_linear_model()
        self.results_dumper = ResultsDumper(experiment_name)
        self.metrics_getter = MetricsGetter(
                                            metrics=ALL_METRICS,
                                            as_indexes=False,
                                            loss_func=AccuracyLossGetter(),
                                            n_folds=5,
                                          )
        self.max_evals = 2000

    def generate_metamodel_param_space(self):
        return self.metamodel_structure


# Run experiment on one fold
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

    df = pd.DataFrame({
        'imp': np.asarray(model_feature_importance),
        'pos': np.asarray(model_features)})

    # df.columns = [str(result.inner_model), 'pos']
    df.to_csv(join('/home/roma/tb_gwas_experiments/experiments_results', "final_model_features{}.csv".format(fold_index)))

    #model.results_dumper.dump_final_result(model._result_model, model._result_metrics)

    ### calculate and plot model performance metrics
    ### read all files with models, read metrics for train data, calculate and plot metrics for test data

    model.results_dumper.plot_all_metrics()



# Run experiment
if __name__ == '__main__':

    drug_name = get_data_keeper().get_possible_second_level_drugs()[2]
    experiment_name_for_drug = get_experiment_name_for_drug("simple_logreg_model", drug_name)

    # create metamodels to test
    metamodelLR = SimpleMetamodel(experiment_name=experiment_name_for_drug)

    # load data  - in X it will return INDEXES of points for which y exists
    X, y = get_data_keeper().get_train_data(drug_name, as_indexes=False)

    n_folds = 5

    if len(y) < 50:
        n_folds = 10

    #init_common()

    processes = list()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(
                                                                n_splits=5,
                                                                shuffle=True,
                                                                random_state=RANDOM_STATE
                                                                 ).split(X, y)
                                                 ):
        process = Process(
            target=run_experiment_fold,
            args=(metamodelLR, X, y, train_index, test_index, i)
                         )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
