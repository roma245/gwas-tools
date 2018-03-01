# -*- coding: utf-8 -*-

import sys
from os.path import isfile, isdir, join

import pandas as pd
import numpy as np

from multiprocessing import Process

from sklearn.model_selection import StratifiedKFold
from base import get_experiment_name_for_drug, PROCESSORS_COUNT, RANDOM_STATE, MAX_EVALS

from metamodel import *
from models import *

from metrics_getter import AccuracyLossGetter, MetricsGetter, ALL_METRICS, PLOT_METRICS
from results_dumper import ResultsDumper

from data_keeper import get_data_keeper

from wrappers_logistic import get_linear_model


# Run experiment on one fold
def run_experiment_fold(model, X, y,  features, objects, train_index, test_index, fold_index):
    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    model.results_dumper.set_subdir(str(fold_index))
    model.results_dumper.set_test_true(y_test)

    objects_train = objects[train_index]
    objects_test = objects[test_index]

    #sys.stdout = sys.stderr = model.results_dumper.get_logs_file()

    model.fit(X_train, y_train,  features, objects_train, X_test)

    feature_indices = model.get_support(as_indices=True)

    model_features = features
    model_feature_importance = model.get_feature_importances()

    print model_features[feature_indices]
    print model_feature_importance[feature_indices]

    df = pd.DataFrame({
        'importance': np.asarray(model_feature_importance[feature_indices]),
        'feature': np.asarray(model_features[feature_indices])})

    df.to_csv(join(model.results_dumper.get_root_folder(), "final_model_features{}.csv".format(fold_index)))

    #model.results_dumper.plot_metrics_progress(metrics=PLOT_METRICS)
    print "Best hyperparams: ", model.get_hyperparams(deep=True)


# Run experiment
if __name__ == '__main__':

    drug_name = get_data_keeper().get_possible_second_level_drugs()[2]
    experiment_name_for_drug = get_experiment_name_for_drug("simple_logreg_experiment_", drug_name)

    # create model
    my_model = SequentialModel(name='simpleLR')
    my_model.add(layer=get_linear_model())

    # create metamodel
    my_metamodel = SimpleFeaturesMetamodel()
    my_metamodel.configure_params(inner_model=my_model)

    my_metamodel.set_result_dumper(result_dumper=ResultsDumper(
        experiment_name=experiment_name_for_drug))

    my_metamodel.set_metrics_getter(metrics_getter=MetricsGetter(
        metrics=ALL_METRICS,
        loss_func=AccuracyLossGetter(),
        n_folds=5
    ))

    data_keeper = get_data_keeper()
    data_keeper.load_genotypes('data/apr17.snps.matrix')
    data_keeper.load_phenotypes('data/drugs_effect17.csv')

    X, y, features, objects = get_data_keeper().get_data_for_drug(drug_name, as_indexes=False)

    processes = list()
    for i, (train_index, test_index) in enumerate(
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE).split(X, y)):
        process = Process(
            target=run_experiment_fold,
            args=(my_metamodel, X, y, features, objects, train_index, test_index, i))

        processes.append(process)
        process.start()

    for process in processes:
        process.join()
