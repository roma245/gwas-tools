
# -*- coding: utf-8 -*-

# Run an XGBoost model with hyperparmaters that are optimized using hyperopt
# The output of the script are the best hyperparmaters
# The optimization part using hyperopt is partly inspired from the following script:
# https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py


# Data wrangling
from data_keeper import get_data_keeper
from base import get_experiment_name_for_drug, PROCESSORS_COUNT, RANDOM_STATE, MAX_EVALS

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde


from multiprocessing import Process

import pandas as pd

# Scientific

import numpy as np

# Test on benchmark dataset

from sklearn.datasets import load_breast_cancer

#




# Machine learning

#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Some constants

SEED = 42
VALID_SIZE = 0.2
TARGET = 'outcome'


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y




#-------------------------------------------------#

# Utility functions

#def intersect(l_1, l_2):
#    return list(set(l_1) & set(l_2))


#def get_features(train, test):
#    intersecting_features = intersect(train.columns, test.columns)
#    intersecting_features.remove('people_id')
#    intersecting_features.remove('activity_id')
#    return sorted(intersecting_features)

#-------------------------------------------------#

# Scoring and optimization functions


def get_score(X_train, y_train, X_valid, y_valid, score_list, C_list):
    def score(params):
        #print("Training with params: ")
        #print(params)
        #num_round = int(params['n_estimators'])
        #del params['n_estimators']
        #dtrain = xgb.DMatrix(train_features, label=y_train)
        #dvalid = xgb.DMatrix(valid_features, label=y_valid)
        #watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

        #gbm_model = xgb.train(params, dtrain, num_round,
        #                      evals=watchlist,
        #                      verbose_eval=True)
        lr_model = LogisticRegression(**params)

        lr_model.fit(X_train, y_train)

        #predictions = gbm_model.predict(dvalid,
        #                                ntree_limit=gbm_model.best_iteration + 1)

        y_pred = lr_model.predict(X_valid)

        score = roc_auc_score(y_valid, y_pred)

        score_list.append(score)
        C_list.append(params['C'])

        # TODO: Add the importance for the selected features
        #print("\tScore {0}\n\n".format(score))
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}
    return score


def optimize(X_train, y_train, X_valid, y_valid, score_list, C_list,
             #trials,
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

    space = {
            'C': hp.uniform('C', 0, 1),
            'random_state': RANDOM_STATE,
            'solver': 'liblinear'
    }

    #space_1 = hp.loguniform('C', -15, 1)

    """
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    """



    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(
        get_score(X_train, y_train, X_valid, y_valid, score_list, C_list),
        space, algo=tpe.suggest,
        # trials=trials,
        max_evals=3000)

    return best

#-------------------------------------------------#


# Load processed data

# You could use the following script to generate a well-processed train and test data sets:
# https://www.kaggle.com/yassinealouini/predicting-red-hat-business-value/features-processing
# I have only used the .head() of the data sets since the process takes a long time to run.
# I have also put the act_train and act_test data sets since I don't have the processed data sets
# loaded.

drug_name = get_data_keeper().get_possible_second_level_drugs()[0]
experiment_name_for_drug = get_experiment_name_for_drug("pure_logreg", drug_name)

# create metamodels to test


# load data
X, y = get_data_keeper().get_train_data(drug_name, as_indexes=False)

# Data from sklearn benchmark
#data = load_breast_cancer()
#X, y = data.data, data.target


#train_df = pd.read_csv('../input/act_train.csv').head(100)
#test_df = pd.read_csv('../input/act_test.csv').head(100)

#FEATURES = get_features(train_df, test_df)
#print(FEATURES)


#-------------------------------------------------#



# Extract the train and valid (used for validation) dataframes from the train_df

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE,
#                                random_state=SEED)
#train_features = train[FEATURES]
#valid_features = valid[FEATURES]
#y_train = train[TARGET]
#y_valid = valid[TARGET]

#print('The training set is of length: ', len(y_train))
#print('The validation set is of length: ', len(y_valid))

#-------------------------------------------------#

def run_experiment_fold(model, X, y, train_index, test_index, fold_index):
    X_train = X[train_index]
    y_train = y[train_index]

    X_valid = X[test_index]
    y_valid = y[test_index]

    score_list = []
    C_list = []

    best_hyperparams = optimize(X_train, y_train, X_valid, y_valid, score_list, C_list
        # trials
    )

    print("The best hyperparameters in fold {0} are: ".format(fold_index), "\n")
    print(best_hyperparams)

    #plt.plot(smooth(np.array(score_list), window_len=101))
    #plt.plot(smooth(np.array(C_list), window_len=101))


    x, y = np.arange(len(C_list)), np.array(C_list)

    ############### window-based approach
    """
    w = 53

    xi, yi = np.mgrid[x.min() + w - 1:x.max():(x.size - w + 1) * 1j, y.min():y.max():y.size * 1j]

    zi = np.zeros((x.size - w + 1, yi[0].size))

    for i in range(w, x.size + 1):
        values = y[i - w:i]

        #print values

        k = gaussian_kde(values)

        zi[i - w] = k(yi[0])

    plt.contourf(xi, yi, zi, alpha=0.5)


    """



    plot_cdf_in_time(y, np.array(score_list)).savefig("Plot {0}".format(fold_index))



    return


def plot_cdf_in_time(y, scores=None):
    x = np.arange(y.size)

    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]

    zi = np.zeros(xi.shape)

    xi_ticks = xi.T[0][1:]
    for i, xi_tick in enumerate(xi_ticks):
        density = gaussian_kde(y[0:int(xi_tick)])
        zi[i] = density(yi[0])
    zi[xi_ticks.size] = zi[xi_ticks.size-1]

    plt.contourf(xi, yi, zi, alpha=0.5)
    #plt.pcolormesh(xi, yi, zi, alpha=0.5)


    rgba_colors = np.zeros((scores.size, 4))
    # for red the first column needs to be one
    rgba_colors[:, 2] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = (scores - scores.min()) / (scores.max() - scores.min())

    plt.scatter(x, y, marker='.', color=rgba_colors)
    #plt.scatter(x, y, marker='+', alpha=0.5)

    return plt



# Run the optimization

# Trials object where the history of search will be stored
# For the time being, there is a bug with the following version of hyperopt.
# You can read the error messag on the log file.
# For the curious, you can read more about it here: https://github.com/hyperopt/hyperopt/issues/234
# => So I am commenting it.
# trials = Trials()


n_folds = 5

#init_common()

processes = list()
for i, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=5,
                                                              shuffle=True,
                                                              random_state=RANDOM_STATE
                                                             ).split(X, y)):
    process = Process(
        target=run_experiment_fold,
        args=(None, X, y, train_index, test_index, i)
                     )
    processes.append(process)
    process.start()

for process in processes:
    process.join()




