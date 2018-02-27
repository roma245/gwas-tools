
# -*- coding: utf-8 -*-

from os import mkdir, listdir
from os.path import isfile, isdir, join
from os.path import expanduser

import cPickle

import numpy as np

from sklearn.externals import joblib
from scipy.stats.kde import gaussian_kde
from hyperopt import Trials

import matplotlib.pyplot as plt

from metrics_getter import METRICS, HYPERPARAMS, ROC_AUC


###############################################################################
SAVE_RESULTS_AFTER = 10


###############################################################################
def smooth(x, window_len=11, window='hanning'):
    """Window-based smoothing."""
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


def plot_cdf_in_time(y, scores=None):
    """Plot estimated probability density of cumulated y values in time."""
    y = np.asarray(y)
    x = np.arange(y.size)

    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = np.zeros(xi.shape)

    xi_ticks = xi.T[0][1:]
    for i, xi_tick in enumerate(xi_ticks):
        density = gaussian_kde(y[0:int(xi_tick)])
        zi[i] = density(yi[0])
    zi[xi_ticks.size] = zi[xi_ticks.size-1]

    plt.contourf(xi, yi, zi, alpha=0.5)

    rgba_colors = np.zeros((y.size, 4))
    # for red the first column needs to be one
    rgba_colors[:, 2] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = 1.0
    if scores is not None:
        scores = np.asarray(scores)
        if scores.ndim == y.ndim:
            rgba_colors[:, 3] = (scores - scores.min()) / (scores.max() - scores.min())

    # plot values as points
    plt.scatter(x, y, marker='.', color=rgba_colors)

    return plt


###############################################################################
def writeDict(dict, filename, sep):
    with open(filename, "a") as f:
        for i in dict.keys():
            f.write(i + " " + sep.join(str(dict[i])) + "\n")


###############################################################################
class ResultsDumper(object):
    """Main class for treating trials."""

    def __init__(self, experiment_name, root_folder="gwas_experiments"):
        self._experiment_name = experiment_name
        self._unflushed = list()
        self._flush_counter = 0
        self._results_counter = 0
        self.set_root_folder(root_folder = join(expanduser("~"), root_folder))
        self._folder = self._root_folder
        self.set_subdir(self._experiment_name)

    def __del__(self):
        self.flush()

    # .. treating directories with trial logs ..

    def set_root_folder(self, root_folder):
        """Set root folder for all experiments."""
        if not isdir(root_folder):
            mkdir(root_folder)
        self._root_folder = root_folder

    def set_subdir(self, subdir):
        """Set folder for current experiments."""
        self._folder = join(self._folder, subdir)
        if not isdir(self.get_folder()):
            mkdir(self.get_folder())

    def get_folder(self):
        return self._folder

    # .. getting files with trial logs ..

    def get_logs_file(self):
        filename = join(self.get_folder(), "execution.log")
        return open(filename, 'a')

    def get_errors_log_filename(self):
        filename = join(self.get_folder(), "errors.log")
        return filename

    # .. working with trial results ..

    def set_test_true(self, test_true):
        """Set ture y values on test set for adding into search logs."""
        self._test_true = test_true

    def add_result(self, result):
        """Add result record into the list.

        Method will flush all results and clear the list of result records
        if list's length is greater than SAVE_RESULTS_AFTER constant.
        """
        result = result.copy()
        result['test_true'] = self._test_true
        self._unflushed.append(result)
        self._results_counter += 1
        with open(join(self.get_folder(), "counter.txt"), 'a') as f:
            f.write("{}\n".format(self._results_counter))
        if len(self._unflushed) >= SAVE_RESULTS_AFTER:
            self.flush()

    def flush(self):
        """Save list of result records into file and clear the list."""
        while True:
            filename = join(self.get_folder(), "{}.pkl".format(self._flush_counter))
            if not isfile(filename):
                joblib.dump(self._unflushed, filename, compress=3)
                self._unflushed = list()
                return
            self._flush_counter += 1

    def flush_result(self, result, filename_pref="final_model", filename_suff=""):
        """Save single result record in file as cPickle object."""
        filename = join(self.get_folder(), "{}_{}.pkl".format(filename_pref, filename_suff))
        if not isfile(filename):
            savefile = open(filename, 'w')
            cPickle.dump(result, savefile, cPickle.HIGHEST_PROTOCOL)

            return

    # .. working with hyperopt trials database without any pre-processing ..

    def load_hp_trials(self, filename):
        """Load trials to continue search for hyperopt library."""
        try:
            trials = cPickle.load(open(filename, "rb"))
            print("Found saved Trials! Loading...")
        except:
            trials = Trials()

        return trials

    def save_hp_trials(self, trials, filename):
        """Save all hyperopt trials in single file."""
        with open(filename, "wb") as fp:
            cPickle.dump(trials, fp)

    # .. plotting search progress ..

    def plot_metrics_progress(self, metrics, window_len=11):
        """Plot metrics progress from trials data."""
        folder = self.get_folder()
        # prepare metrics dictionary
        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric] = list()
        # read files from folder with saved trials
        file_list = list()
        for file in listdir(folder):
            if file.endswith(".pkl"):
                try:
                    cur_id = int(file[:-4])
                    file_list.append(cur_id)
                except ValueError:
                    pass
        # extract metrics values from pickled trials
        file_list.sort()
        for file_id in file_list:
            filename = join(folder, "{}.pkl".format(file_id))
            if isfile(filename):
                objects = joblib.load(filename)
                for obj in objects:
                    for metric in metrics_dict.keys():
                        metrics_dict[metric].append(obj[METRICS][metric])
        # create and save plots
        for metric in metrics_dict.keys():
            plt.plot(smooth(np.array(metrics_dict[metric]), window_len))
            plt.savefig(join(folder, "{}_plot".format(metrics)))
            plt.clf()

    def plot_hp_progress(self, hyperparams):
        """Plot search progress for hyperparameters from trials data."""
        folder = self.get_folder()
        # prepare dictionary
        params_dict = {}
        for param in hyperparams:
            params_dict[param] = list()
        # read files from folder with saved trials
        file_list = list()
        for file in listdir(folder):
            if file.endswith(".pkl"):
                try:
                    cur_id = int(file[:-4])
                    file_list.append(cur_id)
                except ValueError:
                    pass
        scores = list()
        # extract hyperparameter values from pickled trials
        file_list.sort()
        for file_id in file_list:
            filename = join(folder, "{}.pkl".format(file_id))
            if isfile(filename):
                objects = joblib.load(filename)
                for obj in objects:
                    for param in params_dict.keys():
                        params_dict[param].append(obj[HYPERPARAMS][param])
                        scores.append(obj[METRICS][ROC_AUC])
        # plot hyperparameter values
        for param in params_dict.keys():
            plt = plot_cdf_in_time(params_dict[param], scores)
            plt.savefig(join(folder, "{}_hyperparam_plot".format(param)))
            plt.clf()

    # .. working with complex features ..

    def save_tuple(self, result):
        counter = 0
        while True:
            filename = join(self.get_folder(), "complex_features_{}.csv".format(counter))
            if not isfile(filename):
                with open(filename, 'w') as fp:
                    fp.write('\n'.join('%s %s' % (x[0], str(x[1]).replace('\n', '')) for x in result))
                return

            counter += 1
