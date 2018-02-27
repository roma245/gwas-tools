import sys
from os import mkdir, listdir
from os.path import isfile, isdir, join
from collections import defaultdict
from sklearn.externals import joblib
from common import SAVE_RESULTS_AFTER
import numpy as np
import pandas as pd
import cPickle

###
import testing
import pylab
import json



def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y


def writeDict(dict, filename, sep):
    with open(filename, "a") as f:
        for i in dict.keys():
            f.write(i + " " + sep.join(str(dict[i])) + "\n")

##########################


class ResultsDumper(object):
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
        self._unflushed = list()
        self._flush_counter = 0
        self._results_counter = 0
        self._folder = "experiments_results"
        self.set_subdir(self._experiment_name)
        
    def _get_folder(self):
        return self._folder

    def set_test_true(self, test_true):
        self._test_true = test_true

    def set_subdir(self, subdir):
        self._folder = join(self._folder, subdir)
        if not isdir(self._get_folder()):
            mkdir(self._get_folder())

    def add_result(self, result):
        result = result.copy()
        result['test_true'] = self._test_true
        self._unflushed.append(result)
        self._results_counter += 1
        with open(join(self._get_folder(), "counter.txt"), 'a') as f:
            f.write("{}\n".format(self._results_counter))
        if len(self._unflushed) >= SAVE_RESULTS_AFTER:
            self.flush()

    def get_logs_file(self):
        filename = join(self._get_folder(), "execution.log")
        return open(filename, 'a')

    def get_errors_log_filename(self):
        filename = join(self._get_folder(), "errors.log")
        return filename

    def dump_final_result(self, result, result_metrics):

        writeDict(result_metrics, join(self._get_folder(), "final_metrics.txt"), "")

        counter = 0
        while True:
            filename = join(self._get_folder(), "final_model_{}.pkl".format(counter))
            if not isfile(filename):
                #joblib.dump(result, filename, compress=1)

                savefile = open(filename, 'w')
                cPickle.dump(result, savefile, cPickle.HIGHEST_PROTOCOL)

                #savefile = open(filename, 'r')
                #obj = cPickle.load(savefile)

                df = pd.DataFrame({
                    'pos': np.asarray(result.get_support(True)),
                    str(result.inner_model): result.get_feature_importances()})
                #df.columns = [str(result.inner_model), 'pos']
                df.to_csv(join(self._get_folder(), "final_model_features{}.csv".format(counter)))

                return

            counter += 1






    ### plot f1, accuracy from test subset
    def plot_all_metrics(self):
        folder = self._get_folder()

        file_list = []
        for file in listdir(folder):
            if file.endswith(".pkl"):
                try:
                    cur_id = int(file[:-4])
                    file_list.append(cur_id)
                except ValueError:
                    pass

        file_list.sort()

        accuracy_list = []
        f1_list = []

        for file_id in file_list:
            filename = join(folder, "{}.pkl".format(file_id))

            if isfile(filename):
                objects = joblib.load(filename)

                for obj in objects:
                    test_pred = obj['full_metrics']['test_predictions'][1]
                    test_true = obj['test_true']

                    full_test_metrics = testing.get_y_true_y_pred_based_metrics(test_true,
                                                                                test_pred,
                                                                                testing.ALL_Y_TRUE_Y_PRED_BASED_METRICS)

                    accuracy_list.append(full_test_metrics['accuracy'])
                    f1_list.append(full_test_metrics['f1_score'])

        #print f1_list, accuracy_list

        #pylab.plot(range(0, len(accuracy_list)), accuracy_list)
        pylab.plot(smooth(np.array(accuracy_list), window_len=11))
        pylab.savefig(folder+'/accuracy_plot.png')
        pylab.clf()

        #pylab.plot(range(0, len(f1_list)), f1_list)
        pylab.plot(smooth(np.array(f1_list), window_len=11))
        pylab.savefig(folder+'/f1_plot.png')
        pylab.close()




    def save_tuple(self, result):
        counter = 0
        while True:
            filename = join(self._get_folder(), "complex_features_{}.csv".format(counter))
            if not isfile(filename):

                with open(filename, 'w') as fp:
                    fp.write('\n'.join('%s %s' % (x[0], str(x[1]).replace('\n', '')) for x in result))

                return


            counter += 1




    def flush(self):
        while True:
            filename = join(self._get_folder(), "{}.pkl".format(self._flush_counter))
            if not isfile(filename):
                joblib.dump(self._unflushed, filename, compress=3)
                self._unflushed = list()
                return
            self._flush_counter += 1        

    def __del__(self):
        self.flush()
