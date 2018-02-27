from os.path import join, isdir, isfile
from itertools import chain
from sklearn.externals import joblib
import numpy as np
from common import get_experiment_name_for_drug
from testing import get_y_true_y_pred_based_metrics, ALL_Y_TRUE_Y_PRED_BASED_METRICS, TEST_PREDICTIONS


DEFAULT_EXPERIMENTS_RESULTS_PATH = "experiments_results"


def get_fold_path(results_path, fold_index):
	return join(results_path, str(fold_index))


def get_dump_path(fold_path, dump_index):
	return join(fold_path, "{}.pkl".format(dump_index))

# Sample input:
# drug = 'CAPR Capreomycin '
# experiment_name = 'model'
# experiments_results_path = '/home/roma/Documents/CurrentWork/tb_analysis/experiments_results'
#

def load_results(
	experiment_name, 
	drug, 
	experiments_results_path, 
	fields=['status', 'loss', 'full_metrics', 'time_calculated', 'test_true']):
	results_path = join(experiments_results_path, get_experiment_name_for_drug(experiment_name, drug))
	
	results = list()
	fold_index = 0
	while True:
		fold_path = get_fold_path(results_path, fold_index)
		if not isdir(fold_path):
			break
		dump_index = 0
		while True:
			dump_path = get_dump_path(fold_path, dump_index)
			if not isfile(dump_path):
				break
			results_partial = joblib.load(dump_path)
			for result in results_partial:
				result_to_add = dict()
				result_to_add['fold'] = fold_index
				for field in fields:
					result_to_add[field] = result.get(field, None)
				results.append(result_to_add)
			dump_index += 1
		fold_index += 1
	return results


def get_y_true_y_pred_in_time(results):
	max_fold_index = 0
	for el in results:
		max_fold_index = max(max_fold_index, el['fold'])
	folds_count = max_fold_index + 1
	folds_values = [None] * folds_count
	result = list()
	for el in sorted(results, key=lambda result: result['time_calculated']):
		folds_values[el['fold']] = el
		folds_setted = True
		for i in xrange(folds_count):
			if folds_values[i] is None:
				folds_setted = False
				break
		if folds_setted:
			result.append((el['time_calculated'], folds_values[:]))
	return result

def calc_metrics(folds_result):
	y_true = np.array(list(chain.from_iterable(el['test_true'] for el in folds_result)))
	y_pred = np.array(list(chain.from_iterable(el['full_metrics'][TEST_PREDICTIONS][1] for el in folds_result)))
	return get_y_true_y_pred_based_metrics(y_true, y_pred, ALL_Y_TRUE_Y_PRED_BASED_METRICS)


# Top-level functions to get metrics

def get_metrics_through_time(experiment_name, drug, experiments_results_path=DEFAULT_EXPERIMENTS_RESULTS_PATH):
	results = load_results(experiment_name, drug, experiments_results_path)
	y_true_y_pred_in_time = get_y_true_y_pred_in_time(results)
	ans = list()
	for time, folds_results in y_true_y_pred_in_time:
		# Append metrics calculated for joined results from all folds
		ans.append((time, folds_results, calc_metrics(folds_results)))
	return ans



# Get only metrtics from best-performed experiments
def get_optimal_metrics_through_time(experiment_name, drug, experiments_results_path=DEFAULT_EXPERIMENTS_RESULTS_PATH):
	results = load_results(experiment_name, drug, experiments_results_path)
	max_fold_index = 0
	for el in results:
		max_fold_index = max(max_fold_index, el['fold'])
	folds_count = max_fold_index + 1
	folds_values = [None] * folds_count
	folds_losses = [1] * folds_count
	result = list()
	for el in sorted(results, key=lambda result: result['time_calculated']):
		el_fold = el['fold']
		loss = el['loss']
		if loss < folds_losses[el_fold]:
			folds_losses[el_fold] = loss
			folds_values[el_fold] = el
			folds_setted = True
			for i in xrange(folds_count):
				if folds_values[i] is None:
					folds_setted = False
					break
			if folds_setted:
				result.append((el['time_calculated'], folds_values[:], calc_metrics(folds_values)))
	return result