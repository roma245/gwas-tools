# -*- coding: utf-8 -*-


import time
import sys
from os.path import isfile, join
import numpy as np
import pandas as pd
from data_keeper import get_data_keeper
from generate_subsets import SubsetGenerator

RAW_X_BEFORE_SUBSET_GENERATION_PATH = "data/raw_X_before_subsets_generation.csv"
POSSIBLE_COMPLEX_FEATURES_PATH = "possible_complex_features.txt"
get_generator_result = None

#matrix2 = pd.read_csv('data/apr17.snps.matrix', sep='\t', low_memory=False).drop([0, 1], axis=0).set_index('#snp_pos')


def make_new_generator():
    start_time = time.time()
    X = get_data_keeper().get_common_x()

    print "matrix shape before:", X.shape  # Матрица X = наша матрица мутаций (snps)
    X[X!=1] = 0

    to_drop = (X.sum(axis=0) >= (X.shape[0] / 2)) | (X.sum(axis=0) < 3)   # Убираем столбцы (=позиции мутаций), где слищком малое число мутированных образцов
    to_drop = to_drop[to_drop].index
    X = X.drop(to_drop, axis=1)

    # save filtered SNPs matrix (add saving to temp directory)
    X.to_csv(RAW_X_BEFORE_SUBSET_GENERATION_PATH)

    print "matrix shape after:", X.shape    # оставили только слолбцы, где более 3-х мутированных образцов
    sys.stdout.flush()

    # реализовать, чтобы генерация подмножеств не запускалась, если этого не требуется
    generator = SubsetGenerator()

    generator.generate_and_set(X.as_matrix().astype(np.uint8))  # генерируем набор подмножеств (запускаем модуль на Си++)

    print "generating done, time from start spent:", time.time() - start_time

    # временный файл для хранения сгенерированных подмножеств: нужно чтобы это было в каталоге $HOME/gwas/tmp
    generator.store(POSSIBLE_COMPLEX_FEATURES_PATH)
    print "storing done, time from start spent:", time.time() - start_time
    return generator, X


def get_ready_generator_inner(compute_if_not_found=True, folder=None):
    global get_generator_result
    if get_generator_result is None:
        if folder is None:
            raw_X_before_subsets_generation_path = RAW_X_BEFORE_SUBSET_GENERATION_PATH
            possible_complex_features_path = POSSIBLE_COMPLEX_FEATURES_PATH
        else:
            raw_X_before_subsets_generation_path = join(folder, RAW_X_BEFORE_SUBSET_GENERATION_PATH)
            possible_complex_features_path = join(folder, POSSIBLE_COMPLEX_FEATURES_PATH)
        if isfile(raw_X_before_subsets_generation_path) and isfile(possible_complex_features_path):
            generator = SubsetGenerator()
            generator.load(possible_complex_features_path)
            X = pd.read_csv(raw_X_before_subsets_generation_path, index_col=0)
            generator.set_raw_matrix(X.as_matrix().astype(np.uint8))
            get_generator_result = generator, X
        else:
            if compute_if_not_found:
                get_generator_result = make_new_generator()
    return get_generator_result


class GeneratorGetter(object):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return get_ready_generator_inner(*self._args, **self._kwargs)[0]
    
    def __getstate__(self):
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        self.__dict__ = state.copy()


def get_ready_generator(*args, **kwargs):
    generator, X = get_ready_generator_inner(*args, **kwargs)
    return SubsetGeneratorWrapper(GeneratorGetter()), X


class SubsetGeneratorWrapper:
    def __init__(self, gen_getter):
        self._gen_getter = gen_getter

    def __getattr__(self, attr):
        return self._gen_getter().__getattribute__(attr)

    def __getinitargs__(self):
        return [self._gen_getter]




__all__ = ['get_ready_generator']




