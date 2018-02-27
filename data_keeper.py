
# -*- coding: utf-8 -*-

import sys
from os import devnull
import pandas as pd
import numpy as np


###############################################################################
FIRST_LINE = [
    u'ETHA: Ethambutol',
    u'ISON: Isoniazid',
    u'RIFM: rifampicin',
    u'RIFP: Rifapentine',
    u'PYRA: Pyrazinamide',
    u'STRE: Streptomycin',
    u'CYCL: Cycloserine'
]

SECOND_LINE = [
    u'OFLO: Ofloxacin',
    u'CAPR: Capreomycin',
    u'AMIK: Amikacin',
    u'KANA: Kanamycin',
    u'LEVO: Levofloxacin',
    u'MOXI: Moxifloxacin',
    u'PARA: Para-aminosalicyclic acid',
    u'ETHI: Ethionamide'
]


###############################################################################
class DataKeeper(object):
    def __init__(self):
        self._genotypes = None
        self._phenotypes = None
        self.cur_drug = None
        self.cur_features = None
        self.cur_objects = None

    # .. working with files ..

    def read_snps_matrix(self, filename='data/apr17.snps.matrix'):
        """Read genotypes from file in snps.matrix format."""
        df = pd.read_csv(filename, sep='\t', low_memory=False)\
            .drop([0, 1], axis=0)
        # set strain names as index and convert NA values to '-1'
        df = df.set_index(df.columns.values[0])\
            .replace('-', -1)\
            .astype(np.int32)
        return df

    def read_vcf(self, filename):
        """Read genotypes from file in vcf format."""
        pass

    def load_genotypes(self, filename, format='snps.matrix'):
        """Load genotypes into dataframe."""
        if not format in ['snps.matrix', 'vcf']:
            raise ValueError, "Genotypes must be in one of 'snps.matrix', 'vcf'"

        if format == 'snps.matrix':
            self._genotypes = self.read_snps_matrix(filename)
        else:
            self._genotypes = self.read_vcf(filename)

    def load_phenotypes(self, filename='data/drugs_effect17.csv'):
        """Load DST results into phenotype variable."""
        df = pd.read_csv(filename)
        self._phenotypes = df.set_index(df.columns.values[0])

    # .. get possible drug names ..

    def get_possible_first_level_drugs(self):
        return FIRST_LINE

    def get_possible_second_level_drugs(self):
        return SECOND_LINE

    def get_possible_drugs(self):
        return self.get_possible_first_level_drugs() + self.get_possible_second_level_drugs()

    # .. working with genotypes and phenotypes ..

    def _get_x(self):
        return self._genotypes

    def _set_x(self, new_genotypes):
        self._genotypes = new_genotypes

    def _get_y(self, drug_name):
        """Return DST results for specified drug.

        DST values == 2 converted to TRUE, values == 1 converted to FALSE.
        """
        return self._phenotypes[drug_name].dropna() == 2

    def get_objects_names(self):
        objects_names = list(self._get_x().index)
        return objects_names

    def get_object_name_by_index(self, index):
        return self.get_objects_names()[index]

    def get_feature_names(self):
        return list(self._get_x().columns.values)

    def get_feature_name_by_index(self, index):
        return self.get_feature_names()[index]

    # .. getting (X, y) data for specific drug ..

    def filter(self, data, min_maf=0.03, max_maf=0.5):
        """Filter dataframe with genotypes."""
        if not ((0 <= min_maf < 1) and (0 <= max_maf < 1)):
            raise ValueError, 'MAF filters must be in [0, 1) interval.'

        # convert NAs and undefined symbols to 0
        data[data != 1] = 0
        # Filter options:
        # drop columns with more than half strains mutated
        drop_option1 = (data.sum(axis=0) >= max_maf*data.shape[0])
        # drop columns with less than 'maf_sum' mutated strains
        drop_option2 = (data.sum(axis=0) < min_maf*data.shape[0])
        # apply filter options
        to_drop = drop_option1 | drop_option2

        return data.drop(data.columns[to_drop], axis=1)

    def get_data_for_drug(self, drug_name, as_indexes=False):
        """Extract (X, y) dataset for specified drug."""
        if self._genotypes is None or self._phenotypes is None:
            raise ValueError, "Data not loaded! "

        if not as_indexes:
            # phenotype values will be converted into TRUE/FALSE representation
            data = self._get_x().join(self._get_y(drug_name), how='inner')

            data_x = self.filter(
                data.drop(drug_name, axis=1),
                min_maf=0.03,
                max_maf=0.5
            )
            data_y = data[drug_name]
            data_features = data_x.columns.values
            data_objects = data_y.index

            # update status of the class instance
            self.cur_drug = drug_name
            self.cur_features = data_features
            self.cur_objects = data_objects

            return data_x.as_matrix().astype(np.uint8), \
                   data_y.as_matrix().astype(np.bool), \
                   data_features, \
                   data_objects
        else:
            X, y, features, objects = self.get_data_for_drug(drug_name, as_indexes=False)
            # calculate object indexes in the full dataframe of genotypes
            X_indexes = list()
            possible_objects = set(objects)
            for i, el in enumerate(self.get_objects_names()):
                if el in possible_objects:
                    X_indexes.append(i)

            X_indexes = np.array(X_indexes)
            X_indexes = X_indexes.reshape((X_indexes.shape[0], 1))

            return X_indexes, y, features, objects


###############################################################################
DATA_KEEPER = DataKeeper()


def get_data_keeper():
    return DATA_KEEPER


__all__ = ['get_data_keeper']
