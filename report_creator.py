import sys
from os import mkdir, listdir
from os.path import isfile, isdir, join
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from common import SAVE_RESULTS_AFTER
import numpy as np
import pandas as pd
import cPickle

###
import testing
import pylab
import json
import re


#####################################################


class ReportCreator(object):
    def __init__(self):
        self._folder = "experiments_results"


    def _get_folder(self):
        return self._folder



    def set_subdir(self, subdir):
        self._folder = join(self._folder, subdir)


    def getModelFeatures(self, experiment_name, drug_name):
        self._experiment_name = experiment_name
        self._drug_name = drug_name

        self.set_subdir(experiment_name + '(' + drug_name + ')')

        f1_scores = []
        accuracy_scores = []

        res_folds_dicts = []
        res_feature_tables = []


        for fold_name in range(5):
            folder = self._get_folder() + '/' + str(fold_name) + '/'
            #folder = '/home/roma/GWAS-tools/gwas-tools/experiments_results/2017/frn_model(PARA Para-aminosalicyclic acid )' + '/' + str(fold_name) + '/'

            if not isfile(join(folder, "final_model_features0.csv")):
                continue

            ## read info from "final_metrics.txt"

            filename = join(folder, "final_metrics.txt")
            if isfile(filename):
                with open(filename, 'r') as f:
                    content = f.read()

                ## read f1_score for the model

                regexp = re.compile("f1_score (\d+\.\d+)")
                matched_content = regexp.findall(content)[0]

                f1_score = float(matched_content)
                f1_scores.append(f1_score)

                ## read accuracy for the model

                regexp = re.compile("accuracy (\d+\.\d+)")
                matched_content = regexp.findall(content)[0]

                accuracy = float(matched_content)
                accuracy_scores.append(accuracy)

                ## read test predictions

                regexp = re.compile(", array(.*)(?=dtype=bool)", re.DOTALL)
                matched_content = regexp.findall(content)[0]
                test_pred_str = re.findall('True|False', matched_content)

                test_pred = np.array([w=='True' for w in test_pred_str])


            ## get true data from training subset

            file_list = []
            for file in listdir(folder):
                if file.endswith(".pkl"):
                    try:
                        cur_id = int(file[:-4])
                        file_list.append(cur_id)
                    except ValueError:
                        pass

            file_list.sort()

            for file_id in file_list:
                filename = join(folder, "{}.pkl".format(file_id))

                if isfile(filename):
                    objects = joblib.load(filename)

                    try:
                        test_true_init = objects[0]['test_true']
                        test_true_check = False
                        for obj in objects:
                            test_true_check = test_true_check or any(np.logical_xor(test_true_init, obj['test_true']))

                        if not test_true_check:

                            # true test correclty extracted from fold, let's print and calculate confusion matrix
                            test_true = test_true_init

                            prob_true = sum(test_true) / float(len(test_true))
                            prob_false = 1. - prob_true
                    except IndexError:
                        pass


            fold_dict = {}

            fold_dict['test_true'] = test_true
            fold_dict['test_pred'] = test_pred

            fold_dict['prob_true'] = prob_true
            fold_dict['prob_false'] = prob_false


            ##  calculate confusion matrix

            #conf_matr = confusion_matrix(test_true, test_pred, labels=[True, False])

            conf_matr = confusion_matrix(test_true, test_pred)

            tn, fp, fn, tp = confusion_matrix(test_true, test_pred).ravel()

            fold_dict['confusion_matrix'] = conf_matr

            fold_dict[' tn'] = tn
            fold_dict[' fp'] = fp
            fold_dict[' fn'] = fn
            fold_dict[' tp'] = tp


            ## read features and weights

            filename = join(folder, "final_model_features0.csv")

            df = pd.DataFrame.from_csv(filename)

            fold_dict['best_model'] = list(df.columns.values)[0]
            df.columns = ['importance', 'pos']


            ## normalize importance values

            max_imp = df['importance'].max()
            df['importance'] = f1_score * df['importance'] / float(max_imp)

            res_feature_tables.append(df)
            res_folds_dicts.append(fold_dict)


        ## calculate features and weights for all experiment

        res_df = res_feature_tables[0]
        for i in range(1, len(res_feature_tables)):
            res_df = pd.merge(res_df, res_feature_tables[i], how='outer', on=['pos'])

        res_df = res_df.fillna(0)
        res_df.index = res_df['pos']
        res_df.drop('pos', axis=1, inplace=True)

        res_df = pd.DataFrame(data = res_df.sum(axis=1), columns=['importance'])
        res_df = res_df.sort(columns=['importance'], ascending=False)



        ## write features and weights into file

        folder = self._get_folder()

        res_df.to_csv(join(folder, "all_features.csv"))

        f = open(join(folder, "all_statistics.txt"), 'w')

        for fold_name in range(len(f1_scores)):

            str_head = "Fold " + str(fold_name) + ":\n"
            f.write(str_head)

            f.write('f1_score: ' + str(f1_scores[fold_name]) + '\n')
            f.write('accuracy: ' + str(accuracy_scores[fold_name]) + '\n')

            for i in fold_dict.keys():
                f.write(i + ": " + "".join(str(fold_dict[i])) + "\n")


            f.write('======================================================\n')



        f.close()  # you can omit in most cases as the destructor will call it

        self._folder = "experiments_results"






rc = ReportCreator()

##selector_model(PARA Para-aminosalicyclic acid )

rc.getModelFeatures('selector_model', 'PARA Para-aminosalicyclic acid ')

rc.getModelFeatures('selector_model', 'OFLO Ofloxacin ')

rc.getModelFeatures('selector_model', 'Levofloxacin ')

rc.getModelFeatures('selector_model', 'KANA Kanamycin ')

rc.getModelFeatures('selector_model', 'ETHI Ethionamide Prothionamide ')

rc.getModelFeatures('selector_model', 'CAPR Capreomycin ')

rc.getModelFeatures('selector_model', 'AMIK Amikacin ')




from Bio import SeqIO
import pandas as pd

gb_file = "/home/roma/Documents/CurrentWork/R_scripts/snpEff_latest_core/snpEff/data/m_tuberculosis_H37Rv_Broad/genes.gbk"

for gb_record in SeqIO.parse(open(gb_file,"r"), "genbank") :
    # now do something with the record
    print "Name %s, %i features" % (gb_record.name, len(gb_record.features))
    print repr(gb_record.seq)


features_count = len(gb_record.features)

tags = []
names = []
for i in range(features_count):
    gb_feature = gb_record.features[i]

    if ('locus_tag' in gb_feature.qualifiers.keys()) and ('product' in gb_feature.qualifiers.keys()):
        tags.append(gb_feature.qualifiers['locus_tag'])
        names.append(gb_feature.qualifiers['product'])

ftags = [val for sublist in tags for val in sublist]
fnames = [val for sublist in names for val in sublist]


writer = pd.ExcelWriter('simple-report.xlsx', engine='xlsxwriter')
df.to_excel(writer, index=False)
writer.save()

