# given a raw X, return the level4 prediction

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
from os import path
#import os
#import re
import pickle
import xgboost as xgb
from sklearn.pipeline import make_pipeline, make_union

from level2_data import get_new_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class level2_pred:
    def __init__(self, data, prob, reps, yix):
        self.args = Namespace(data=data, prob=prob, reps=reps, yix=yix)

    def transform(self, X):
        if self.args.data == '21k':
            X = X[:, :2048]
        elif self.args.data == 'colHist':
            X = X[:, 2048:(2048 + 772)]
        else:
            X = X[:, (2048 + 772):6916]

        X_0 = get_new_test(self.args, X)
        lvl2_ftr_dir = '_'.join(
            ('../level2-feature/' + str(self.args.yix) + '/' + self.args.reps,
             self.args.prob, self.args.data))
        ftr_list = np.load(path.join(lvl2_ftr_dir, 'feature_list.npy'))
        X_1 = np.hstack((X_0, X[:, ftr_list]))
        X_input = np.hstack((X_0, X_1))

        lvl2_m_dir = '_'.join(
            ('../level2-models/' + str(self.args.yix) + '/' + self.args.reps,
             self.args.prob, self.args.data))
        with open(path.join(lvl2_m_dir, "model.pkl"), "rb") as f:
            clf = pickle.load(f)

        d_input = xgb.DMatrix(X_input)
        return clf.predict(d_input, output_margin=False)


X = np.load('../feature/1_100/X_train.npy')

global_args = parse_args()
data_ch = ['21k', 'v3', 'colHist']
prob_ch = ['75', '50']
reps_ch = ['5', '9']
lvl2_trans = []
for data in data_ch:
    for prob in prob_ch:
        for reps in reps_ch:
            lvl2_trans.append(level2_pred(data, prob, reps, global_args.yix))
print(lvl2_trans)

lvl2_pred = make_union(lvl2_trans)
print(lvl2_pred.transform(X).shape)
