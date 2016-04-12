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
        result = clf.predict(d_input, output_margin=False)
        return result.reshape((result.size, 1))


class selectKFromModel:
    def __init__(self, estimator, k, prefit=False):
        self.estimator = estimator
        self.prefit = prefit
        self.k = k

    def fit(self, X, y=None, **fit_params):
        if self.prefit is False:
            self.estimator.fit(X, y, **fit_params)
        self.importances = self.estimator.feature_importances_
        self.indices = np.argsort(self.importances)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.indices]


class level3_extra:
    def __init__(self, yix):
        self.args = Namespace(yix=yix)

    def transform(self, X):
        with open('../extra_ftrs/' + str(self.args.yix) +
                  '/pipe.pkl', 'rb') as f_pipe:
            pipeline = pickle.load(f_pipe)
        sel_ix = np.load('../extra_ftrs/' + str(self.args.yix) + '/selix.npy')
        print(pipeline)
        print(sel_ix)
        X_0 = pipeline.transform(X)
        X_1 = X[:, sel_ix]
        return np.hstack((X_0, X_1))


class level3_pred:
    def __init__(self, yix):
        self.args = Namespace(yix=yix)

    def transform(self, X):
        with open('../level3-model-final/' + str(self.args.yix) +
                  '/model.pkl', 'rb') as f:
            bst = pickle.load(f)

        with open('../level3-model-final/' + str(self.args.yix) +
                  '/model_rf.pkl', 'rb') as f:
            rf = pickle.load(f)

        with open('../level3-model-final/' + str(self.args.yix) +
                  '/model_ext.pkl', 'rb') as f:
            ext = pickle.load(f)

        d_X = xgb.DMatrix(X)
        pred_bst = bst.predict(d_X)

        pred_list_rf = []
        for clf in rf:
            pred_list_rf.append(clf.predict_proba(X)[:, 1])
        pred_rf = np.mean(np.array(pred_list_rf), axis=0)

        pred_list_ext = []
        for clf in ext:
            pred_list_ext.append(clf.predict_proba(X)[:, 1])
        pred_ext = np.mean(np.array(pred_list_ext), axis=0)

        return np.vstack((
            pred_bst,
            pred_rf,
            pred_ext)).T

X = np.load('../feature/1_100/X_train.npy')

global_args = parse_args()
data_ch = ['21k', 'colHist', 'v3']
prob_ch = ['50', '75']
reps_ch = ['5', '9']
lvl2_trans = []
for reps in reps_ch:
    for prob in prob_ch:
        for data in data_ch:
            print(reps, prob, data)
            lvl2_trans.append(level2_pred(data, prob, reps, global_args.yix))
print(lvl2_trans)

lvl2_pred = make_union(*lvl2_trans)
X_lvl2_pred = lvl2_pred.transform(X)
X_lvl3_extra = level3_extra(global_args.yix).transform(X)
X_lvl3_pred = level3_pred(global_args.yix).transform(
    np.hstack((X_lvl2_pred, X_lvl3_extra)))

print(X_lvl2_pred.shape)
print(X_lvl3_extra.shape)
print(X_lvl3_pred.shape)
