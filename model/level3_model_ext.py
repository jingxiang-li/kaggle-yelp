from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

import argparse
from os import path
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils import *

np.random.seed(8089)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


# functions for hyperparameters optimization
class Score:
    def __init__(self, X, y):
        self.y = y
        self.X = X

    def get_score(self, params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        params['n_estimators'] = int(params['n_estimators'])

        print('Training with params:')
        print(params)

        # cross validation here
        scores = []
        for train_ix, test_ix in makeKFold(5, self.y, 1):
            X_train, y_train = self.X[train_ix, :], self.y[train_ix]
            X_test, y_test = self.X[test_ix, :], self.y[test_ix]
            weight = y_train.shape[0] / (2 * np.bincount(y_train))
            sample_weight = np.array([weight[i] for i in y_train])

            clf = ExtraTreesClassifier(**params)
            cclf = CalibratedClassifierCV(base_estimator=clf,
                                          method='isotonic',
                                          cv=makeKFold(3, y_train, 1))
            cclf.fit(X_train, y_train, sample_weight)
            pred = cclf.predict(X_test)
            scores.append(f1_score(y_true=y_test, y_pred=pred))

        print(scores)
        score = np.mean(scores)

        print(score)
        return {'loss': -score, 'status': STATUS_OK}


def optimize(trials, X, y, max_evals):
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 600, 50),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth', 1, 7, 1),
        'min_samples_split': hp.quniform('min_samples_split', 1, 9, 2),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1
    }
    s = Score(X, y)
    best = fmin(s.get_score,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals
                )
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    best['min_samples_split'] = int(best['min_samples_split'])
    best['min_samples_leaf'] = int(best['min_samples_leaf'])
    best['n_estimators'] = int(best['n_estimators'])
    best['criterion'] = ['gini', 'entropy'][best['criterion']]
    best['bootstrap'] = True
    best['oob_score'] = True
    best['n_jobs'] = -1
    del s
    return best


def out_fold_pred(params, X, y):
    # cross validation here
    preds = np.zeros((y.shape[0]))

    for train_ix, test_ix in makeKFold(5, y, 1):
        X_train, y_train = X[train_ix, :], y[train_ix]
        X_test = X[test_ix, :]
        weight = y_train.shape[0] / (2 * np.bincount(y_train))
        sample_weight = np.array([weight[i] for i in y_train])

        clf = ExtraTreesClassifier(**params)
        cclf = CalibratedClassifierCV(base_estimator=clf,
                                      method='isotonic',
                                      cv=makeKFold(3, y_train, 1))
        cclf.fit(X_train, y_train, sample_weight)
        pred = cclf.predict_proba(X_test)[:, 1]
        preds[test_ix] = pred
    return preds


def get_model(params, X, y):
    clf = ExtraTreesClassifier(**params)
    cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=makeKFold(3, y, 1))
    weight = y.shape[0] / (2 * np.bincount(y))
    sample_weight = np.array([weight[i] for i in y])
    cclf.fit(X, y, sample_weight)
    return cclf


args = parse_args()
data_dir = '../level3-feature/' + str(args.yix)
X_train = np.load(path.join(data_dir, 'X_train.npy'))
X_test = np.load(path.join(data_dir, 'X_test.npy'))
y_train = np.load(path.join(data_dir, 'y_train.npy'))
print(X_train.shape, X_test.shape, y_train.shape)

X_train_ext = np.load('../extra_ftrs/' + str(args.yix) + '/X_train_ext.npy')
X_test_ext = np.load('../extra_ftrs/' + str(args.yix) + '/X_test_ext.npy')
print(X_train_ext.shape, X_test_ext.shape)

X_train = np.hstack((X_train, X_train_ext))
X_test = np.hstack((X_test, X_test_ext))
print('Add Extra')
print(X_train.shape, X_test.shape, y_train.shape)


# Now we have X_train, X_test, y_train
trials = Trials()
params = optimize(trials, X_train, y_train, 80)
out_fold = out_fold_pred(params, X_train, y_train)
clf = get_model(params, X_train, y_train)
preds = clf.predict_proba(X_test)[:, 1]

save_dir = '../level3-model-final/' + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

# save model, parameter, outFold_pred, pred
with open(path.join(save_dir, 'model_ext.pkl'), 'wb') as f_model:
    pickle.dump(clf.calibrated_classifiers_, f_model)

with open(path.join(save_dir, 'param_ext.pkl'), 'wb') as f_param:
    pickle.dump(params, f_param)

np.save(path.join(save_dir, 'pred_ext.npy'), preds)
np.save(path.join(save_dir, 'outFold_ext.npy'), out_fold)
