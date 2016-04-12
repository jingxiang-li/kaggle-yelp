from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import xgboost as xgb
import argparse
from os import path
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from utils import *

np.random.seed(6796)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


# functions for xgboost training
def evalF1(preds, dtrain):
    from sklearn.metrics import f1_score
    labels = dtrain.get_label()
    return 'f1-score', f1_score(labels, preds > 0.5)


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)


# functions for hyperparameters optimization


class Score:
    def __init__(self, X, y):
        self.dtrain = xgb.DMatrix(X, label=y)

    def get_score(self, params):
        params["max_depth"] = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["num_boost_round"] = int(params["num_boost_round"])

        print("Training with params:")
        print(params)

        cv_result = xgb.cv(params=params,
                           dtrain=self.dtrain,
                           num_boost_round=params['num_boost_round'],
                           nfold=5,
                           stratified=True,
                           feval=evalF1,
                           maximize=True,
                           fpreproc=fpreproc,
                           verbose_eval=True)

        score = cv_result.ix[params['num_boost_round'] - 1, 0]
        print(score)
        return {'loss': -score, 'status': STATUS_OK}


def optimize(trials, X, y, max_evals):
    space = {
        'num_boost_round': hp.quniform('num_boost_round', 10, 200, 10),
        'eta': hp.quniform('eta', 0.1, 0.3, 0.1),
        'gamma': hp.quniform('gamma', 0, 1, 0.2),
        'max_depth': hp.quniform('max_depth', 1, 6, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
        'subsample': hp.quniform('subsample', 0.8, 1, 0.1),
        'silent': 1,
        'objective': 'binary:logistic'
    }
    s = Score(X, y)
    best = fmin(s.get_score,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals
                )
    best["max_depth"] = int(best["max_depth"])
    best["min_child_weight"] = int(best["min_child_weight"])
    best["num_boost_round"] = int(best["num_boost_round"])
    del s
    return best


def get_model(params, X, y):
    dtrain = xgb.DMatrix(X, label=y)
    params['silent'] = 1
    params['objective'] = 'binary:logistic'
    params['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)

    bst = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round=params['num_boost_round'],
                    evals=[(dtrain, 'train')],
                    feval=evalF1,
                    maximize=True,
                    verbose_eval=None)
    return bst

# extra features


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


def get_extra_features(args):
    forest = ExtraTreesClassifier(n_estimators=2000,
                                  criterion='entropy',
                                  max_features='sqrt',
                                  max_depth=6,
                                  min_samples_split=8,
                                  n_jobs=-1,
                                  bootstrap=True,
                                  oob_score=True,
                                  verbose=1,
                                  class_weight='balanced')
    pca = PCA(n_components=200)
    ica = FastICA(n_components=200, max_iter=1000)
    kmeans = KMeans(n_clusters=200, n_init=20, max_iter=1000)

    pipeline = make_pipeline(selectKFromModel(forest, k=1000),
                             StandardScaler(),
                             make_union(pca, ica, kmeans))

    X_train = np.load("../feature/1_100/X_train.npy")
    y_train = np.load("../feature/1_100/y_train.npy")
    X_test = np.load("../feature/1_100/X_test.npy")

    pipeline.fit(X_train, y_train[:, args.yix])
    sel_ixs = pipeline.steps[0][1].indices[:400]
    X_train_ext = np.hstack((pipeline.transform(X_train), X_train[:, sel_ixs]))
    X_test_ext = np.hstack((pipeline.transform(X_test), X_test[:, sel_ixs]))
    return X_train_ext, X_test_ext


args = parse_args()
data_dir = "../level3-feature/" + str(args.yix)
X_train = np.load(path.join(data_dir, "X_train.npy"))
X_test = np.load(path.join(data_dir, "X_test.npy"))
y_train = np.load(path.join(data_dir, "y_train.npy"))
print(X_train.shape, X_test.shape, y_train.shape)

X_train_ext, X_test_ext = get_extra_features(args)
print(X_train_ext.shape, X_test_ext.shape)

X_train = np.hstack((X_train, X_train_ext))
X_test = np.hstack((X_test, X_test_ext))
print("Add Extra")
print(X_train.shape, X_test.shape, y_train.shape)

trials = Trials()
params = optimize(trials, X_train, y_train, 100)
clf = get_model(params, X_train, y_train)
dtest = xgb.DMatrix(X_test)
preds = clf.predict(dtest)

save_dir = "../level3-models/" + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

np.save(path.join(save_dir, "pred.npy"), preds)
