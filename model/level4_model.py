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
from utils import *
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold

np.random.seed(545648)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


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


class Score:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_score(self, params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])

        k_val = params.pop('k')
        print('Training with params:')
        print(params)

        forest = ExtraTreesClassifier(n_estimators=1000,
                                      criterion='entropy',
                                      max_features='sqrt',
                                      max_depth=6,
                                      min_samples_split=8,
                                      n_jobs=-1,
                                      bootstrap=True,
                                      oob_score=True,
                                      verbose=1,
                                      class_weight='balanced')

        clf = xgb.XGBClassifier(**params)
        pipeline = make_pipeline(selectKFromModel(forest, k=k_val), clf)
        score = cross_val_score(estimator=pipeline,
                                X=self.X,
                                y=self.y,
                                scoring='f1',
                                cv=StratifiedKFold(y, 5, True),
                                n_jobs=-1,
                                verbose=1)
        print(score)
        return {'loss': -np.mean(score), 'status': STATUS_OK}


def optimize(trials, X, y, max_evals):
    space = {
        'k': hp.quniform('k', 200, 1000, 100),
        'n_estimators': hp.quniform('n_estimators', 10, 100, 10),
        'learning_rate': hp.quniform('learning_rate', 0.1, 0.3, 0.1),
        'gamma': hp.quniform('gamma', 0, 1, 0.2),
        'max_depth': hp.quniform('max_depth', 1, 6, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
        'subsample': hp.quniform('subsample', 0.8, 1, 0.1),
        'silent': 1,
        'objective': 'binary:logistic',
        'scale_pos_weight': float(np.sum(y == 0)) / np.sum(y == 1)
    }
    s = Score(X, y)
    best = fmin(s.get_score,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals
                )
    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])
    best['n_estimators'] = int(best['n_estimators'])
    best['silent'] = 1
    best['objective'] = 'binary:logistic'
    best['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)
    del s
    return best


def get_model(params, X, y):
    print('best params')
    print(params)

    k_val = params.pop('k')
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  criterion='entropy',
                                  max_features='sqrt',
                                  max_depth=6,
                                  min_samples_split=8,
                                  n_jobs=-1,
                                  bootstrap=True,
                                  oob_score=True,
                                  verbose=1,
                                  class_weight='balanced')

    clf = xgb.XGBClassifier(**params)
    pipeline = make_pipeline(selectKFromModel(forest, k=k_val), clf)
    pipeline.fit(X, y)
    return pipeline


args = parse_args()
data_dir = '../level4-feature/' + str(args.yix)
X_train = np.load(path.join(data_dir, 'X_train.npy'))
y_train = np.load('../feature/1_100/y_train.npy')[:, args.yix]
X_test = np.load(path.join(data_dir, 'X_test.npy'))

pic_dir = '../pic-feature-final/' + str(args.yix)
X_train_pic = np.load(path.join(pic_dir, 'train.npy'))
X_test_pic = np.load(path.join(pic_dir, 'test.npy'))

X_train_all = np.hstack((X_train, X_train_pic))
X_test_all = np.hstack((X_test, X_test_pic))

print(X_train_all.shape, X_test_all.shape)

trials = Trials()
params = optimize(trials, X_train_all, y_train, 100)
model = get_model(params, X_train_all, y_train)
preds = model.predict_proba(X_test_all)[:, 1]

save_dir = '../level4-model/' + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

with open(path.join(save_dir, 'model.pkl'), 'wb') as f_model:
    pickle.dump(model, f_model)

with open(path.join(save_dir, 'param.pkl'), 'wb') as f_param:
    pickle.dump(params, f_param)

np.save(path.join(save_dir, 'pred.npy'), preds)
