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

np.random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()

# functions for xgboost training


def evalF1(preds, dtrain):
    from sklearn.metrics import f1_score
    labels = dtrain.get_label()
    y_agg = labels[range(0, labels.shape[0], reps)]
    pred_agg = agg_preds(preds, reps, vote_by_majority)
    return 'f1-score', f1_score(y_agg, pred_agg)


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)


class Score:
    def __init__(self, X, y):
        self.dtrain = xgb.DMatrix(X, label=y)

    def get_score(self, params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['num_boost_round'] = int(params['num_boost_round'])

        print('Training with params:')
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
        'num_boost_round': hp.quniform('num_boost_round', 10, 150, 10),
        'eta': hp.quniform('eta', 0.1, 0.3, 0.1),
        'gamma': hp.quniform('gamma', 0, 1, 0.2),
        'max_depth': hp.quniform('max_depth', 1, 8, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
        'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.1),
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
    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])
    best['num_boost_round'] = int(best['num_boost_round'])
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

d_test = xgb.DMatrix(X_test_all)
preds = model.predict(d_test)

save_dir = '../level4-model/' + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

with open(path.join(save_dir, 'model.pkl'), 'wb') as f_model:
    pickle.dump(model, f_model)

with open(path.join(save_dir, 'param.pkl'), 'wb') as f_param:
    pickle.dump(params, f_param)

np.save(path.join(save_dir, 'pred.npy'), preds)
