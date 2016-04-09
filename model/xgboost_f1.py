from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import xgboost as xgb
import argparse
from os import path, getcwd
import os
import re
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle

from utils import *

import logging
import logging.handlers

fmt = ("%(asctime)s - %(filename)s:%(lineno)s - "
       "%(name)s - %(levelname)s - %(message)s")
formatter = logging.Formatter(fmt)

handler = logging.FileHandler("xgboost_f1.log")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.WARNING)

logger = logging.getLogger('xgboost_f1')
logger.addHandler(handler)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

# logger.debug('This is debug message')
# logger.info('This is info message')
# logger.warning('This is warning message')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="../feature/5_075")
    parser.add_argument('--data_set', choices=['21k', 'v3', 'colHist'],
                        default="21k")
    return parser.parse_args()


def get_params(args):
    data_dir_path = path.abspath(path.join(getcwd(), args.data_folder))
    assert path.exists(data_dir_path)
    dir_name = path.basename(data_dir_path)
    m = re.match(r"(\d+)_(\d+)", dir_name)
    reps, prob = int(m.group(1)), int(m.group(2)) / 100
    return data_dir_path, reps, prob


def get_data_train(data_dir, args):
    X_path = path.join(data_dir, "X_train.npy")
    y_path = path.join(data_dir, "y_train.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    if args.data_set == '21k':
        sel_range = range(0, 2048)
    elif args.data_set == 'colHist':
        sel_range = range(2048, 2048 + 772)
    else:
        sel_range = range(2048 + 772, 6916)
    return (X[:, sel_range], y)


########################################################################

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

# functions for hyperparameters optimization


class Score:
    def __init__(self, X, y, y_ix, reps):
        self.y = y[:, y_ix]
        self.dtrain = xgb.DMatrix(X, label=self.y)
        self.reps = reps

    def get_score(self, params):
        params["max_depth"] = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        params["num_boost_round"] = int(params["num_boost_round"])

        logger.info("Training with params:")
        logger.info(params)

        cv_result = xgb.cv(params=params,
                           dtrain=self.dtrain,
                           num_boost_round=params['num_boost_round'],
                           nfold=5,
                           folds=makeKFold(5, self.y, self.reps),
                           feval=evalF1,
                           maximize=True,
                           fpreproc=fpreproc,
                           verbose_eval=True)

        score = cv_result.ix[params['num_boost_round'] - 1, 0]
        logger.info(score)
        return {'loss': -score, 'status': STATUS_OK}


def optimize(trials, X, y, y_ix, reps, max_evals):
    space = {
        'num_boost_round': hp.quniform('num_boost_round', 1, 2, 1),
        'eta': hp.quniform('eta', 0.1, 0.5, 0.1),
        'gamma': hp.quniform('gamma', 0, 1, 0.2),
        'max_depth': hp.quniform('max_depth', 1, 6, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 7, 2),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
        'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.1),
        'lambda': hp.loguniform('lambda', -0.7, 1.7),
        'silent': 1,
        'objective': 'binary:logistic'
    }
    s = Score(X, y, y_ix, reps)
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


def out_fold_pred(params, X, y_array, y_ix, reps):
    preds = np.zeros((y_array.shape[0]))
    y = y_array[:, y_ix]
    params['silent'] = 1
    params['objective'] = 'binary:logistic'
    params['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)
    for train_ix, test_ix in makeKFold(5, y, reps):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train = y[train_ix]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(params=params,
                        dtrain=dtrain,
                        num_boost_round=params['num_boost_round'],
                        evals=[(dtrain, 'train')],
                        feval=evalF1,
                        maximize=True,
                        verbose_eval=None)
        preds[test_ix] = bst.predict(dtest, output_margin=True)
    return preds


def get_model(params, X, y_array, y_ix, reps):
    y = y_array[:, y_ix]
    dtrain = xgb.DMatrix(X, label=y)
    params['silent'] = 1
    params['objective'] = 'binary:logistic'
    params['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)

    bst = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round=params['num_boost_round'],
                    evals=[(dtrain, 'train')],
                    feval=evalF1,
                    maximize=True)
    return bst

if __name__ == "__main__":
    args = parse_args()
    data_dir, reps, prob = get_params(args)
    X, y = get_data_train(data_dir, args)
    trials = Trials()
    params = optimize(trials, X, y, 0, reps, 2)
    preds = out_fold_pred(params, X, y, 0, reps)
    model = get_model(params, X, y, 0, reps)

    save_dir = path.join(path.dirname(path.abspath(__file__)),
                         path.basename(__file__)[:-3])
    if not path.isdir(save_dir):
        os.mkdir(save_dir)

    np.save(path.join(save_dir, "outFold.npy"), preds)
    pickle.dump(model, open(path.join(save_dir, "model.pkl"), mode='wb'))
