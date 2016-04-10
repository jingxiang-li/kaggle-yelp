from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
from os import path, getcwd
import os
import re
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

from utils import *

import logging
import logging.handlers

fmt = ("%(asctime)s - %(filename)s:%(lineno)s - "
       "%(name)s - %(levelname)s - %(message)s")
formatter = logging.Formatter(fmt)

handler = logging.FileHandler("glm_l1.log")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.WARNING)

logger = logging.getLogger('glm_l1')
logger.addHandler(handler)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

# logger.debug('This is debug message')
# logger.info('This is info message')
# logger.warning('This is warning message')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="../feature/5_75")
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

# functions for hyperparameters optimization

class Score:
    def __init__(self, X, y, y_ix, reps):
        self.y = y[:, y_ix]
        self.X = X
        self.reps = reps

    def get_score(self, params):
        logger.info("Training with params:")
        params['max_iter'] = int(params['max_iter'])
        logger.info(params)

        # cross validation here
        clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
        score_list = cross_val_score(estimator=clf,
                                     X=self.X,
                                     y=self.y,
                                     scoring='f1',
                                     cv=makeKFold(5, self.y, self.reps),
                                     n_jobs=-1,
                                     verbose=1)
        score = np.mean(score_list)
        logger.info(score)
        return {'loss': -score, 'status': STATUS_OK}


def optimize(trials, X, y, y_ix, reps, max_evals):
    space = {
        'C': hp.loguniform('C', -3, 2),
        'max_iter': hp.quniform('max_iter', 100, 300, 50),
        'n_jobs': -1,
        'class_weight': 'balanced',
        'penalty': 'l1'
    }
    s = Score(X, y, y_ix, reps)
    best = fmin(s.get_score,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals
                )
    best['max_iter'] = int(best['max_iter'])
    best['n_jobs'] = -1
    best['class_weight'] = 'balanced'
    best['penalty'] = 'l1'
    del s
    return best


def out_fold_pred(params, X, y_array, y_ix, reps):
    y = y_array[:, y_ix]

    # cross validation here
    preds = np.zeros((y_array.shape[0]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(**params))

    for train_ix, test_ix in makeKFold(5, y, reps):
        X_train, y_train = X[train_ix, :], y[train_ix]
        X_test = X[test_ix, :]
        clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:, 1]
        preds[test_ix] = pred
    return preds


def get_model(params, X, y_array, y_ix, reps):
    y = y_array[:, y_ix]
    clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
    clf.fit(X, y)
    return clf


if __name__ == "__main__":
    args = parse_args()

    logger.info("Data Directory: " + args.data_folder)
    logger.info("Data set: " + args.data_set)

    data_dir, reps, prob = get_params(args)
    X, y = get_data_train(data_dir, args)

    # save place
    save_dir_name = path.basename(__file__)[:-3] + "_" + \
        str(reps) + "_" + str(int(100 * prob)) + "_" + args.data_set
    save_dir = path.join(path.dirname(path.abspath(__file__)),
                         save_dir_name)
    if not path.isdir(save_dir):
        os.mkdir(save_dir)

    # begin trainnig
    for y_ix in range(9):
        logger.info("training for class " + str(y_ix))
        trials = Trials()
        params = optimize(trials, X, y, y_ix, reps, 15)
        preds = out_fold_pred(params, X, y, y_ix, reps)
        model = get_model(params, X, y, y_ix, reps)

        pickle.dump(params,
                    open(path.join(save_dir, "param_" + str(y_ix) + ".pkl"),
                         'wb'))
        np.save(path.join(save_dir, "outFold_" + str(y_ix) + ".npy"), preds)
        pickle.dump(model,
                    open(path.join(save_dir, "model_" + str(y_ix) + ".pkl"),
                         'wb'))
        logger.info(str(y_ix) + " completes!")
