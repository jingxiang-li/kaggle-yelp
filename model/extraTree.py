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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from utils import *

import logging
import logging.handlers

fmt = ("%(asctime)s - %(filename)s:%(lineno)s - "
       "%(name)s - %(levelname)s - %(message)s")
formatter = logging.Formatter(fmt)

handler = logging.FileHandler("extraTree.log")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.WARNING)

logger = logging.getLogger('extraTree')
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
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])
        params["min_samples_split"] = int(params["min_samples_split"])
        params["min_samples_leaf"] = int(params["min_samples_leaf"])
        params["n_estimators"] = int(params["n_estimators"])

        logger.info("Training with params:")
        logger.info(params)

        # cross validation here
        scores = []
        for train_ix, test_ix in makeKFold(5, self.y, self.reps):
            X_train, y_train = self.X[train_ix, :], self.y[train_ix]
            X_test, y_test = self.X[test_ix, :], self.y[test_ix]
            weight = y_train.shape[0] / (2 * np.bincount(y_train))
            sample_weight = np.array([weight[i] for i in y_train])

            clf = ExtraTreesClassifier(**params)
            cclf = CalibratedClassifierCV(base_estimator=clf,
                                          method='isotonic',
                                          cv=makeKFold(3, y_train, self.reps))
            cclf.fit(X_train, y_train, sample_weight)
            pred = cclf.predict(X_test)
            scores.append(f1_score(y_true=y_test, y_pred=pred))

        print(scores)
        score = np.mean(scores)

        logger.info(score)
        return {'loss': -score, 'status': STATUS_OK}


def optimize(trials, X, y, y_ix, reps, max_evals):
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 250, 50),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth', 1, 6, 1),
        'min_samples_split': hp.quniform('min_samples_split', 1, 9, 2),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        'bootstrap': False,
        'oob_score': False,
        'n_jobs': -1
    }
    s = Score(X, y, y_ix, reps)
    best = fmin(s.get_score,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals
                )
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["min_samples_split"] = int(best["min_samples_split"])
    best["min_samples_leaf"] = int(best["min_samples_leaf"])
    best["n_estimators"] = int(best["n_estimators"])
    best["criterion"] = ['gini', 'entropy'][best["criterion"]]
    del s
    return best


def out_fold_pred(params, X, y_array, y_ix, reps):
    y = y_array[:, y_ix]
    params['bootstrap'] = False
    params['oob_score'] = False
    params['n_jobs'] = -1

    # cross validation here
    preds = np.zeros((y_array.shape[0]))

    for train_ix, test_ix in makeKFold(5, y, reps):
        X_train, y_train = X[train_ix, :], y[train_ix]
        X_test = X[test_ix, :]
        weight = y_train.shape[0] / (2 * np.bincount(y_train))
        sample_weight = np.array([weight[i] for i in y_train])

        clf = ExtraTreesClassifier(**params)
        cclf = CalibratedClassifierCV(base_estimator=clf,
                                      method='isotonic',
                                      cv=makeKFold(3, y_train, reps))
        cclf.fit(X_train, y_train, sample_weight)
        pred = cclf.predict_proba(X_test)[:, 1]
        preds[test_ix] = pred
    return preds


def get_model(params, X, y_array, y_ix, reps):
    y = y_array[:, y_ix]
    params['bootstrap'] = False
    params['oob_score'] = False
    params['n_jobs'] = -1

    clf = ExtraTreesClassifier(**params)
    cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=makeKFold(3, y, reps))
    weight = y.shape[0] / (2 * np.bincount(y))
    sample_weight = np.array([weight[i] for i in y])
    cclf.fit(X, y, sample_weight)
    return cclf


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
        params = optimize(trials, X, y, y_ix, reps, 30)
        preds = out_fold_pred(params, X, y, y_ix, reps)
        model = get_model(params, X, y, y_ix, reps)

        pickle.dump(params,
                    open(path.join(save_dir, "param_" + str(y_ix) + ".pkl"),
                         'wb'))
        np.save(path.join(save_dir, "outFold_" + str(y_ix) + ".npy"), preds)
        pickle.dump(model.calibrated_classifiers_,
                    open(path.join(save_dir, "model_" + str(y_ix) + ".pkl"),
                         'wb'))
        logger.info(str(y_ix) + " completes!")
