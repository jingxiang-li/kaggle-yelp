# get level-1 model for each dataset, for example the averaged model
# for data 5-57-21k, which includes predictions from xgboost,
# random forest ..., projected features from pca, ica, and selected features
# from some models, the model should be able to give prediction score
# for each biz_id, used for level-2 model

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from os import path
import os
import pickle
import argparse

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sets import Set
from utils import *

np.random.seed(123213)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        choices=['21k', 'v3', 'colHist'],
                        default="21k")
    parser.add_argument('--prob',
                        choices=['75', '50'],
                        default='75')
    parser.add_argument('--reps',
                        choices=['5', '9'],
                        default='5')
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


def get_data_train(args):
    data_dir = path.join("../feature/", args.reps + "_" + args.prob)
    if args.data == '21k':
        X_path = path.join(data_dir, "X_train.npy")
        y_path = path.join(data_dir, "y_train.npy")
        X = np.load(X_path)
        y = np.load(y_path)
        sel_range = range(0, 2048)
        return X[:, sel_range], y[:, args.yix]
    elif args.data == 'colHist':
        X_path = path.join(data_dir, "X_train.npy")
        y_path = path.join(data_dir, "y_train.npy")
        X = np.load(X_path)
        y = np.load(y_path)
        sel_range = range(2048, 2048 + 772)
        return X[:, sel_range], y[:, args.yix]
    elif args.data == 'v3':
        X_path = path.join(data_dir, "X_train.npy")
        y_path = path.join(data_dir, "y_train.npy")
        X = np.load(X_path)
        y = np.load(y_path)
        sel_range = range(2048 + 772, 6916)
        return X[:, sel_range], y[:, args.yix]


def get_data_test(args):
    data_dir = path.join("../feature/", args.reps + "_" + args.prob)
    if args.data == '21k':
        X_path = path.join(data_dir, "X_test.npy")
        biz_path = path.join(data_dir, "bizlist.npy")
        X = np.load(X_path)
        biz_list = np.load(biz_path)
        sel_range = range(0, 2048)
        return X[:, sel_range], biz_list
    elif args.data == 'colHist':
        X_path = path.join(data_dir, "X_test.npy")
        biz_path = path.join(data_dir, "bizlist.npy")
        X = np.load(X_path)
        biz_list = np.load(biz_path)
        sel_range = range(2048, 2048 + 772)
        return X[:, sel_range], biz_list
    elif args.data == 'v3':
        X_path = path.join(data_dir, "X_test.npy")
        biz_path = path.join(data_dir, "bizlist.npy")
        X = np.load(X_path)
        biz_list = np.load(biz_path)
        sel_range = range(2048 + 772, 6916)
        return X[:, sel_range], biz_list


class level1_model:
    def get_fitted(self):
        pass

    def get_feature_list(self, X_train=None, y_train=None):
        pass

    def predict(self, X_test):
        pass


class level1_xgboost(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("xgboost_f1", args.reps, args.prob, args.data))
        model_name = "model_" + str(args.yix) + ".pkl"
        outfold_name = "outFold_" + str(args.yix) + ".npy"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return np.array([self.outfold_pred]).T

    def get_feature_list(self, X_train=None, y_train=None):
        f_score = self.clf.get_fscore()
        k = np.array(f_score.keys())
        v = np.array(f_score.values())
        f_ix = k[v > np.mean(v)]
        f_list = map(lambda x: int(x[1:]), f_ix)
        return {args.data: f_list}

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return np.array([self.clf.predict(dtest, output_margin=True)]).T


class level1_rf(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("randomForest", args.reps, args.prob, args.data))
        model_name = "model_" + str(args.yix) + ".pkl"
        outfold_name = "outFold_" + str(args.yix) + ".npy"
        param_name = "param_" + str(args.yix) + ".pkl"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)
        param_path = path.join(path.dirname(__file__), model_dir, param_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        with open(param_path, 'r') as f:
            self.params = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return np.array([self.outfold_pred]).T

    def get_feature_list(self, X_train, y_train):
        weight = y_train.shape[0] / (2 * np.bincount(y_train))
        sample_weight = np.array([weight[i] for i in y_train])
        clf = RandomForestClassifier(**self.params)
        clf.fit(X_train, y_train, sample_weight)
        score = np.array(clf.feature_importances_)
        sel_ix = np.arange(score.shape[0])[score > np.mean(score)]
        return {args.data: list(sel_ix)}

    def predict(self, X_test):
        pred_list = []
        for cclf in self.clf:
            pred = cclf.predict_proba(X_test)[:, 1]
            pred_list.append(pred)
        return np.array([np.mean(np.array(pred_list), axis=0)]).T


class level1_extree(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("extraTree", args.reps, args.prob, args.data))
        model_name = "model_" + str(args.yix) + ".pkl"
        outfold_name = "outFold_" + str(args.yix) + ".npy"
        param_name = "param_" + str(args.yix) + ".pkl"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)
        param_path = path.join(path.dirname(__file__), model_dir, param_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        with open(param_path, 'r') as f:
            self.params = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return np.array([self.outfold_pred]).T

    def get_feature_list(self, X_train, y_train):
        weight = y_train.shape[0] / (2 * np.bincount(y_train))
        sample_weight = np.array([weight[i] for i in y_train])
        clf = ExtraTreesClassifier(**self.params)
        clf.fit(X_train, y_train, sample_weight)
        score = np.array(clf.feature_importances_)
        sel_ix = np.arange(score.shape[0])[score > np.mean(score)]
        return {args.data: list(sel_ix)}

    def predict(self, X_test):
        pred_list = []
        for cclf in self.clf:
            pred = cclf.predict_proba(X_test)[:, 1]
            pred_list.append(pred)
        return np.array([np.mean(np.array(pred_list), axis=0)]).T


class level1_pca(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("pca", args.reps, args.prob, args.data))
        model_name = "model.pkl"
        outfold_name = "outFold.npy"
        param_name = "param.pkl"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)
        param_path = path.join(path.dirname(__file__), model_dir, param_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        with open(param_path, 'r') as f:
            self.params = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return self.outfold_pred

    def get_feature_list(self, X_train=None, y_train=None):
        return {args.data: []}

    def predict(self, X_test):
        return self.clf.transform(X_test)


class level1_ica(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("ica", args.reps, args.prob, args.data))
        model_name = "model.pkl"
        outfold_name = "outFold.npy"
        param_name = "param.pkl"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)
        param_path = path.join(path.dirname(__file__), model_dir, param_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        with open(param_path, 'r') as f:
            self.params = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return self.outfold_pred

    def get_feature_list(self, X_train=None, y_train=None):
        return {args.data: []}

    def predict(self, X_test):
        return self.clf.transform(X_test)


class level1_glm(level1_model):
    def __init__(self, args):
        self.args = args
        model_dir = "_".join(("glm_l1", args.reps, args.prob, args.data))
        model_name = "model_" + str(args.yix) + ".pkl"
        outfold_name = "outFold_" + str(args.yix) + ".npy"
        param_name = "param_" + str(args.yix) + ".pkl"

        model_path = path.join(path.dirname(__file__), model_dir, model_name)
        outfold_path = path.join(
            path.dirname(__file__), model_dir, outfold_name)
        param_path = path.join(path.dirname(__file__), model_dir, param_name)

        with open(model_path, 'r') as f:
            self.clf = pickle.load(f)
        with open(param_path, 'r') as f:
            self.params = pickle.load(f)
        self.outfold_pred = np.load(outfold_path)

    def get_fitted(self):
        return np.array([self.outfold_pred]).T

    def get_feature_list(self, X_train=None, y_train=None):
        sel_ix = SelectFromModel(self.clf.steps[1][1],
                                 threshold="4 * mean",
                                 prefit=True).get_support(True)
        return {args.data: list(sel_ix)}

    def predict(self, X_test):
        return np.array([self.clf.predict_proba(X_test)[:, 1]]).T


def get_new_train(args, X):
    models = [level1_xgboost(args), level1_rf(args), level1_extree(args),
              level1_glm(args), level1_pca(args), level1_ica(args)]
    X_new = None
    for model in models:
        fitted = model.get_fitted()
        if X_new is None:
            X_new = fitted
        else:
            X_new = np.concatenate((X_new, fitted), axis=1)
    return X_new


def get_new_test(args, X):
    models = [level1_xgboost(args), level1_rf(args), level1_extree(args),
              level1_glm(args), level1_pca(args), level1_ica(args)]
    X_new = None
    for model in models:
        preds = model.predict(X)
        if X_new is None:
            X_new = preds
        else:
            X_new = np.concatenate((X_new, preds), axis=1)
    return X_new


def get_sel_features(args, X_train, y_train):
    models = [level1_rf(args), level1_extree(args), level1_glm(args)]
    feature_list = Set()
    for model in models:
        feature_list.update(
            model.get_feature_list(X_train, y_train).values()[0])
    return sorted(feature_list)


args = parse_args()
X_train, y_train = get_data_train(args)
X_test, biz_list = get_data_test(args)

X_train_new = get_new_train(args, X_train)
X_test_new = get_new_test(args, X_test)
feature_list = get_sel_features(args, X_train, y_train)
print(args.data)
print(feature_list)

X_train_new = np.hstack((X_train_new, X_train[:, feature_list]))
X_test_new = np.hstack((X_test_new, X_test[:, feature_list]))

save_dir = "_".join(("../level2-feature/" + str(args.yix) + "/" +
                     args.reps, args.prob, args.data))

if not path.exists(save_dir):
    os.makedirs(save_dir)
np.save(path.join(save_dir, "X_train.npy"), X_train_new)
np.save(path.join(save_dir, "y_train.npy"), y_train)
np.save(path.join(save_dir, "X_test.npy"), X_test_new)
np.save(path.join(save_dir, "feature_list.npy"), feature_list)
