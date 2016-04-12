from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import xgboost as xgb
import argparse
from os import path
import os
from utils import *
import pickle

from sklearn.cross_validation import StratifiedKFold

np.random.seed(888)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


def get_data(args):
    data_dir = '../level4-feature/' + str(args.yix)
    X_train = np.load(path.join(data_dir, 'X_train.npy'))
    y_train = np.load('../feature/1_100/y_train.npy')[:, args.yix]
    X_test = np.load(path.join(data_dir, 'X_test.npy'))

    pic_dir = '../pic-feature-final/' + str(args.yix)
    X_train_pic = np.load(path.join(pic_dir, 'train.npy'))
    X_test_pic = np.load(path.join(pic_dir, 'test.npy'))

    X_train_all = np.hstack((X_train, X_train_pic))
    X_test_all = np.hstack((X_test, X_test_pic))

    return X_train_all, X_test_all, y_train


args = parse_args()
X_train, X_test, y_train = get_data(args)
y_test_init = np.load('../level4-model/' + str(args.yix) + '/pred.npy')
with open('../level4-model/' + str(args.yix) + '/param.pkl', 'rb') as f:
    params = pickle.load(f)

print(X_train.shape, X_test.shape, y_train.shape, y_test_init.shape)
print(params)

y_test_prob = np.copy(y_test_init)
y_test_class = (y_test_init > 0.5).astype(int)

for replication in range(10):
    for train_ix, test_ix in StratifiedKFold(y_test_class,
                                             n_folds=5,
                                             shuffle=True):
        X = np.vstack((X_train, X_test[train_ix, :]))
        y = np.hstack((y_train, y_test_class[train_ix]))
        print(X.shape)
        print(y.shape)
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(params=params,
                        dtrain=dtrain,
                        num_boost_round=params['num_boost_round'])
        dtest = xgb.DMatrix(X_test[test_ix, :])
        pred = bst.predict(dtest)
        y_test_prob[test_ix] = pred
        y_test_class[test_ix] = (pred > 0.5).astype(int)
        print(np.sum((y_test_prob - y_test_init) ** 2))


save_dir = '../semi/' + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

np.save(path.join(save_dir, 'pred.npy'), y_test_prob)
