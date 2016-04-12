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
