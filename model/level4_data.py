from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from os import path, listdir
import os
import pickle as pkl
import argparse
import re

import numpy as np
import xgboost as xgb
from scipy.special import expit

from utils import *

np.random.seed(998)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


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

model_dir = '../level3-model-final/' + str(args.yix)
X_train_pred = np.vstack((
    np.load(path.join(model_dir, 'outFold.npy')),
    np.load(path.join(model_dir, 'outFold_rf.npy')),
    np.load(path.join(model_dir, 'outFold_ext.npy'))
)).T
X_test_pred = np.vstack((
    np.load(path.join(model_dir, 'pred.npy')),
    np.load(path.join(model_dir, 'pred_rf.npy')),
    np.load(path.join(model_dir, 'pred_ext.npy'))
)).T

X_train_all = np.hstack((X_train, X_train_pred))
X_test_all = np.hstack((X_test, X_test_pred))

print(X_train.shape)
print(X_test.shape)

save_dir = path.join("../level4-feature/" + str(args.yix))
if not path.exists(save_dir):
    os.makedirs(save_dir)

np.save(path.join(save_dir, "X_train.npy"), X_train_all)
np.save(path.join(save_dir, "X_test.npy"), X_test_all)
