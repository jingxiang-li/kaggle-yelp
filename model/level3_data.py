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

np.random.seed(14567547)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


def parse_dir_name(dir_name):
    m = re.match(r"(\d+)_(\d+)_(21k|v3|colHist)", dir_name)
    reps = int(m.group(1))
    prob = float(m.group(2)) / 100
    data = str(m.group(3))
    return reps, prob, data

args = parse_args()
models_lvl2_path = "../level2-models/" + str(args.yix)

X_train_next_list = []
X_test_next_list = []
for model_dir in listdir(models_lvl2_path):
    dir_path = path.join(models_lvl2_path, model_dir)
    reps, prob, data = parse_dir_name(model_dir)
    # get model
    with open(path.join(dir_path, "model.pkl"), "rb") as f:
        clf = pkl.load(f)
    # get x_train_next
    X_train = expit(np.load(path.join(dir_path, "outFold.npy")))
    X_train_next = agg_preds(X_train, reps, mean_pool)
    # get x_test_next
    X_test_path = "../level2-feature/" + str(args.yix) + "/" + model_dir
    X_test = np.load(path.join(X_test_path, "X_test.npy"))
    dtest = xgb.DMatrix(X_test)
    X_test_tmp = clf.predict(dtest, output_margin=False)
    X_test_next = agg_preds(X_test_tmp, reps, mean_pool)

    X_train_next_list.append(X_train_next)
    X_test_next_list.append(X_test_next)

X_train_all = np.array(X_train_next_list).T
X_test_all = np.array(X_test_next_list).T

y_train_tmp = np.load("../feature/5_75/y_train.npy")
y_train = y_train_tmp[::5, args.yix]

save_dir = path.join("../level3-feature/" + str(args.yix))
if not path.exists(save_dir):
    os.makedirs(save_dir)

np.save(path.join(save_dir, "X_train.npy"), X_train_all)
np.save(path.join(save_dir, "X_test.npy"), X_test_all)
np.save(path.join(save_dir, "y_train.npy"), y_train)

print("class", args.yix)
print(X_train_all.shape)
print(X_test_all.shape)
print(y_train.shape)
print("completes!")
