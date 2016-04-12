from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import os
from os import path

from predict import get_level4_features


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


def agg_function(x):
    return np.concatenate(([np.mean(x),
                            np.std(x)],
                           np.percentile(x, range(0, 101, 10)),
                           [np.sum(x > 0.5) / x.size,
                            np.sum(x > 0.55) / x.size,
                            np.sum(x > 0.6) / x.size,
                            np.sum(x > 0.65) / x.size,
                            np.sum(x > 0.7) / x.size,
                            np.sum(x > 0.45) / x.size,
                            np.sum(x > 0.4) / x.size,
                            np.sum(x > 0.35) / x.size,
                            np.sum(x > 0.3) / x.size,
                            np.sum(x > 0.5),
                            np.sum(x > 0.55),
                            np.sum(x > 0.6),
                            np.sum(x > 0.65),
                            np.sum(x > 0.7),
                            np.sum(x > 0.45),
                            np.sum(x > 0.4),
                            np.sum(x > 0.35),
                            np.sum(x > 0.3),
                            x.size]))

args = parse_args()

save_dir = path.join("../pic-feature/" + str(args.yix))
if not path.exists(save_dir):
    os.makedirs(save_dir)

feature_21k = np.load('../feature/inception-21k-train.npy')
feature_colHist = np.load('../feature/colHist-train.npy')
feature_v3 = np.load('../feature/inception-v3-train.npy')

ft_raw_tmp = np.hstack((feature_21k, feature_colHist, feature_v3))
ft_raw = np.hstack((ft_raw_tmp, np.zeros((ft_raw_tmp.shape[0], 3458))))
ft_lvl4 = get_level4_features(ft_raw, args)

np.save(path.join(save_dir, 'pic_train.npy'), ft_lvl4)

del feature_21k
del feature_colHist
del feature_v3
del ft_raw_tmp
del ft_raw
del ft_lvl4

feature_21k = np.load('../feature/inception-21k-test.npy')
feature_colHist = np.load('../feature/colHist-test.npy')
feature_v3 = np.load('../feature/inception-v3-test.npy')

ft_raw_tmp = np.hstack((feature_21k, feature_colHist, feature_v3))
ft_raw = np.hstack((ft_raw_tmp, np.zeros((ft_raw_tmp.shape[0], 3458))))
ft_lvl4 = get_level4_features(ft_raw, args)

np.save(path.join(save_dir, 'pic_test.npy'), ft_lvl4)
