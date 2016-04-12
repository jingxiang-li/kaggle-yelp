from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import argparse
import os
from os import path


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


def get_extra_features(args):
    forest = ExtraTreesClassifier(n_estimators=2000,
                                  criterion='entropy',
                                  max_features='sqrt',
                                  max_depth=6,
                                  min_samples_split=8,
                                  n_jobs=-1,
                                  bootstrap=True,
                                  oob_score=True,
                                  verbose=1,
                                  class_weight='balanced')
    pca = PCA(n_components=200)
    ica = FastICA(n_components=200, max_iter=1000)
    kmeans = KMeans(n_clusters=200, n_init=20, max_iter=1000)

    pipeline = make_pipeline(selectKFromModel(forest, k=1000),
                             StandardScaler(),
                             make_union(pca, ica, kmeans))

    X_train = np.load("feature/1_100/X_train.npy")
    y_train = np.load("feature/1_100/y_train.npy")
    X_test = np.load("feature/1_100/X_test.npy")

    pipeline.fit(X_train, y_train[:, args.yix])
    sel_ixs = pipeline.steps[0][1].indices[:500]
    X_train_ext = np.hstack((pipeline.transform(X_train), X_train[:, sel_ixs]))
    X_test_ext = np.hstack((pipeline.transform(X_test), X_test[:, sel_ixs]))
    return X_train_ext, X_test_ext


args = parse_args()
print(args.yix)

X_train_ext, X_test_ext = get_extra_features(args)

save_dir = "extra_ftrs/" + str(args.yix)
print(save_dir)
if not path.exists(save_dir):
    os.makedirs(save_dir)

np.save(path.join(save_dir, "X_train_ext.npy"), X_train_ext)
np.save(path.join(save_dir, "X_test_ext.npy"), X_test_ext)
