from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from sklearn.cross_validation import StratifiedKFold

NUM_BIZ_TRAIN = 2000
NUM_BIZ_TEST = 10000


def makeKFold(n_folds, y, reps):
    assert y.shape[0] % reps == 0
    assert int(y.shape[0] / reps) == NUM_BIZ_TRAIN

    y_compact = y[range(0, y.shape[0], reps)]
    SKFold = StratifiedKFold(y_compact, n_folds=n_folds, shuffle=True)
    for train_index, test_index in SKFold:
        train_ix = np.array([np.arange(reps * i, reps * i + reps)
                             for i in train_index]).flatten()
        test_ix = np.array([np.arange(reps * i, reps * i + reps)
                            for i in test_index]).flatten()
        yield (train_ix, test_ix)


def vote_by_majority(pred_list):
    reps = pred_list.size
    npos = np.sum(pred_list > 0.5)
    return 1 if npos > int(np.floor(reps / 2)) else 0


def vote_by_mean(pred_list):
    score = np.mean(pred_list)
    return 1 if score > 0.5 else 0


def agg_preds(preds, reps, vote_func):
    assert preds.shape[0] % reps == 0

    n_samples = int(preds.shape[0] / reps)
    preds_r = np.reshape(preds, (n_samples, reps))
    return np.apply_along_axis(vote_func, axis=1, arr=preds_r)
