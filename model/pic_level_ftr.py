from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
from os import path
import pickle
import pandas as pd

from predict import get_level4_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yix', type=int, default=0)
    return parser.parse_args()


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

img_df = pd.read_csv('../data/imglist_train.txt', sep='\t', header=None)
img_index = img_df.ix[:, 0].as_matrix()
feature_21k = np.load('../feature/inception-21k-train.npy')
feature_colHist = np.load('../feature/colHist-train.npy')
feature_v3 = np.load('../feature/inception-v3-train.npy')

ft_raw = np.hstack((feature_21k, feature_colHist, feature_v3))
print(ft_raw.shape)
ft_raw = np.hstack((ft_raw, np.zeros(ft_raw.shape[0], 3458)))

print(ft_raw.shape)
ft_lvl4 = get_level4_features(ft_raw, args)
print(ft_lvl4.shape)


# img_feature_df = pd.DataFrame(
#     data=np.hstack((feature_21k, feature_colHist, feature_v3)),
#     index=img_index)

# photo2biz = pd.read_csv(
#     "data/train_photo_to_biz_ids.csv", index_col='photo_id')
# biz_list = photo2biz["business_id"].unique()

# get_level4_features
