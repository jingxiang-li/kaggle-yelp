from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import argparse
import os
from os import path
import pandas as pd


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

save_dir = path.join('../pic-feature-final/', str(args.yix))
if not path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = path.join('../pic-feature/', str(args.yix))


# for training data
X_train = np.load(path.join(data_dir, 'pic_train.npy'))[:, -3:]

img_df = pd.read_csv('../data/imglist_train.txt', sep='\t', header=None)
img_index = img_df.ix[:, 0].as_matrix()

img_feature_df = pd.DataFrame(data=X_train, index=img_index)

photo2biz = pd.read_csv('../data/train_photo_to_biz_ids.csv',
                        index_col='photo_id')
biz_list = photo2biz['business_id'].unique()

result_list = []
for biz_id in biz_list:
    photo_ids = photo2biz[photo2biz['business_id'] == biz_id].index.values
    print(biz_id, len(photo_ids))
    preds = img_feature_df.loc[photo_ids, :].as_matrix()
    ft = np.apply_along_axis(agg_function, 0, preds).flatten(order='F')
    print(ft.shape)
    result_list.append(ft)

result_array = np.asarray(result_list)
np.save(path.join(save_dir, 'train.npy'), result_array)


# for test data
X_test = np.load(path.join(data_dir, 'pic_test.npy'))[:, -3:]

img_df = pd.read_csv('../data/imglist_test.txt', sep='\t', header=None)
img_index = img_df.ix[:, 0].as_matrix()

img_feature_df = pd.DataFrame(data=X_test, index=img_index)

photo2biz = pd.read_csv('../data/test_photo_to_biz.csv',
                        index_col='photo_id')
biz_list = photo2biz['business_id'].unique()

result_list = []
for biz_id in biz_list:
    photo_ids = photo2biz[photo2biz['business_id'] == biz_id].index.values
    print(biz_id, len(photo_ids))
    preds = img_feature_df.loc[photo_ids, :].as_matrix()
    ft = np.apply_along_axis(agg_function, 0, preds).flatten(order='F')
    print(ft.shape)
    result_list.append(ft)

result_array = np.asarray(result_list)
np.save(path.join(save_dir, 'test.npy'), result_array)
