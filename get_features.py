from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd


def parse_labels(s, n=9):
    v = [0] * n
    if type(s) is str:
        for x in s.split(' '):
            v[int(x)] = 1
    return v

prob = 0.75
reps = 5

# Process Training Data
img_df = pd.read_csv("data/imglist_train.txt", sep="\t", header=None)
img_index = img_df.ix[:, 0].as_matrix()
feature_21k = np.load("feature/inception-21k-train.npy")
feature_colHist = np.load("feature/colHist-train.npy")

img_feature_df = pd.DataFrame(
    data=np.hstack((feature_21k, feature_colHist)),
    index=img_index)

photo2biz = pd.read_csv(
    "data/train_photo_to_biz_ids.csv", index_col='photo_id')
biz_list = photo2biz["business_id"].unique()

result_list = []

for biz_id in biz_list:
    photo_ids = photo2biz[photo2biz["business_id"] == biz_id].index.values
    length = len(photo_ids)
    for i in range(reps):
        sample = np.random.choice(
            photo_ids, size=int(prob * length), replace=False)
        fset = img_feature_df.loc[sample, :]
        f_all = np.hstack(
            (np.mean(fset, axis=0), np.std(fset, axis=0)))
        result_list.append(f_all)

X_train = np.asarray(result_list)

biz2label = pd.read_csv(
    "data/train.csv", index_col='business_id')
y_train = np.zeros((X_train.shape[0], 9), dtype="int8")
for i, biz_id in enumerate(biz_list):
    y_train[range(5 * i, 5 * i + 5), :] = \
        parse_labels(biz2label.loc[biz_id]["labels"])

np.save("feature/mean_5_075/X_train.npy", X_train)
np.save("feature/mean_5_075/y_train.npy", y_train)
