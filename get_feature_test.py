from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from sys import argv
import os
from os import path


def main(prob=0.75, reps=5):

    # process test data
    img_df = pd.read_csv("data/imglist_test.txt", sep="\t", header=None)
    img_index = img_df.ix[:, 0].as_matrix()
    feature_21k = np.load("feature/inception-21k-test.npy")
    feature_colHist = np.load("feature/colHist-test.npy")
    feature_v3 = np.load("feature/inception-v3-test.npy")

    img_feature_df = pd.DataFrame(
        data=np.hstack((feature_21k, feature_colHist, feature_v3)),
        index=img_index)

    photo2biz = pd.read_csv(
        "data/test_photo_to_biz.csv", index_col='photo_id')
    biz_list = photo2biz["business_id"].unique()

    print(biz_list.shape)
    result_list = []

    for c, biz_id in enumerate(biz_list):
        photo_ids = photo2biz[photo2biz["business_id"] == biz_id].index.values
        length = len(photo_ids)
        for i in range(reps):
            sample = np.random.choice(
                photo_ids, size=int(prob * length), replace=False)
            fset = img_feature_df.loc[sample, :]
            f_all = np.hstack(
                (np.mean(fset, axis=0), np.std(fset, axis=0)))
            result_list.append(f_all)
        if c % 100 == 0:
            print(c)

    X_test = np.asarray(result_list)

    dir_path = "feature/" + str(reps) + "_" + str(int(100 * prob))
    print(dir_path)
    if not path.isdir(dir_path):
        os.mkdir(dir_path)
    np.save(path.join(dir_path, "X_test.npy"), X_test)
    np.save(path.join(dir_path, "bizlist.npy"), np.asarray(biz_list))


if __name__ == "__main__":
    assert(len(argv) == 3)
    prob, reps = float(argv[1]), int(argv[2])
    main(prob, reps)
