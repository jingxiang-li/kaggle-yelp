from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import xgboost as xg
import numpy as np
import pandas as pd
from os import listdir


def aggregate_features(X):
    return np.mean(X, axis=0)


def get_features():
    biz_id_list = []
    result_array = np.zeros((2000, 1024))
    path = 'features/inception-21k-global/'
    feature_files = listdir(path)
    for i, f in enumerate(feature_files):
        biz_id = int(f[:-4])
        feature_bag = np.load('features/inception-21k-global/' + f)
        out_feature = aggregate_features(feature_bag)
        biz_id_list.append(biz_id)
        result_array[i, :] = out_feature
    col_names = ['incpt21k-glp-avg-' + str(i) for i in range(result_array.shape[1])]
    feature_frame = pd.DataFrame(data=result_array, index=biz_id_list, columns=col_names)
    return feature_frame


def get_response():
    biz2label = pd.read_csv("rawdata/train.csv")
    result_array = np.zeros((2000, 9))
    for class_no in range(9):
        response = [
            1 if str(class_no) in str(l).split(" ") else 0 for l in biz2label["labels"]]
        result_array[:, class_no] = response
    response_frame = pd.DataFrame(
        data=result_array,
        index=biz2label["business_id"],
        columns=['class' + str(i) for i in range(9)],
        dtype=int)
    return response_frame


X = get_features()
Y = get_response()
dataframe = pd.merge(X, Y, left_index=True, right_index=True)


