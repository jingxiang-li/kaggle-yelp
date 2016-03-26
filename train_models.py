from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import xgboost as xgb
import numpy as np
import pandas as pd
from os import listdir
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score

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


def get_data():
    X = get_features()
    Y = get_response()
    dataframe = pd.merge(X, Y, left_index=True, right_index=True)
    return dataframe


dataframe = get_data().as_matrix()
X = dataframe[:, 0:1024]
y = dataframe[:, -1].astype(int)
print(np.sum(y == 1) / y.shape[0])
dtrain = xgb.DMatrix(X, label=y)

param = {'max_depth':1, 'eta':1, 'silent':1, 'objective':'binary:logitraw', 'lambda':1}
param['nthread'] = 4

# watchlist  = [(dtrain,'train'), (dtest,'eval')]
num_round = 2000

def evalerror(preds, dtrain):
    pred_labels = [1 if p > 0 else 0 for p in preds]
    labels = dtrain.get_label()
    return 'f1-score', f1_score(labels, pred_labels)

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

xgb.cv(param, dtrain, num_round, feval=evalerror, early_stopping_rounds=30, maximize=True, verbose_eval=True, fpreproc=fpreproc)









# sss = StratifiedShuffleSplit(dataframe[:, -3], 1, test_size=0.2)
# for train_index, test_index in sss:
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
# param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logitraw', 'lambda':1}
# param['nthread'] = 4

# watchlist  = [(dtrain,'train'), (dtest,'eval')]
# num_round = 2000


# def evalerror(preds, dtrain):
#     pred_labels = [1 if p > 0 else 0 for p in preds]
#     labels = dtrain.get_label()
#     return 'f1-score', f1_score(labels, pred_labels)

# bst = xgb.train(param, dtrain, num_round, watchlist, feval=evalerror, early_stopping_rounds=30, maximize=True)
