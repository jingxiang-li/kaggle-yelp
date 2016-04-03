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
    col_names = [
        'incpt21k-glp-avg-' +
        str(i) for i in range(
            result_array.shape[1])]
    feature_frame = pd.DataFrame(
        data=result_array,
        index=biz_id_list,
        columns=col_names)
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


# def get_testdata():
#     photo2ftr = pd.read_csv("features/inception-21k-global-test.csv",
#                             index_col=0, header=None)
#     photo2biz = pd.read_csv("rawdata/test_photo_to_biz.csv")
#     biz_ids = np.unique(photo2biz["business_id"])
#     test_data = np.zeros((len(biz_ids), 1024))
#     for i, biz_id in enumerate(biz_ids):
#         dd = photo2biz[photo2biz["business_id"] == biz_id]
#         photo_ids = np.unique(dd["photo_id"])
#         feature = np.mean(photo2ftr.loc[photo_ids].as_matrix(), axis=0)
#         test_data[i, :] = feature
#     np.save('features/inception-21k-global-test.npy', test_data)

# dataframe = get_data().as_matrix()
# np.save('features/train_data.npy', dataframe)


def evalerror(preds, dtrain):
    pred_labels = [1 if p > 0 else 0 for p in preds]
    labels = dtrain.get_label()
    return 'f1-score', f1_score(labels, pred_labels)


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

dataframe = np.load('features/train_data.npy')
X = dataframe[:, 0:1024]
y_array = dataframe[:, -9:].astype(int)
X_test = np.load("features/inception-21k-global-test.npy")
print(X_test.shape)
dtest = xgb.DMatrix(X_test)
preds_array = np.zeros((X_test.shape[0], y_array.shape[1]))

for y_index in range(9):
    y = y_array[:, y_index]
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        'max_depth': 3,
        'eta': 0.5,
        'silent': 1,
        'objective': 'binary:logitraw',
        'nthread': 4}

    cv_result = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=2000,
        nfold=5,
        feval=evalerror,
        early_stopping_rounds=30,
        maximize=True,
        verbose_eval=True,
        fpreproc=fpreproc,
        show_stdv=False)

    # train model and predict on test set
    opt_round_num = cv_result.shape[0] - 1
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    # add dtest here

    clf = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=opt_round_num,
        evals=[(dtrain, 'train')],
        feval=evalerror,
        maximize=True)

    preds = (clf.predict(dtest) > 0).astype(int)
    print(preds.shape)
    print(preds_array.shape)
    preds_array[:, y_index] = preds

np.savetxt("output/first_try.csv", preds_array, delimiter=",")
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
