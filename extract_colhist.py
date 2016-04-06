from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import cv2
from os import path
import pandas as pd


def get_colHist(img_path):
    features = []

    # RGB histogram
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    chans = cv2.split(img)
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [64], [0, 256])[:, 0]
        features.extend(hist / np.sum(hist))

    # HLS histogram
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    chans_hls = cv2.split(img_hls)
    for chan in chans_hls:
        hist = cv2.calcHist([chan], [0], None, [64], [0, 256])[:, 0]
        features.extend(hist / np.sum(hist))

    # Shape information
    features.extend(img.shape[:2])

    return features


# handle training data
prefix = "/home/s_ariel/Documents/kaggle-yelp/data/"
dataframe = pd.read_csv(
    path.join(
        prefix,
        "imglist_train.txt"),
    sep="\t",
    header=None)
img_names = dataframe.ix[:, 0].astype(str)
img_list = [
    path.join(
        prefix,
        "train_photos/",
        img_name +
        ".jpg") for img_name in img_names]

number_imgs = len(img_list)
result_matrix = np.zeros((number_imgs, 64 * 6 + 2), dtype="float32")
for i, img_path in enumerate(img_list):
    feature = get_colHist(img_path)
    result_matrix[i, :] = feature
    if i % 2000 == 0:
        print(i)

print(result_matrix.shape)
np.save(path.join(prefix, "colHist-train.npy"), result_matrix)
del result_matrix

# handle test data
dataframe = pd.read_csv(
    path.join(
        prefix,
        "imglist_test.txt"),
    sep="\t",
    header=None)
img_names = dataframe.ix[:, 0].astype(str)
img_list = [
    path.join(
        prefix,
        "test_photos/",
        img_name +
        ".jpg") for img_name in img_names]

number_imgs = len(img_list)
result_matrix = np.zeros((number_imgs, 64 * 6 + 2), dtype="float32")
for i, img_path in enumerate(img_list):
    feature = get_colHist(img_path)
    result_matrix[i, :] = feature
    if i % 2000 == 0:
        print(i)

print(result_matrix.shape)
np.save(path.join(prefix, "colHist-test.npy"), result_matrix)
