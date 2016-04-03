from __future__ import print_function
from skimage import img_as_ubyte, img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sys import argv
import numpy as np
import cv2
from scipy.fftpack import dct
import pandas as pd
import re


def image_normalize(input_path):
    img = img_as_float(imread(input_path))
    if img.shape[0] > img.shape[1]:
        margin = (img.shape[0] - img.shape[1]) / 2
        img = img[margin:margin + img.shape[1], :]
    else:
        margin = (img.shape[1] - img.shape[0]) / 2
        img = img[:, margin:margin + img.shape[0]]

    return resize(img, (128, 128))


def feature_small_img(input_img):
    img = resize(input_img, (16, 16))
    return img.flatten()


def feature_fft(input_img):
    img = rgb2gray(input_img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = resize(20 * np.log(np.abs(fshift)), (16, 16))
    return magnitude_spectrum.flatten()


def feature_dct(input_img):
    img = img_as_float(rgb2gray(input_img))
    d = resize(dct(img), (16, 16))
    return d.flatten()


def extract_features(input_img):
    img = image_normalize(input_img)
    result = np.concatenate(
        (feature_small_img(img), feature_fft(img), feature_dct(img)))
    return result

# impath = "/home/s_ariel/Desktop/13.jpg"
# img = extract_features(impath)
# print img

df = pd.read_csv(
    "/home/s_ariel/Documents/yelp/rawdata/train_photo_to_biz_ids.csv")
df.sort_values("photo_id", inplace=True)
# str(row["photo_id"]) + ".jpg\n"

img_dir = "/home/s_ariel/Documents/yelp/rawdata/train_photos"

count = 0
with open("feature_set.csv", "w+") as fout:
    for index, row in df.iterrows():
        count += 1
        print(count)
        img_path = img_dir + "/" + str(row["photo_id"]) + ".jpg"
        features = extract_features(img_path)
        fout.write(str(row["photo_id"]))
        fout.write("\t")
        [fout.write(str(x) + "\t") for x in features]
        fout.write("\n")


# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = np.log(np.abs(fshift))
# print magnitude_spectrum


# d = dct(img / 255.0)
# magnitude_spectrum = np.log(np.abs(d))
# print magnitude_spectrum.shape

# print img.shape
# plt.imshow(magnitude_spectrum)
# plt.show()
