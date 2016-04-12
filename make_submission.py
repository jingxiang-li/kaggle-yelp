from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
from os import path, makedirs
from sklearn.preprocessing import binarize
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="level4-model/")
    parser.add_argument('--out_ix', type=str, default="")
    return parser.parse_args()


def parse_class(preds_cls):
    result = []
    for pred in preds_cls:
        pred_ixs = np.arange(9)[pred].astype(str)
        if len(pred_ixs) == 0:
            result.append("")
        else:
            result.append(" ".join(pred_ixs))
    return result

args = parse_args()
pred_list = []
for y_ix in range(9):
    pred_path = path.join(
        args.output_dir, str(y_ix), "pred" + args.out_ix + ".npy")
    pred = np.load(pred_path) > 0.5
    pred_list.append(pred)

preds = np.array(pred_list)
preds_cls = binarize(X=preds, threshold=0.5, copy=True).T
preds_cls_idx = parse_class(preds_cls)

biz_list = np.load("./feature/5_75/bizlist.npy")

output_dir = "output"
if not path.exists(output_dir):
    makedirs(output_dir)

now = datetime.datetime.now().strftime(r"%b-%d-%H-%M")

with open(path.join(output_dir, "submission-" + now + ".csv"), "w") as fout:
    fout.write("business_id,labels\n")
    for biz_id, pred in zip(biz_list, preds_cls_idx):
        fout.write(biz_id + "," + pred + "\n")
