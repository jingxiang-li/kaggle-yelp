from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from os import path
from sys import argv


def parse_labels(s, n=9):
    v = ['0'] * 9
    if type(s) is str:
        for x in s.split(' '):
            v[int(x)] = '1'
    return '\t'.join(v).strip()


def main(prefix="sample_data/"):
    # process training data
    biz2label = pd.read_csv(
        path.join(prefix, 'train.csv'), index_col='business_id')
    photo2biz = pd.read_csv(
        path.join(prefix, 'train_photo_to_biz_ids.csv'), index_col='photo_id')

    fout = open(path.join(prefix, "imglist_train.txt"), "w")
    for biz_id, row in biz2label.iterrows():
        biz_label = row[0]
        photo_ids = photo2biz[photo2biz["business_id"] == biz_id].index
        labels = parse_labels(biz_label)
        for photo_id in photo_ids:
            fout.write(str(photo_id) + "\t")
            fout.write(labels + "\t")
            fout.write(str(photo_id) + ".jpg\n")
    fout.close()
    print("imglist_train.txt Comlete!!!")

    # process test data, there are duplications in test photo_id
    photo2biz = pd.read_csv(path.join(prefix, 'test_photo_to_biz.csv'))
    photo_ids = photo2biz["photo_id"].unique()
    fout = open(path.join(prefix, "imglist_test.txt"), "w")
    for photo_id in photo_ids:
        fout.write(str(photo_id) + "\t")
        fout.write("0" + "\t")
        fout.write(str(photo_id) + ".jpg\n")
    fout.close()
    print("imglist_test.txt Comlete!!!")


if __name__ == "__main__":
    assert(len(argv) == 2)
    prefix = argv[1]
    main(prefix)
