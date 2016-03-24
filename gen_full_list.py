from __future__ import division, with_statement, print_function
import pandas as pd
import numpy as np

biz2label = pd.read_csv("/home/s_ariel/Documents/yelp/rawdata/train.csv")
photo2biz = pd.read_csv(
    "/home/s_ariel/Documents/yelp/rawdata/train_photo_to_biz_ids.csv")

# merge two datasets together and sort according to photo id
df = photo2biz.merge(biz2label, on="business_id")
df.sort_values("photo_id", inplace=True)


# encode labels to n-hot ways
def encode_labels(in_label):
    if in_label is np.nan:
        res = ["0"] * 9
    else:
        ix = map(int, in_label.strip().split(" "))
        res = ["1" if i in ix else "0" for i in range(9)]
    return "\t".join(res)

with open("train_list.txt", "w+") as fout:
    for index, row in df.iterrows():
        fout.write(str(row["photo_id"]))
        fout.write("\t")
        fout.write(encode_labels(row["labels"]))
        fout.write("\t")
        fout.write(str(row["photo_id"]) + ".jpg\n")
