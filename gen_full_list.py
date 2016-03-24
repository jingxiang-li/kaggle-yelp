from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import pandas as pd
from transfer_features import *


biz2label = pd.read_csv("rawdata/train.csv", index_col=0)
photo2biz = pd.read_csv("rawdata/train_photo_to_biz_ids.csv", index_col=0)
biz2label.sort_index(inplace=True)


for biz_id, biz_label in biz2label.iterrows():
    photo_ids = photo2biz[photo2biz["business_id"] == biz_id].index
    batch_size = len(photo_ids)
    img_list = ['rawdata/train_photos/' + str(id) + '.jpg' for id in photo_ids]
    # pprint(img_list)
    out_file = 'features/inception-21k-global/' + str(biz_id) + '.npy'
    X = get_features(img_list, 'models/inception-21k/Inception', 9)
    np.save(out_file, X)
    print(out_file, 'finished!!')
