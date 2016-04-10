from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

ftr_ixs = [315, 472, 1581, 2517, 2575, 2670, 3245, 4881, 5097]
ftr_array = np.load("data/ftr_crt.npy")

for reps in [5, 9]:
    for probs in ["75", "50"]:
        data_path = "feature/" + str(reps) + "_" + probs + "/X_test.npy"
        X = np.load(data_path)
        for c, ix in enumerate(ftr_ixs):
            X[range(reps * ix, reps * ix + reps), :] = ftr_array[c, :]
        print(np.any(np.isnan(X)))
        np.save(data_path, X)
    break
