import mxnet as mx
import numpy as np
from skimage import io, transform


def preprocess_img(path):
    # load image
    img = io.imread(path)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 117
    normed_img = np.resize(normed_img, (1, 3, 224, 224))
    return normed_img


def preprocess_imglist(path_list):
    img_list = [preprocess_img(path) for path in path_list]
    return np.concatenate(img_list, axis=0)


def get_feature_extractor(prefix, num_round, batch_size):
    model = mx.model.FeedForward.load(
        prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals["global_pool_output"]
    feature_extractor = mx.model.FeedForward(
        ctx=mx.gpu(),
        symbol=fea_symbol,
        numpy_batch_size=1,
        arg_params=model.arg_params,
        aux_params=model.aux_params,
        allow_extra_params=True)
    return feature_extractor


def get_features(img_list, prefix, num_round):
    batch_size = len(img_list)
    extractor = get_feature_extractor(prefix, num_round, batch_size)
    batch = preprocess_imglist(img_list)
    features = extractor.predict(batch)
    features.resize(features.shape[0:2])
    return features


# img_list = ["/home/s_ariel/Desktop/100007.jpg"] * 10
# prefix = "models/inception-21k/Inception"
# num_round = 9
# features = get_features(img_list, prefix, num_round)
# print(features.shape)
