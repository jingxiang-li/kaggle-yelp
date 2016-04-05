import mxnet as mx
import numpy as np
import argparse


def get_iter(imgrec_path, imglist_path, label_width, batch_size):
    result = mx.io.ImageRecordIter(
        path_imgrec=imgrec_path,
        data_shape=(3, 224, 224),
        path_imglist=imglist_path,
        label_width=label_width,
        batch_size=1,
        mean_r=117,
        mean_g=117,
        mean_b=117,
        round_batch=False
    )
    return result


def get_extractor(prefix, num_round, batch_size=2, dev=mx.gpu()):
    model = mx.model.FeedForward.load(
        prefix, num_round, ctx=mx.gpu(), numpy_batch_size=batch_size)
    internals = model.symbol.get_internals()
    fea_symbol = internals["global_pool_output"]
    feature_extractor = mx.model.FeedForward(
        ctx=dev,
        symbol=fea_symbol,
        numpy_batch_size=batch_size,
        arg_params=model.arg_params,
        aux_params=model.aux_params,
        allow_extra_params=True)
    return feature_extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description='extract features from a pretrained')
    parser.add_argument('--imgrec_path', type=str)
    parser.add_argument('--imglist_path', type=str)
    parser.add_argument('--label_width', type=int)
    parser.add_argument('--model_prefix', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--devs', type=str)
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_iter = get_iter(
        args.imgrec_path,
        args.imglist_path,
        args.label_width,
        args.batch_size)
    devs = [mx.gpu(int(x)) for x in args.devs.split(",")]
    extractor = get_extractor(
        args.model_prefix,
        args.num_round,
        args.batch_size,
        devs)
    preds = extractor.predict(img_iter)[:, :, 0, 0]
    print(preds.shape)
    np.save(args.output_path, preds)
