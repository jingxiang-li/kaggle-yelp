#!/bin/bash


wget http://data.dmlc.ml/mxnet/models/imagenet/inception-v3.tar.gz
tar zxf inception-v3.tar.gz
mv model inception-v3

wget http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz
tar zxf inception-21k.tar.gz
mv model inception-21k
