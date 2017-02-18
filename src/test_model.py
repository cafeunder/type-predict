# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
import numpy as np
import chainer
import chainer.links as L

from dataset.preprocessed_dataset import PreprocessedDataset
from models import alexnet

model = alexnet.Alex(18)
insize = model.insize
model = L.Classifier(model)
chainer.serializers.load_npz("../model_final", model)

mean = np.load("../mean_train.npy")
train = PreprocessedDataset("../train.txt", ".", mean, insize)
val = PreprocessedDataset("../test.txt", ".", mean, insize, False)

y = model.predictor(np.array([val.get_example(4152)[0]]))
y = y.data
y = np.exp(y) / np.sum(np.exp(y))
print(y)
print(np.argmax(y))
