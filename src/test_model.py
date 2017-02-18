# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
import numpy as np
import chainer
import chainer.links as L
from PIL import Image

from dataset.preprocessed_dataset import PreprocessedDataset
from models import alexnet
from resize_image import resize_image


def preprocess_image(path, mean):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=np.float32)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)
    crop_size = insize

    _, h, w = image.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    image = image[:, top:bottom, left:right]
    image -= mean[:, top:bottom, left:right]
    image *= (1.0 / 255.0)
    return image


model = alexnet.Alex(18)
insize = model.insize
model = L.Classifier(model)
chainer.serializers.load_npz("../model_final", model)

mean = np.load("../mean_train.npy")
y = model.predictor(
    np.array([preprocess_image("../test/dorami_256.png", mean)]))
y = y.data
y = np.exp(y) / np.sum(np.exp(y))
print(y)
print(np.argmax(y))
