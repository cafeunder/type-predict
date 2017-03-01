# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
import argparse

import numpy as np
import chainer
import chainer.links as L
from PIL import Image
from src.models import alexnet


def preprocess_image(path, mean, insize):
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

    image = image[0:3, top:bottom, left:right]
    image -= mean
    image *= (1.0 / 255.0)
    return image


def main():
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--learn_type', type=int, default=1,
                        help='Type number to learn')
    parser.add_argument('--model', default='../../model_final',
                        help='Path to learned model')
    parser.add_argument('--mean', '-o', default='../../mean.npy',
                        help='path to mean array')
    parser.add_argument('--label', default='../../labels.txt',
                        help='Path to label file')
    parser.add_argument('--img', help='Path to image file')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    if args.learn_type == 1:
        model = alexnet.Alex(18, False)
    else:
        model = alexnet.Alex(19, False)
    model = L.Classifier(model)
    chainer.serializers.load_npz(args.model + "_type" + str(args.learn_type),
                                 model)

    # 平均画像の読み込み
    mean = np.load(args.mean)

    # 画像からタイプを予測
    y = model.predictor(
        np.array([preprocess_image(args.img, mean, 224)]))
    y = y.data
    y = np.exp(y) / np.sum(np.exp(y))  # ソフトマックス関数で各タイプの確率を計算

    # 推定されたタイプを出力
    type_file = open(args.label, 'r')
    type_list = type_file.read().split("\n")

    for i in range(len(type_list)):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y[0][i] * 100)), "%")
    print("Type of this image is : " + type_list[np.argmax(y)])


if __name__ == '__main__':
    main()
