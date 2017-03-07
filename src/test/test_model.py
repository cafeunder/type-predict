# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
import argparse

import numpy as np
import chainer
import chainer.links as L
import cv2
from PIL import Image
from src.models import alexnet


def preprocess_image(path, mean, insize):
    img = cv2.imread(path)
    width, height = np.shape(img)[0], np.shape(img)[1]
    rate = float(insize) / min(width, height)
    img = cv2.resize(img, (int(width * rate), int(height * rate)))
    print(np.shape(img))
    image = np.asarray(img, dtype=np.float32)
    if image.ndim == 2:
        # 画像の次元数が2の場合，1次元分足す
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)

    _, h, w = image.shape
    top = (h - insize) // 2
    left = (w - insize) // 2
    bottom = top + insize
    right = left + insize
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
        out_size = 18
    else:
        out_size = 19
    model = alexnet.Alex(out_size, False)
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

    for i in range(out_size):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y[0][i] * 100)), "%")
    print("Type of this image is : " + type_list[np.argmax(y)])


if __name__ == '__main__':
    main()
