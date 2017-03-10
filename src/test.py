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
from models import alexnet

def preprocess_image(path, mean, insize):
    img = cv2.imread(path)
    width, height = np.shape(img)[0], np.shape(img)[1]
    rate = float(insize) / min(width, height)
    img = cv2.resize(img, (int(width * rate), int(height * rate)))
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


def make_model(model_path, learn_type):
    """
    学習済みモデルの読み込みを行うメソッド．
    :param model_path: 学習済みモデルのパス
    :param learn_type: テストするタイプの番号
    :return: 読み込んだ番号
    """
    # 学習済みモデルの読み込み
    if learn_type == 1:
        out_size = 18
    else:
        out_size = 19
    model = alexnet.Alex(out_size, False)
    model = L.Classifier(model)
    chainer.serializers.load_npz(model_path + "_type" + str(learn_type),
                                 model)
    return model, out_size


def main():
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--model', default='../../model_final',
                        help='Path to learned model')
    parser.add_argument('--mean', '-o', default='../../mean.npy',
                        help='path to mean array')
    parser.add_argument('--label', default='../../labels.txt',
                        help='Path to label file')
    parser.add_argument('--img', help='Path to image file')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    model_type1, out_size_type1 = make_model(args.model, 1)
    model_type2, out_size_type2 = make_model(args.model, 2)

    # 平均画像の読み込み
    mean = np.load(args.mean)

    # 推定されたタイプを出力
    type_file = open(args.label, 'r')
    type_list = type_file.read().split("\n")

    # 画像からタイプを予測
    y_type1 = model_type1.predictor(
        np.array([preprocess_image(args.img, mean, 224)]))
    y_type1 = y_type1.data
    y_type1 = np.exp(y_type1) / np.sum(np.exp(y_type1))  # ソフトマックス関数で各タイプの確率を計算

    for i in range(out_size_type1):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y_type1[0][i] * 100)), "%")
    print("Type1 of this image is : " + type_list[np.argmax(y_type1)])

    # 画像からタイプを予測
    y_type2 = model_type2.predictor(
        np.array([preprocess_image(args.img, mean, 224)]))
    y_type2 = y_type2.data
    y_type2 = np.exp(y_type2) / np.sum(np.exp(y_type2))  # ソフトマックス関数で各タイプの確率を計算

    for i in range(out_size_type2):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y_type2[0][i] * 100)), "%")
    print("Type2 of this image is : " + type_list[np.argmax(y_type2)])


if __name__ == '__main__':
    main()
