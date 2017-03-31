# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
import argparse

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import cv2
import random
from PIL import Image
from models import alexnet

def preprocess_image(path, mean, insize):
    mean = mean.astype('f')
    f = Image.open(path).convert('RGBA')

    # 縦横比を変えずinsize x insizeにリサイズ
    width, height = f.size
    rate = float(insize) / max(width, height)
    f = f.resize((int(width * rate), int(height * rate)), Image.ANTIALIAS)
    color = (255, 255, 255)
    background = Image.new('RGB', (insize, insize), color)
    background.paste(f, (0, 0), f.split()[3])

    try:
        image = np.asarray(background, np.float32)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if image.ndim == 2:
        # 画像の次元数が2の場合，1次元分足す
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)

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
    chainer.serializers.load_npz(model_path, model)
    return model, out_size


def main():
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--root', default='../data', help='')
    parser.add_argument('--label', default='../labels.txt',
                        help='Path to label file')
    parser.add_argument('--img', help='Path to image file')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    model_type1, out_size_type1 = make_model(args.root + '/type1/model_final', 1)
    model_type2, out_size_type2 = make_model(args.root + '/type2/model_final', 2)

    # 平均画像の読み込み
    mean1 = np.load(args.root + '/type1/mean.npy')
    mean2 = np.load(args.root + '/type2/mean.npy')

    # 推定されたタイプを出力
    type_file = open(args.label, 'r')
    type_list = type_file.read().split("\n")

    # 画像からタイプを予測
    y_type1 = F.softmax(model_type1.predictor(np.array([preprocess_image(args.img, mean1, 224)]))).data

    for i in range(out_size_type1):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y_type1[0][i] * 100)), "%")
    print("Type1 of this image is : " + type_list[np.argmax(y_type1)])

    # 画像からタイプを予測
    y_type2 = F.softmax(model_type2.predictor(np.array([preprocess_image(args.img, mean2, 224)]))).data

    for i in range(out_size_type2):
        if type_list[i] == "":
            break
        print(type_list[i], ":", str(int(y_type2[0][i] * 100)), "%")
    print("Type2 of this image is : " + type_list[np.argmax(y_type2)])
    print(type_list[np.argmax(y_type1)] + "," + type_list[np.argmax(y_type2)])

if __name__ == '__main__':
    main()
