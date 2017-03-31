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

def nine_test(path, mean, model, gpu=-1):
    size = [168, 224, 280]

    max_type = {}
    max_val = {}
    for si in range(3):
        for ai in range(3):
            if gpu == -1:
                y_type1 = F.softmax(model.predictor(np.array([preprocess_image(path, mean, 224, size[si], ai)]))).data
            else:
                y_type1 = F.softmax(model_type1.predictor(chainer.cuda.cupy.array([preprocess_image(path, mean, 224, size[si], ai)]))).data

            t = np.argmax(y_type1)
            if t not in max_type:
                max_type[t] = 0
                max_val[t] = 0
            max_type[t] += 1

            v = np.max(y_type1[0])
            if v > max_val[t]:
                max_val[t] = v

    result = 0
    result_num = 0
    result_val = 0
    for key in max_type:
        if max_type[key] > result_num or (max_type[key] == result_num and max_val[key] > result_val):
            result = key
            result_num = max_type[key]
            result_val = max_val[key]

    return result


def preprocess_image(path, mean, insize, exp, align=0):
    mean = mean.astype('f')
    f = Image.open(path).convert('RGBA')

    # 縦横比を変えずinsize x insizeにリサイズ
    width, height = f.size
    rate = float(exp) / max(width, height)
    f = f.resize((int(width * rate), int(height * rate)), Image.ANTIALIAS)

    width, height = f.size
    color = (255, 255, 255)
    background = Image.new('RGB', (insize, insize), color)

    x = 0
    y = 0
    if align == 1:
        x = int((insize - width) / 2)
        y = int((insize - height) / 2)
    elif align == 2:
        x = int(insize - width)
        y = int(insize - height)

    background.paste(f, (x, y), f.split()[3])

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
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    model_type1, out_size_type1 = make_model(args.root + '/type1/model_final',
                                             1)
    model_type2, out_size_type2 = make_model(args.root + '/type2/model_final',
                                             2)
    if args.gpu >= 0:
        model_type1.to_gpu()
        model_type2.to_gpu()

    # 平均画像の読み込み
    mean1 = np.load(args.root + '/type1/mean.npy')
    mean2 = np.load(args.root + '/type2/mean.npy')

    # 推定されたタイプを出力
    type_file = open(args.label, 'r')
    type_list = type_file.read().split("\n")

    # 画像からタイプを予測
    type1 = nine_test(args.img, mean1, model_type1)
    type2 = nine_test(args.img, mean2, model_type2)

    print("Type1 of this image is : " + type_list[type1])
    print("Type2 of this image is : " + type_list[type2])


if __name__ == '__main__':
    main()
