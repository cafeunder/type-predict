# coding:utf-8
import os
import cv2
import numpy as np
import random


def resize_image(filename, size):
    # 拡張子を除いた画像名を取得
    base, ext = os.path.splitext(filename)

    # 名前-フォルムの部分を抽出
    name = base.split("_")[0]

    # 元画像の読み込みとサイズの取得
    img = cv2.imread("../img/" + filename)
    src_width = len(img[0])
    src_height = len(img)
    src = np.array(img)

    # 長辺方向の比率を計算
    rate = size / src_height if (src_height > src_width) else size / src_width

    # 元画像を拡大
    expand = cv2.resize(img, (int(src_width * rate), int(src_height * rate)))
    exp_width = len(expand[0])
    exp_height = len(expand)

    # 余白をランダムに決定する
    x = random.randint(0, size - exp_width)
    y = random.randint(0, size - exp_height)

    # 矩形領域に貼り付け
    dst = np.ndarray([size, size, 3])
    dst.fill(255)
    dst[y:y + exp_height, x:x + exp_width] = expand
    return dst


if __name__ == "__main__":
    result = resize_image("../img/absol_588.png", 256)
    cv2.imwrite("../result.png", result)
