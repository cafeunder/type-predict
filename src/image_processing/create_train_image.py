# coding:utf-8
"""
学習に用いる画像を前処理してデータセットを生成するクラス
"""
import os
import cv2
import numpy as np
import random
import argparse
import glob


def scale_augmentation(filename):
    """
    Scale Augmentationを行う
    ついでにHorizontal Flipもする
    """
    CROP_SIZE = 224
    RESIZE_MIN, RESIZE_MAX = 224, 360

    # 拡張子を除いた画像名を取得
    base, ext = os.path.splitext(filename)

    # 名前-フォルムの部分を抽出
    name = base.split('_')[0]

    # 元画像の読み込みとサイズの取得
    img = cv2.imread(filename)
    src_width = len(img[0])
    src_height = len(img)
    src = np.array(img)

    # [RESIZE_MIN, RESIZE_MAX]の範囲でランダムにリサイズする
    size = random.randint(RESIZE_MIN, RESIZE_MAX)

    # 短辺方向の比率を計算
    rate = size / float(src_height) if (
    src_height < src_width) else size / float(src_width)

    # 元画像を拡大
    expand = cv2.resize(img, (int(src_width * rate), int(src_height * rate)))
    exp_width = len(expand[0])
    exp_height = len(expand)

    # 1/2の確率で左右反転
    if random.randint(0, 1) == 1:
        expand = cv2.flip(expand, 1)

    # ランダムな位置を取り出す
    x = random.randint(0, exp_width - CROP_SIZE) if exp_width > CROP_SIZE else 0
    y = random.randint(0,
                       exp_height - CROP_SIZE) if exp_height > CROP_SIZE else 0

    # 矩形領域に貼り付け
    dst = expand[y:y + CROP_SIZE, x:x + CROP_SIZE]
    return dst


def trans_background(image, background):
    reshape_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    trans = np.random.randint(0, 255, 3)
    trans_image = np.array([trans if all(reshape_image[i] == background) else reshape_image[i] for i in range(reshape_image.shape[0])])
    return trans_image.reshape(image.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create train image')
    parser.add_argument('--srcdir', help='Path to directory of original image')
    parser.add_argument('--dstdir', default='../../train_data',
                        help='Path to train image file')
    args = parser.parse_args()

    # 訓練画像を保存するフォルダが存在しない場合，作成
    if not os.path.exists(args.dstdir):
        os.makedirs(args.dstdir)

    # 元画像のリストを取得
    original_image_list = glob.glob(args.srcdir + "/*.png")
    augmentation_list = []
    img_name_list = []

    background = np.array([255, 255, 255])
    # Scale Augmentationを行い、結果をリストにまとめる
    for image in original_image_list:
        img_name = os.path.basename(image) # 画像名
        # ポケモンごとに画像フォルダを作成
        if not os.path.exists(args.dstdir + "/" + img_name.split("_")[0]):
            os.makedirs(args.dstdir + "/" + img_name.split("_")[0])
        print(img_name)

        trans_image = trans_background(scale_augmentation(image), background)

        # ファイルに書き出し
        cv2.imwrite(args.dstdir + "/" + img_name.split("_")[0]
                    + "/" + os.path.splitext(img_name)[0] + ".png", trans_image)
