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

def add_background(filename):
    # 拡張子を除いた画像名を取得
    base, ext = os.path.splitext(filename)

    # 名前-フォルムの部分を抽出
    name = base.split('_')[0]

    # 画像読み込み
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    # マスクの取り出し
    mask = cv2.cvtColor(src[:,:,3], cv2.COLOR_GRAY2RGB)
    mask = mask / 255.0

    # 次元を背景に合わせるため、アルファチャンネルなしの画像に変換
    src = cv2.cvtColor(src, cv2.COLOR_RGBA2RGB)

    # 背景を作成
    background = np.zeros(src.shape, dtype=np.float64)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.rectangle(background, (0, 0), (src.shape[1], src.shape[0]), color, cv2.FILLED)

    # 画像を背景にのせる
    background *= 1 - mask
    background += src * mask

    return background


def scale_augmentation(image):
    """
    Scale Augmentationを行う
    ついでにHorizontal Flipもする
    """
    CROP_SIZE = 224
    RESIZE_MIN, RESIZE_MAX = 224, 360

    # 元画像の読み込みとサイズの取得
    src_width = len(image[0])
    src_height = len(image)
    src = np.array(image)

    # [RESIZE_MIN, RESIZE_MAX]の範囲でランダムにリサイズする
    size = random.randint(RESIZE_MIN, RESIZE_MAX)

    # 短辺方向の比率を計算
    rate = size / float(src_height) if (
    src_height < src_width) else size / float(src_width)

    # 元画像を拡大
    expand = cv2.resize(image, (int(src_width * rate), int(src_height * rate)))
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
    original_filename_list = glob.glob(args.srcdir + "/*.png")
    augmentation_list = []
    img_name_list = []

    # Scale Augmentationを行い、結果をリストにまとめる
    for filename in original_filename_list:
        img_name = os.path.basename(filename) # 画像名
        # ポケモンごとに画像フォルダを作成
        if not os.path.exists(args.dstdir + "/" + img_name.split("_")[0]):
            os.makedirs(args.dstdir + "/" + img_name.split("_")[0])
        print(img_name)

        image = scale_augmentation(add_background(filename))

        # ファイルに書き出し
        cv2.imwrite(args.dstdir + "/" + img_name.split("_")[0]
                    + "/" + os.path.splitext(img_name)[0] + ".png", image)
