# coding:utf-8
import argparse
import glob
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
    img = cv2.imread(filename)
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


def main():
    parser = argparse.ArgumentParser(
        description='Resize Image to size (256, 256)')
    parser.add_argument('--imgdir', default='../test_img',
                        help='Path to directory of image')
    parser.add_argument('--resizedir', default='../resize_img',
                        help='Path to resized image file')
    parser.add_argument('--fortrain', type=bool, default=False)
    args = parser.parse_args()

    # リサイズした画像を保存するフォルダが存在しない場合，作成
    if not os.path.exists(args.resizedir):
        os.makedirs(args.resizedir)
    if args.fortrain:  # 訓練データ用の画像を作成する場合
        # リサイズ前の画像のリストを取得
        image_list = glob.glob(args.imgdir + "/*.png")
        image_list = image_list + glob.glob(args.imgdir + "/*.jpg")
        image_list = image_list + glob.glob(args.imgdir + "/*.jpeg")
        # リサイズ前の画像をリサイズして保存
        for image in image_list:
            img_name = os.path.basename(image)  # 画像名
            # ポケモンごとに画像フォルダを作成
            if not os.path.exists(args.resizedir + "/" + img_name.split("_")[
                0]):
                os.makedirs(args.resizedir + "/" + img_name.split("_")[0])
            print(img_name)
            # 画像をリサイズ
            result = resize_image(image, 256)
            # リサイズ後の画像を保存
            cv2.imwrite(
                args.resizedir + "/" + img_name.split("_")[
                    0] + "/" + os.path.splitext(img_name)[0] + ".png", result)
    else:  # 学習後のテスト用の画像を作成する場合
        # リサイズ前の画像のリストを取得
        image_list = glob.glob(args.imgdir + "/*.png")
        image_list = image_list + glob.glob(args.imgdir + "/*.jpg")
        image_list = image_list + glob.glob(args.imgdir + "/*.jpeg")
        # リサイズ前の画像をリサイズして保存
        for image in image_list:
            img_name = os.path.basename(image)  # 画像名
            print(image)
            result = resize_image(image, 227)
            cv2.imwrite(
                args.resizedir + "/" + os.path.splitext(img_name)[0] + ".png",
                result)


if __name__ == "__main__":
    main()
