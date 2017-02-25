# coding:utf-8
'''
pngディレクトリにある画像から、白を除いた最小矩形の画像を生成する
画像は、ノイズが無く境界が閉じているpng画像とする
'''
import argparse
import numpy as np
import cv2
import os

large_value = 10000000
# 白を除いた場合の最小矩形領域を計算する
def calcRect(img):
    # 背景は(255,255,255)である前提でグレースケール化
    gray = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    gray_index = np.where(gray != 255)

    left = min(gray_index[1])
    top = min(gray_index[0])
    right = max(gray_index[1])
    bottom = max(gray_index[0])

    return left, top, (right - left), (bottom - top)


def main():
    parser = argparse.ArgumentParser(description='Clip the image to the smallest bounding-box')
    parser.add_argument('--imgdir', help='Path to directory of image')
    parser.add_argument('--clipdir', default="../clip_img", help='Path to cliped image file')
    args = parser.parse_args()

    if args.imgdir is None or not os.path.exists(args.imgdir):
        print(str(args.imgdir) + ": invalid value of --imgdir")
        parser.print_help()
        return
    if not os.path.exists(args.clipdir):
        os.makedirs(args.clipdir)

    files = os.listdir(args.imgdir)
    # すべての画像を読み込み、最小の矩形領域を抽出
    for filename in files:
        base, ext = os.path.splitext(filename)
        if not ext == ".png": continue

        img = cv2.imread(args.imgdir + "/" + filename)
        x, y, width, height = calcRect(img)
        dst = img[y:y + height, x:x + width]
        cv2.imwrite(args.clipdir + "/" + os.path.splitext(filename)[0] + ".png", dst)


if __name__ == "__main__":
    main()
