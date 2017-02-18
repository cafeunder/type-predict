# coding:utf-8
'''
pngディレクトリにある画像から、白を除いた最小矩形の画像を生成する
'''

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

files = os.listdir("../test")
# すべての画像を読み込み、最小の矩形領域を抽出
for filename in files:
    base, ext = os.path.splitext(filename)
    if not (ext == ".png" or ext == ".jpg" or ext == ".jpeg"): continue

    img = cv2.imread("../test/" + filename)
    print(filename)
    x, y, width, height = calcRect(img)
    dst = img[y:y + height, x:x + width]
    os.remove("../test/" + filename)
    cv2.imwrite("../test/" + os.path.splitext(filename)[0] + ".png", dst)
