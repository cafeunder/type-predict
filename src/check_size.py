# coding:utf-8
'''
画像の最大・最小サイズを調べる。
'''

import os
import cv2

max_width = [-1, None]
min_width = [10000000, None]
max_height = [-1, None]
min_height = [10000000, None]

files = os.listdir('../img')
for filename in files:
    # 拡張子を除いた画像名を取得
    base, ext = os.path.splitext(filename)
    if not ext == ".png": continue

    # 名前-フォルムの部分を抽出
    name = base.split("_")[0]

    img = cv2.imread("../img/" + filename)
    width = len(img[0])
    height = len(img)
    if max_width[0] < width:
        max_width[0] = width
        max_width[1] = name
    if min_width[0] > width:
        min_width[0] = width
        min_width[1] = name
    if max_height[0] < height:
        max_height[0] = height
        max_height[1] = name
    if min_height[0] > height:
        min_height[0] = height
        min_height[1] = name

print("max width : " + str(max_width[0]) + "," + max_width[1])
print("min width : " + str(min_width[0]) + "," + min_width[1])
print("max height : " + str(max_height[0]) + "," + max_height[1])
print("min height : " + str(min_height[0]) + "," + min_height[1])
