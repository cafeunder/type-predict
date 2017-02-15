'''
画像と辞書を使ったプログラムのサンプル
'''

import os
import cv2

# 辞書を読み込む
dict = {}
for line in open("../dictionary.csv", "r"):
    record = line.split(",")
    type = list()
    type.append(record[1])
    if not record[2][:-1] == "":
        type.append(record[2][:-1])
    dict[record[0]] = type

# 画像を読み込む
files = os.listdir('../img')
for filename in files:
    # 拡張子を除いた画像名を取得
    base, ext = os.path.splitext(filename)
    if not ext == ".png": continue

    # 名前-フォルムの部分を抽出
    name = base.split("_")[0]
    if not name in dict:
        continue

    img = cv2.imread("../img/" + filename)
    print(name + ":" + dict[name][0] + ":" + str(len(img[0])) + "," + str(len(img)))
