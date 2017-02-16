'''
gifディレクトリにある画像の全フレームをpng画像化する
'''
import cv2
import os
import re
import time

# gif画像の全フレームを読み込む
def read_gif_all(filename):
    gif = cv2.VideoCapture(filename)
    result = list()
    while(True):
        _, img = gif.read()
        if img is None: break
        result.append(img)
    return result

count = 0
files = os.listdir('../gif')
for filename in files:
    # 拡張子を除いた画像名を取得
    imgname, _ = os.path.splitext(filename)

    name_form = imgname.split("-")
    # フォルムを表す部分があるなら
    if len(name_form) >= 2:
        # フォルムを表す部分が通し番号 or 性別なら無視する
        ignore_form_pattern = r"[0-9]|^m$|^f$"
        if re.match(ignore_form_pattern, name_form[1]):
            imgname = name_form[0]

    print(imgname)
    # gifの全フレームを読み込んで画像配列を作成
    imgList = read_gif_all("../gif/" + filename)

    for img in imgList:
        # countは重複回避のためにつけられるため、値自体に意味はない
        cv2.imwrite("../png/" + imgname + "_" + str(count) + ".png", img)
        count += 1
