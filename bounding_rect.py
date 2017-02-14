import cv2
import os

large_value = 10000000
# 白を除いた場合の最小矩形領域を計算する
# ひどすぎるのでなんとかしたい
def calcRect(img):
    # 背景は(255,255,255)である前提でグレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    left = large_value
    for y in range(len(gray)):
        max_x = -1
        for x in range(len(gray[0])):
            if gray[y][x] == 255 and max_x < x:
                max_x = x
            else:
                break
        if not max_x == -1 and max_x < left:
            left = max_x

    top = large_value
    for x in range(len(gray[0])):
        max_y = -1
        for y in range(len(gray)):
            if gray[y][x] == 255 and max_y < y:
                max_y = y
            else:
                break
        if not max_y == -1 and max_y < top:
            top = max_y

    right = -1
    for y in range(len(gray)):
        min_x = large_value
        for x in range(len(gray[0]))[::-1]:
            if gray[y][x] == 255 and min_x > x:
                min_x = x
            else:
                break
        if not min_x == large_value and min_x > right:
            right = min_x

    bottom = -1
    for x in range(len(gray[0])):
        min_y = large_value
        for y in range(len(gray))[::-1]:
            if gray[y][x] == 255 and min_y > y:
                min_y = y
            else:
                break
        if not min_y == large_value and min_y > bottom:
            bottom = min_y

    return left, top, (right - left), (bottom - top)

files = os.listdir("png")
# すべての画像を読み込み、最小の矩形領域を抽出
for filename in files:
    root, ext = os.path.splitext(filename)
    if not ext == ".png": continue

    img = cv2.imread("png/" + filename)
    print(filename)
    x, y, width, height = calcRect(img)
    dst = img[y:y + height, x:x + width]
    cv2.imwrite("img/" + filename, dst)