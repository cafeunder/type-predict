# coding:utf-8
import os
import argparse
import random
import glob
import numpy as np
import cv2

def add_background(src):
    # マスクの取り出し
    mask = cv2.cvtColor(src[:,:,3], cv2.COLOR_GRAY2RGB)
    mask = mask / 255.0

    # 次元を背景に合わせるため、アルファチャンネルなしの画像に変換
    src = cv2.cvtColor(src, cv2.COLOR_RGBA2RGB)

    # 背景を作成
    background = np.zeros(src.shape, dtype=np.float64)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.rectangle(background, (0, 0), (src.shape[1], src.shape[0]), color, -1)

    # 画像を背景にのせる
    background *= 1 - mask
    background += src * mask

    return background


def scale_augmentation(image):
    """
    Scale Augmentationを行う
    ついでにHorizontal Flipもする
    """
    SIZE = 224
    RESIZE_MIN, RESIZE_MAX = 168, 280

    # 元画像の読み込みとサイズの取得
    src_width = len(image[0])
    src_height = len(image)
    src = np.array(image)

    # [RESIZE_MIN, RESIZE_MAX]の範囲でランダムにリサイズする
    size = random.randint(RESIZE_MIN, RESIZE_MAX)

    # 長辺方向の比率を計算
    rate = size / float(src_height) if (src_height > src_width) else size / float(src_width)

    # 元画像を拡大
    expand = cv2.resize(image, (int(src_width * rate + 0.5), int(src_height * rate + 0.5)))
    exp_width = len(expand[0])
    exp_height = len(expand)

    # 1/2の確率で左右反転
    if random.randint(0, 1) == 1:
        expand = cv2.flip(expand, 1)

    # 矩形領域に貼り付け
    if exp_width > SIZE:
        x = random.randint(0, exp_width - SIZE)
        exp_width = SIZE
        expand = expand[0:exp_height, x:x + exp_width]
    if exp_height > SIZE:
        y = random.randint(0, exp_height - SIZE)
        exp_height = SIZE
        expand = expand[y: y + exp_height, 0:exp_width]

    # ランダムな位置を取り出す
    x = random.randint(0, SIZE - exp_width)
    y = random.randint(0, SIZE - exp_height)
    dst = np.zeros((SIZE, SIZE, 4), dtype='uint8')
    dst[y:y + exp_height, x:x + exp_width] = expand
    return dst

if __name__ == '__main__':
    NO_OF_POKE_EACHTYPE = 10000

    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--type', type=int)
    parser.add_argument('--imgdir', help='Path to image directory')
    parser.add_argument('--dictdir', default='../dictionary.csv',
                        help='Path to dictionary file (Pokemon to Type)')
    parser.add_argument('--label', default='../labels.txt',
                        help='Path to label file')
    parser.add_argument('--dstdir', default='../image/',
                        help='Path to train image file')
    args = parser.parse_args()

    type_file = open(args.label, 'r')  # 存在するタイプを記述してあるファイル
    type_list = type_file.read().split('\n')  # 存在するタイプのリスト（和名，英名）
    type_to_poke = {}
    for type in type_list:
        # 存在するタイプを辞書に登録
        if type == '':
            break
        type_to_poke[type] = []

    # ポケモン名を受け取るとタイプを返す辞書を作成
    poke_dict_file = open(args.dictdir, 'r')  # ポケモンとそのタイプを記述してあるファイル
    poke_to_type = {}  # ポケモン→タイプの辞書
    for poke in poke_dict_file:
        record = poke.split(',')
        record[2] = record[2].replace('\n', '')
        if record[2]:
            poke_to_type[record[0]] = (record[1], record[2])
        else:
            poke_to_type[record[0]] = (record[1], 'none')

    # 画像データとして存在するポケモンの名前をリストに追加
    for poke in os.listdir(args.imgdir):
        if os.path.isdir(os.path.join(args.imgdir, poke)):
            type1, type2 = poke_to_type[poke]
            if args.type == 1:
                type_to_poke[type1].append(poke)
            else:
                type_to_poke[type2].append(poke)

    # 各タイプから画像を取り出す
    for type in type_to_poke:
        if len(type_to_poke[type]) == 0:
            continue
        for i in range(NO_OF_POKE_EACHTYPE):
            poke = random.choice(type_to_poke[type])
            image_list = glob.glob(os.path.join(args.imgdir, poke) + '/*.png')
            print(type + ' : ' + str(i))
            filename = random.choice(image_list)
            img_name = os.path.basename(filename)

            if not os.path.exists(args.dstdir + '/type' + str(args.type) + '/' + img_name.split('_')[0]):
                os.makedirs(args.dstdir + '/type' + str(args.type) + '/' + img_name.split('_')[0])

            src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            image = add_background(scale_augmentation(src))

            cv2.imwrite(args.dstdir + '/type' + str(args.type) + '/' + img_name.split('_')[0]
                        + '/' + img_name.split('_')[0] + '_' + str(i) + '.png', image)
