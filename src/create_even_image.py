# coding:utf-8
import os
import argparse
import random
import glob
from create_train_image import *

if __name__ == '__main__':
    NO_OF_POKE_EACHTYPE = 2500

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
    type_list = type_file.read().split("\n")  # 存在するタイプのリスト（和名，英名）
    type_to_poke = {}
    for type in type_list:
        # 存在するタイプを辞書に登録
        if type == "":
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
            image_list = glob.glob(os.path.join(args.imgdir, poke) + "/*.png")
            print(type + " : " + str(i))
            filename = random.choice(image_list)
            img_name = os.path.basename(filename)

            if not os.path.exists(args.dstdir + "/type" + str(args.type) + "/" + img_name.split("_")[0]):
                os.makedirs(args.dstdir + "/type" + str(args.type) + "/" + img_name.split("_")[0])

            src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            image = add_background(scale_augmentation(src))

            cv2.imwrite(args.dstdir + "/type" + str(args.type) + "/" + img_name.split("_")[0]
                        + "/" + img_name.split('_')[0] + "_" + str(i) + ".png", image)
