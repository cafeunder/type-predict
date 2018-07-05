# coding:utf-8
import os
import argparse
import glob
from create_train_image import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--imgdir', default='../train_data')
    parser.add_argument('--dictdir', default='../dictionary.csv')
    parser.add_argument('--label', default='../labels.txt')
    parser.add_argument('--type', type=int, default=1)
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
        for poke in type_to_poke[type]:
            image_list = glob.glob(args.imgdir + '/' + poke + '/*.png')
            total_rate = 0
            count = 0
            for image_name in image_list:
                src = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                height = len(src)
                width = len(src[0])
                total_rate += float(height) / width
                count += 1
            print(poke + ' : ' + str(total_rate / count))
    print(count)
