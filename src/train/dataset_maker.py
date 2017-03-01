# coding:utf-8
"""
学習に用いるデータセットを作成するクラス
"""
import argparse
import glob
import os
import random


def main():
    parser = argparse.ArgumentParser(description='Make Dataset')
    parser.add_argument('--train', default='../../train.txt',
                        help='Path to training image-label list file')
    parser.add_argument('--val', default='../../test.txt',
                        help='Path to validation image-label list file')
    parser.add_argument('--dictdir', default='../../dictionary.csv',
                        help='Path to dictionary file (Pokemon to Type)')
    parser.add_argument('--label', default='../../labels.txt',
                        help='Path to label file')
    parser.add_argument('--imgdir', default='../../train_data/',
                        help='Path to directory of image file')
    args = parser.parse_args()

    poke_names = []  # ポケモンの名前のリスト
    # 画像データとして存在するポケモンの名前をリストに追加
    for label in os.listdir(args.imgdir):
        if os.path.isdir(os.path.join(args.imgdir, label)):
            poke_names.append(label)

    train = open(args.train, 'w')  # 訓練データのテキストファイル
    test = open(args.val, 'w')  # テストデータのテキストファイル

    # タイプを受け取るとint型の番号を返す辞書を作成
    type_file = open(args.label, 'r')  # 存在するタイプを記述してあるファイル
    type_to_int = {}  # タイプ→番号の辞書
    type_list = type_file.read().split("\n")  # 存在するタイプのリスト（和名，英名）
    cnt = 0
    for type in type_list:
        # 存在するタイプを辞書に登録
        if type == "":
            break
        type_to_int[type] = cnt
        cnt += 1

    # ポケモン名を受け取るとタイプを返す辞書を作成
    poke_dict_file = open(args.dictdir, 'r')  # ポケモンとそのタイプを記述してあるファイル
    poke_to_type = {}  # ポケモン→タイプの辞書
    poke_dict = poke_dict_file.read().split(
        "\n")  # ポケモンとそのタイプのリスト（ポケモン，第一タイプ，第二タイプ）
    cnt = 0
    for poke in poke_dict:
        # ポケモンを辞書に登録
        if len(poke.split(",")) < 3:
            break
        poke_to_type[poke.split(",")[0]] = poke.split(",")[1]

    # （ポケモンの画像のパス，タイプ番号）をデータとする訓練データとテストデータを作成
    cnts = [0 for i in range(len(type_list))]
    for poke_name in poke_names:
        print(poke_name)
        # 各ポケモンの画像を全て取得
        image_list = glob.glob(os.path.join(args.imgdir, poke_name) + "/*.png")
        cnt = 0
        # 訓練データとテストデータを作成
        for image in image_list:
            # 各ポケモンの画像の3/4は訓練データに，1/4はテストデータにする
            cnts[type_to_int[poke_to_type[poke_name]]] += 1
            if random.uniform(0.0, 1.0) < 0.75:
                train.write(
                    image + " " + str(
                        type_to_int[poke_to_type[poke_name]]) + "\n")
            else:
                test.write(
                    image + " " + str(
                        type_to_int[poke_to_type[poke_name]]) + "\n")
            cnt += 1

    for i in range(len(type_list)):
        print(type_list[i], cnts[i])
    type_file.close()
    train.close()
    test.close()


if __name__ == '__main__':
    main()
