# coding:utf-8
"""
学習に用いるデータセットを作成するクラス
"""
import glob
import os
import random

path = "../data"  # 画像データが存在するパス
poke_names = []  # ポケモンの名前のリスト
# 画像データとして存在するポケモンの名前をリストに追加
for label in os.listdir(path):
    if os.path.isdir(os.path.join(path, label)):
        poke_names.append(label)

train = open('../train.txt', 'w')  # 訓練データのテキストファイル
test = open('../test.txt', 'w')  # テストデータのテキストファイル
labelsText = open('../labels.txt', 'w')  # ラベルのテキストファイル

# タイプを受け取るとint型の番号を返す辞書を作成
type_file = open("../type_table.csv", 'r')  # 存在するタイプを記述してあるファイル
type_to_int = {}  # タイプ→番号の辞書
type_list = type_file.read().split("\n")  # 存在するタイプのリスト（和名，英名）
cnt = 0
for type in type_list:
    # 存在するタイプを辞書に登録
    if len(type.split(",")) < 2:
        break
    labelsText.write(type.split(",")[1] + "\n")
    type_to_int[type.split(",")[1]] = cnt
    cnt += 1

# ポケモン名を受け取るとタイプを返す辞書を作成
poke_dict_file = open("../dictionary.csv", 'r')  # ポケモンとそのタイプを記述してあるファイル
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
    image_list = glob.glob(os.path.join(path, poke_name) + "/*.png")
    cnt = 0
    # 訓練データとテストデータを作成
    for image in image_list:
        # 各ポケモンの画像の3/4は訓練データに，1/4はテストデータにする
        if random.uniform(0.0, 1.0) < 0.75:
            train.write(
                image + " " + str(type_to_int[poke_to_type[poke_name]]) + "\n")
        else:
            cnts[type_to_int[poke_to_type[poke_name]]] += 1
            test.write(
                image + " " + str(type_to_int[poke_to_type[poke_name]]) + "\n")
        cnt += 1

print(cnts)
type_file.close()
train.close()
test.close()
labelsText.close()
