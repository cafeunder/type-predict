# coding:utf-8
# TODO 正しく動きません
"""
imgディレクトリにある画像と辞書を照合し、辞書にない画像ファイルを検出する
"""

import os

dict = {}
for line in open('../dictionary.csv', 'r'):
    row = line.split(',')
    dict[row[0]] = row[1] + ',' + row[2][:-1]

not_exist = {}
files = os.listdir('../img')
# すべての画像を読み込み、最小の矩形領域を抽出
for filename in files:
    # 拡張子とファイル名に分割
    base, ext = os.path.splitext(filename)
    if not ext == '.png': continue

    # 名前-フォルムの部分を抽出
    name = base.split('_')[0]
    if not name in dict:
        not_exist[name] = 0

# 辞書に存在しないポケモンを出力
not_exist_file = open('../not_exist.csv', 'w')
for key in not_exist:
    name_form = key.split('-')
    not_exist_file.write(name_form[0] + ',' + key + '\n')
not_exist_file.close()
