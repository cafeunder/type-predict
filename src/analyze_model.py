# coding:utf-8
"""
学習したモデルをテストするプログラム
指定したフォルダ直下にある各ポケモンのフォルダ内の画像に対するaccuracyを計算します
"""
import os
from test import *

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin'


def main():
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--root', default='../data', help='')
    parser.add_argument('--label', default='../labels.txt',
                        help='Path to label file')
    parser.add_argument('--dictdir', default='../dictionary.csv',
                        help='Path to dictionary file (Pokemon to Type)')
    parser.add_argument('--imgdir', help='Path to image directory')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if args.gpu >= 0:
        # GPUを使用する場合，GPUメソッドを指定
        chainer.cuda.get_device(args.gpu).use()

    # 学習済みモデルの読み込み
    model_type1, out_size_type1 = make_model(args.root + '/type1/model_final', 1)
    model_type2, out_size_type2 = make_model(args.root + '/type2/model_final', 2)
    if args.gpu >= 0:
        model_type1.to_gpu()
        model_type2.to_gpu()

    # 平均画像の読み込み
    mean1 = np.load(args.root + '/type1/mean.npy')
    mean2 = np.load(args.root + '/type2/mean.npy')

    poke_names = []  # ポケモンの名前のリスト
    # 画像データとして存在するポケモンの名前をリストに追加
    for label in os.listdir(args.imgdir):
        if os.path.isdir(os.path.join(args.imgdir, label)):
            poke_names.append(label)

    # タイプを受け取るとint型の番号を返す辞書を作成
    type_file = open(args.label, 'r')  # 存在するタイプを記述してあるファイル
    type_to_int = {}  # タイプ→番号の辞書
    type_list = type_file.read().split('\n')  # 存在するタイプのリスト（和名，英名）
    cnt = 0
    for type in type_list:
        # 存在するタイプを辞書に登録
        if type == '':
            break
        type_to_int[type] = cnt
        cnt += 1

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

    success_count_image = [0, 0, 0]
    total_count_image = 0
    for poke_name in poke_names:
        total_count_poke = 0
        success_count_poke = [0, 0, 0]

        # 各ポケモンの画像を全て取得
        image_list = glob.glob(os.path.join(args.imgdir, poke_name) + '/*.png')
        # 訓練データとテストデータを作成
        for image in image_list:
            y_type1 = nine_test(image, mean1, model_type1, args.gpu)
            y_type2 = nine_test(image, mean2, model_type2, args.gpu)

            if y_type1 == type_to_int[poke_to_type[poke_name][0]]:
                success_count_image[0] += 1
                success_count_poke[0] += 1
            if y_type2 == type_to_int[poke_to_type[poke_name][1]]:
                success_count_image[1] += 1
                success_count_poke[1] += 1

            if y_type1 == type_to_int[poke_to_type[poke_name][0]]\
                    and y_type2 == type_to_int[poke_to_type[poke_name][1]]:
                success_count_image[2] += 1
                success_count_poke[2] += 1
            total_count_poke += 1

            total_count_image += 1
        print(poke_name + ' accuracy')
        print('type1['
              + str(int(float(success_count_poke[0]) / total_count_poke * 100))
              + '%] type2[' + str(int(float(success_count_poke[1]) / total_count_poke * 100))
              + '%] perfect[' + str(int(float(success_count_poke[2]) / total_count_poke * 100))
              + '%]')
    print('type1 accuracy : '
          + str(float(success_count_image[0]) / total_count_image))
    print('type2 accuracy : '
          + str(float(success_count_image[1]) / total_count_image))
    print('perfect accuracy : '
          + str(float(success_count_image[2]) / total_count_image))


if __name__ == '__main__':
    main()
