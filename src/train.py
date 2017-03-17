# coding:utf-8
"""
学習をさせるプログラム
"""
import glob
import os
import argparse
import random

import numpy as np
import chainer
import chainer.links as L
import sys
from chainer import optimizers, iterators, training
from chainer.training import extensions

from models import alexnet
from dataset.preprocessed_dataset import PreprocessedDataset

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin'


def make_dataset(learn_type, train_file, val_file, label_file, dict_dir,
                 imgdir):
    poke_names = []  # ポケモンの名前のリスト
    # 画像データとして存在するポケモンの名前をリストに追加
    for name in os.listdir(imgdir):
        if os.path.isdir(os.path.join(imgdir, name)):
            poke_names.append(name)

    train = open(train_file, 'w')  # 訓練データのテキストファイル
    test = open(val_file, 'w')  # テストデータのテキストファイル

    # タイプを受け取るとint型の番号を返す辞書を作成
    type_file = open(label_file, 'r')  # 存在するタイプを記述してあるファイル
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
    poke_dict_file = open(dict_dir, 'r')  # ポケモンとそのタイプを記述してあるファイル
    poke_to_type = {}  # ポケモン→タイプの辞書
    poke_dict = poke_dict_file.read().split(
        "\n")  # ポケモンとそのタイプのリスト（ポケモン，第一タイプ，第二タイプ）
    for poke in poke_dict:
        record = poke.split(',')
        # ポケモンを辞書に登録
        if len(record) < 3:
            break
        if learn_type == 1:
            poke_to_type[record[0]] = record[1]
        else:
            record[2] = record[2].replace('\n', '')
            if record[2]:
                poke_to_type[record[0]] = record[2]
            else:
                poke_to_type[record[0]] = "none"
    # （ポケモンの画像のパス，タイプ番号）をデータとする訓練データとテストデータを作成
    cnts = [0 for i in range(len(type_list))]
    for poke_name in poke_names:
        print(poke_name)
        # 各ポケモンの画像を全て取得
        image_list = glob.glob(os.path.join(imgdir, poke_name) + "/*.png")
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


def compute_mean(train_file, output_dir):
    # 訓練データのパスからデータセットを生成
    dataset = chainer.datasets.LabeledImageDataset(train_file)
    # 平均画像を計算
    # npy形式で保存
    print('compute mean image')
    mean = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        mean += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    np.save(output_dir, mean / N)


def main():
    parser = argparse.ArgumentParser(description='Type Prediction: Pokemon')
    parser.add_argument('--learn_type', type=int, default=1,
                        help='Type number to learn')
    parser.add_argument('--imgdir', default='../image/',
                        help='Path to directory of image file')
    parser.add_argument('--root', default='../data', help='')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # 学習に用いるデータセットを作成
    make_dataset(args.learn_type,
                 args.root + '/type' + str(args.learn_type) + '/train.txt',
                 args.root + '/type' + str(args.learn_type) + '/test.txt',
                 '../labels.txt', '../dictionary.csv', args.imgdir)

    # 学習に用いる画像の平均画像を計算
    compute_mean(args.root + '/type' + str(args.learn_type) + '/train.txt',
                 args.root + '/type' + str(args.learn_type) + '/mean.npy')

    # 学習モデル
    if args.learn_type == 1:
        model = alexnet.Alex(18)
    else:
        model = alexnet.Alex(19)
    model = L.Classifier(model)
    if args.initmodel:
        # 学習済みのモデルを使う場合，読み込み
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        # GPUを使用する場合，GPUメソッドを指定
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # データとラベルの紐付け
    mean = np.load(args.root + '/type' + str(args.learn_type) + '/mean.npy')
    train = PreprocessedDataset(
        args.root + '/type' + str(args.learn_type) + '/train.txt', mean)
    val = PreprocessedDataset(
        args.root + '/type' + str(args.learn_type) + '/test.txt', mean)

    # 学習データとテストデータの定義
    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize)
    val_iter = iterators.SerialIterator(val, batch_size=args.val_batchsize,
                                        repeat=False, shuffle=False)

    # 最適化手法の設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 学習の設定
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                               out=args.root + "/type" + str(args.learn_type))

    val_interval = (10 if args.test else 100000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    # 学習後の評価の設定
    # エポック終了毎に評価される
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))

    # ログの記録
    trainer.extend(extensions.dump_graph('main/loss'))  # 学習曲線のグラフを保存
    trainer.extend(extensions.snapshot(), trigger=val_interval)  # 一定間隔でプログレスを保存
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'),
        trigger=val_interval)  # 一定間隔でモデルを保存
    trainer.extend(extensions.LogReport(trigger=log_interval))  # 一定間隔でログを保存
    trainer.extend(extensions.observe_lr(), trigger=log_interval)  # 一定間隔で学習率を保存
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy', 'lr']), trigger=log_interval)  # ログを出力
    trainer.extend(extensions.ProgressBar(update_interval=10))  # プログレスバー

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # 学習の実行
    trainer.run()

    # 学習後のモデルの保存
    chainer.serializers.save_npz(
        os.path.join(args.root + '/type' + str(args.learn_type),
                     'model_final_type'), model)


if __name__ == '__main__':
    main()
