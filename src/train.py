# coding:utf-8
"""
学習をさせるプログラム
"""
import argparse

import chainer
import chainer.links as L
from chainer import optimizers, iterators, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions

from src.models import alexnet
from src.models.mlp import MLP


def main():
    parser = argparse.ArgumentParser(description='Type Prediction: Pokemon')
    parser.add_argument('--train', default='img',
                        help='Path to training image-label list file')
    parser.add_argument('--val', default='img',
                        help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--out', '-o', default='../result',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    args = parser.parse_args()

    # 学習モデル
    # model = L.Classifier(alexnet.Alex(18))
    model = L.Classifier(MLP())
    if args.initmodel:
        # 学習済みのモデルを使う場合，読み込み
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        # GPUを使用する場合，GPUメソッドを指定
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # 最適化手法の設定
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # データとラベルの紐付け
    # train = PreprocessedDataset(args.train, args.root, model.insize)
    # val = PreprocessedDataset(args.val, args.root, model.insize)
    train, val = chainer.datasets.get_mnist()

    # 学習データとテストデータの定義
    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize)
    val_iter = iterators.SerialIterator(val, batch_size=args.val_batchsize,
                                        repeat=False, shuffle=False)

    # 学習の設定
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # 学習後の評価の設定
    # エポック終了毎に評価される
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))

    # ログの記録
    trainer.extend(extensions.dump_graph('main/loss'))  # 学習曲線のグラフを保存
    trainer.extend(extensions.LogReport())  # ログを保存
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy']))  # ログを出力
    trainer.extend(extensions.ProgressBar(update_interval=500))  # プログレスバー

    # 学習の実行
    trainer.run()


if __name__ == '__main__':
    main()
