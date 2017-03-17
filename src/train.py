# coding:utf-8
"""
学習をさせるプログラム
"""
import os
import argparse
import numpy as np
import chainer
import chainer.links as L
from chainer import optimizers, iterators, training
from chainer.training import extensions

from models import alexnet
from dataset.preprocessed_dataset import PreprocessedDataset

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin'


def main():
    parser = argparse.ArgumentParser(description='Type Prediction: Pokemon')
    parser.add_argument('--learn_type', type=int, default=1,
                        help='Type number to learn')
    parser.add_argument('--train', default='../train.txt',
                        help='Path to training image-label list file')
    parser.add_argument('--val', default='../test.txt',
                        help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--mean', '-m', default='../mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='../result',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

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
    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean)
    val = PreprocessedDataset(args.val, args.root, mean)

    # 学習データとテストデータの定義
    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize)
    val_iter = iterators.SerialIterator(val, batch_size=args.val_batchsize,
                                        repeat=False, shuffle=False)

    # 最適化手法の設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 学習の設定
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

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
        os.path.join(args.out, 'model_final_type' + str(args.learn_type)),
        model)


if __name__ == '__main__':
    main()
