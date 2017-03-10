# coding:utf-8
"""
訓練データの平均画像を生成するクラス．
出力は拡張子npy．
"""
import argparse
import sys
import numpy as np
import chainer


def compute_mean(dataset):
    """
    データセット内の画像の平均を計算するメソッド．
    :param dataset: データセット
    :return: 画像の平均値
    """
    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    return sum_image / N


def main():
    """
    訓練データ内の平均画像をnpy形式で出力するメソッド．
    :return:
    """
    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('--dataset', default='../train.txt',
                        help='Path to training image-label list file')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--output', '-o', default='../mean.npy',
                        help='path to output mean array')
    args = parser.parse_args()

    # 訓練データのパスからデータセットを生成
    dataset = chainer.datasets.LabeledImageDataset(args.dataset, args.root)
    # 平均画像を計算
    mean = compute_mean(dataset)
    # npy形式で保存
    np.save(args.output, mean)


if __name__ == '__main__':
    main()
