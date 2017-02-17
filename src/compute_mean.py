# coding:utf-8
import argparse
import sys

import numpy as np

import chainer


def compute_mean(dataset):
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
    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('--dataset', default='../train.txt',
                        help='Path to training image-label list file')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--output', '-o', default='../mean_train.npy',
                        help='path to output mean array')
    args = parser.parse_args()

    dataset = chainer.datasets.LabeledImageDataset(args.dataset, args.root)
    mean = compute_mean(dataset)
    np.save(args.output, mean)


if __name__ == '__main__':
    main()
