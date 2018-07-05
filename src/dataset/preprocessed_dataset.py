# coding:utf-8
"""
学習に用いるデータセットを定義するクラス
"""
import chainer


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, mean):
        self.base = chainer.datasets.LabeledImageDataset(path)
        self.mean = mean.astype('f')

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image -= self.mean
        image *= (1.0 / 255.0)
        return image, label
