# coding:utf-8
"""
学習をさせるプログラム
"""
import chainer
import chainer.links as L
from chainer import optimizers, iterators, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions

from src.models import alexnet
from src.models.mlp import MLP

# 学習モデル
# model = L.Classifier(alexnet.Alex())
model = L.Classifier(MLP())

# 最適化手法の設定
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

# データとラベルの紐付け
# train = tuple_dataset.TupleDataset(x_train, y_train)
# test = tuple_dataset.TupleDataset(x_test, y_test)
train, test = chainer.datasets.get_mnist()

# 学習データとテストデータの定義
train_iter = iterators.SerialIterator(train, batch_size=100)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False,
                                     shuffle=False)

# 学習の設定
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
# trainer = training.Trainer(updater, (10000, 'iteration'), out='result')

# 学習後の評価の設定
# エポック終了毎に評価される
trainer.extend(extensions.Evaluator(test_iter, model))

# ログの記録
trainer.extend(extensions.dump_graph('main/loss'))  # 学習曲線のグラフを保存
trainer.extend(extensions.LogReport())  # ログを保存
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
     'validation/main/accuracy']))  # ログを出力
trainer.extend(extensions.ProgressBar())  # プログレスバー

# 学習の実行
trainer.run()
