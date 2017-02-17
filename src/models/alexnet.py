# coding:utf-8
'''
AlexNetのクラス
'''
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class Alex(Chain):
    """
    AlexNetの出力層のみ変更したクラス．
    任意の分類クラスに対応．
    """

    insize = 227

    def __init__(self, out_size, train=True):
        super(Alex, self).__init__(
            # 第一層：入力のチャンネル数=3，出力のチャンネル数=96，フィルターのサイズ=(11,11)，ストライド=4
            conv1=L.Convolution2D(3, 96, 11, stride=4),
            # 第二層：入力のチャンネル数=96，出力のチャンネル数=256，フィルターのサイズ=(5,5)，パディングの幅=2
            conv2=L.Convolution2D(96, 256, 5, pad=2),
            # 第三層：入力のチャンネル数=256，出力のチャンネル数=384，フィルターのサイズ=(3,3)，パディングの幅=1
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            # 第四層：入力のチャンネル数=384，出力のチャンネル数=384，フィルターのサイズ=(3,3)，パディングの幅=1
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            # 第五層：入力のチャンネル数=384，出力のチャンネル数=256，フィルターのサイズ=(3,3)，パディングの幅=1
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            # 第六層：入力次元数=9216，出力次元数=4096
            fc6=L.Linear(9216, 4096),
            # 第七層：入力次元数=4096，出力次元数=4096
            fc7=L.Linear(4096, 4096),
            # 第八層：入力次元数=4096，出力次元数=分類クラス数
            fc8=L.Linear(4096, out_size=out_size),
        )
        self.train = train

    def __call__(self, x):
        """
        forwardが呼び出された際に呼ばれるメソッド．
        入力データの分類結果（各クラスの確率）を返す．
        :param x: 訓練データ
        :return: 分類結果
        """
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h
