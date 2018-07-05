# coding:utf-8
"""
学習したモデルをテストするプログラム
"""
from test import *

def main():
    parser = argparse.ArgumentParser(description='Test Learned Model')
    parser.add_argument('--root', default='../data', help='')
    parser.add_argument('--label', default='../labels.txt',
                        help='Path to label file')
    parser.add_argument('--imgdir', help='Path to image file')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # 学習済みモデルの読み込み
    model_type1, out_size_type1 = make_model(args.root + '/type1/model_final',
                                             1)
    model_type2, out_size_type2 = make_model(args.root + '/type2/model_final',
                                             2)
    if args.gpu >= 0:
        model_type1.to_gpu()
        model_type2.to_gpu()

    # 平均画像の読み込み
    mean1 = np.load(args.root + '/type1/mean.npy')
    mean2 = np.load(args.root + '/type2/mean.npy')

    # 推定されたタイプを出力
    type_file = open(args.label, 'r')
    type_list = type_file.read().split("\n")

    max_type1 = {}
    max_type2 = {}
    for type in type_list:
        max_type1[type] = 0
        max_type2[type] = 0

    # 画像からタイプを予測
    image_list = glob.glob(args.imgdir + '/*.png')
    for image in image_list:
        type1 = nine_test(image, mean1, model_type1, args.gpu)
        type2 = nine_test(image, mean2, model_type2, args.gpu)
        max_type1[type_list[type1]] += 1
        max_type2[type_list[type2]] += 1

    for key in type_list:
        print("{0} : {1} {2}".format(key, max_type1[key], max_type2[key]))


if __name__ == '__main__':
    main()
