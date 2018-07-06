# coding:utf-8
"""
githubのファイル制限に引っかからないよう、モデルデータを分割する
"""
import os
import argparse

size = 1024 * 1024 * 99  # 99MB


# モデルデータを読み込み、サブファイルに分割する
def divide_model(filename, outdir):
    file = open(filename, 'rb')
    bin = file.read()
    print('model size : ' + str(len(bin)))

    chunk = [bin[i:i + size] for i in range(0, len(bin), size)]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count = 0
    for li in chunk:
        dst = open(outdir + '/chunk_{0}'.format(count), 'wb')
        dst.write(li)
        dst.close()
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide model file')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--output', help='Path to output directory')
    args = parser.parse_args()

    divide_model(args.model, args.output)
