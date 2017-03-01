# coding:utf-8
'''
githubのファイル制限に引っかからないよう、モデルデータを分割する
'''
import glob

size = 1024 * 1024 * 99 # 99MB

# モデルデータを読み込み、サブファイルに分割する
def split_model(filename):
    file = open(filename, "rb")
    bin = file.read()
    print(len(bin))

    chunk = [bin[i:i + size] for i in range(0, len(bin), size)]

    count = 0
    for li in chunk:
        dst = open("../../premade_model/chunk_{0}".format(count), "wb")
        dst.write(li)
        dst.close()
        count += 1


# サブファイルをマージして、モデルデータを復元する
def merge_model(basename):
    chunk_list = glob.glob("../../premade_model/" + basename + "*")
    dst = open("../../model_final", "wb")
    for chunk in chunk_list:
        file = open(chunk, "rb")
        bin = file.read()
        dst.write(bin)
    dst.close()


if __name__ == "__main__":
    # Trueなら分割したモデルデータを復元
    # Falseならモデルデータを分割する
    # 利用時は書き換える必要ない
    merge = False
    if merge:
        merge_model("../../premade_model")
    else:
        split_model("../../model_final")
