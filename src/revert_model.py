# coding:utf-8
'''
githubのファイル制限に引っかからないよう、モデルデータを分割する
'''
import glob

size = 1024 * 1024 * 99  # 99MB

# サブファイルをマージして、モデルデータを復元する
def merge_model(basename, modelname):
    chunk_list = glob.glob(basename + "/chunk*")
    dst = open(modelname, "wb")
    for chunk in chunk_list:
        print(chunk)
        file = open(chunk, "rb")
        bin = file.read()
        dst.write(bin)
    dst.close()


if __name__ == "__main__":
    merge_model("../data/type1", "../data/type1/model_final")
    merge_model("../data/type2", "../data/type2/model_final")
