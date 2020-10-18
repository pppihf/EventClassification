#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
from tools.tokenizer import *
import pandas as pd
import tqdm


def load_cut_dat(file_name, labelname):
    """
    加载分词后的文件
    :param file_name: 分词文件
    :param labelname: 列名
    :return: 分词list
    """
    # 加载数据
    data = pd.read_csv(file_name, sep=',', encoding='utf-8')
    data.dropna(inplace=True)
    # 将分词数据变为list形式
    cut = []
    for d in data[labelname]:
        line = [word.strip() for word in d.split(' ')][:-1]
        cut.append(line)
    return cut


if __name__ == "__main__":
    train_data = pd.read_csv(settings.SOURCE_DATA+'10000_train_data.csv')
    test_data = pd.read_csv(settings.SOURCE_DATA+'1000_test_data.csv')
    valid_data = pd.read_csv(settings.SOURCE_DATA + '1000_valid_data.csv')
    Cut = ChinesePreprocessor(stopwords_path=settings.STATIC_DIR+'baidu_stopwords.txt')
    Cut.word_cut(train_data['title'])

