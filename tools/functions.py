#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
from tools.tokenizer import *
import pandas as pd
import tqdm


def load_cut_dat(file_name):
    """
    加载分词后的文件,返回list形式的分词
    :param file_name: 分词文件
    :return: 分词list
    """
    # 加载数据
    data = pd.read_csv(file_name, sep=',', encoding='utf-8')
    data.dropna(inplace=True)
    # 将分词数据变为list形式
    cut = []
    for d in data["title"]:
        line = [word.strip() for word in d.split(' ')][:-1]
        cut.append(line)
    return data['label1'], data['label2'], cut


def load_tfidf_cut(file_name):
    """
    加载TF-IDF词向量化所需要的的分词数据
    :param file_name: 分词文件
    :return: 顶级标签，次级标签，文本分词内容
    """
    # 加载数据
    data = pd.read_csv(file_name, sep=',', encoding='utf-8')
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    return data['label1'], data['label2'], data['title']



""" 
if __name__ == "__main__":
    # train_data = pd.read_csv(settings.SOURCE_DATA+'10000_train_data.csv')
    test_data = pd.read_csv(settings.SOURCE_DATA+'1000_test_data.csv')
    # valid_data = pd.read_csv(settings.SOURCE_DATA + '1000_valid_data.csv')
    Cut = ChinesePreprocessor(stopwords_path=settings.STATIC_DIR+'baidu_stopwords.txt')
    new_data = []
    for i in range(test_data.shape[0]):
        L = []
        # L.append(valid_data['label1'][i])
        L.append(test_data['label1_number'][i])
        # L.append(valid_data['label2'][i])
        L.append(test_data['label2_number'][i])
        word = Cut.word_cut(text=test_data['title'][i])
        L.append(word)
        new_data.append(L)
    new_data = pd.DataFrame(data=new_data, columns=['label1', 'label2', 'title'])
    new_data.dropna(inplace=True)
    new_data.to_csv(settings.SOURCE_DATA + '1000_test_cut.csv', index=None, encoding='utf-8')
"""
