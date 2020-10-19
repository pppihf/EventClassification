#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
from tools.functions import *
import settings
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tools.TFIDF_vector import *
from sklearn.externals import joblib


class HIERARCHICAL_BAYES:
    def __init__(self, layers=2):
        self.layers = layers

    @staticmethod
    def simple_layer(data, labels, alpha=1, prior=True, c_prior=None):
        """
        用传入的训练数据训练多项式贝叶斯；@staticmethod保证了，不需要实例化也可直接调用该函数
        :param data:  传入模型的数据
        :param labels: 传入模型的标签
        :param alpha: 先验平滑因子，默认等于1，当等于1时表示拉普拉斯平滑。
        :param prior: 是否去学习类的先验概率，默认是True
        :param c_prior: 各个类别的先验概率，如果没有指定，则模型会根据数据自动学习;
        每个类别的先验概率相同，等于类标记总个数N分之一。
        :return: 训练好的模型
        """
        model = MultinomialNB(alpha=alpha, fit_prior=prior, class_prior=c_prior)
        model = model.fit(data, labels)
        return model

    @staticmethod
    def save_model(model, save_path):
        joblib.dump(model, save_path)  # 根据传入的路径信息，保存model

    @staticmethod
    def load_model(save_path):
        model = joblib.load(save_path)  # 根据传入的路径信息，加载model
        return model

    def train(self, data, first_label, second_label, save_path):
        """
        训练分层分类贝叶斯
        :param data: 训练数据
        :param first_label: 数据的第一级目录
        :param second_label: 数据的第二级标签
        :return:
        """
        # 用一个dict来容纳不同的model，并进行标记
        model_dict = {}
        # 首先调用贝叶斯进行一次分类
        clf = self.simple_layer(data=data, labels=first_label)
        model_dict['top'] = clf
        # 对于每类数据进行细分，训练多个模型
        if self.layers == 2:
            for i in set(first_label):  # 按大类划分
                # 找到每个类别对应的index, 从而根据索引找到训练数据和标签
                small_data = []
                small_label = []
                flag = 0
                for j in first_label:
                    if j == i:
                        r = data[flag].todense()
                        small_data.append(r.getA()[0])
                        small_label.append(second_label[flag])
                    flag += 1
                # 调用单层贝叶斯，根据小数据进行训练
                model = self.simple_layer(small_data, small_label)
                model_dict[i] = model
        # 存储生成的模型
        self.save_model(model_dict, save_path)

    def predict(self, data, model_dict):
        """
        调用分层模型，进行预测
        :param data: 测试数据
        :param model_dict: 贝叶斯分层模型
        :return: 数据预测标签
        """
        # 先进行第一次分层
        clf = model_dict['top']
        top_label = []
        if self.layers == 1:
            for d in data:
                r = clf.predict(d)
                top_label.append(r[0])
            return top_label
        if self.layers == 2:
            second_label = []
            for d in data:
                r = clf.predict(d)
                top_label.append(r[0])
                model = model_dict[r[0]]
                second_label.append(model.predict(d)[0])
            return top_label, second_label
        else:
            print('Parameter layers Values Error')


if __name__ == '__main__':
    # 加载训练用的分词数据
    train_file = settings.SOURCE_DATA+'10000_train_cut.csv'
    train_data = pd.read_csv(train_file)
    train_top_label, train_second_label, train_cut = load_tfidf_cut(train_file)
    # 调用TF_IDF生成词的向量形式
    train_vector, count_vect = tfidf_class_vaector(train_cut)
    # 加载测试集
    test_file = settings.SOURCE_DATA + '1000_test_cut.csv'
    test_top_label, test_second_label, test_cut = load_tfidf_cut(test_file)
    test_vector = count_vect.transform(test_cut)
    # 尝试多层分类
    HB = HIERARCHICAL_BAYES()
    HB.train(train_vector, train_top_label,
             train_second_label, settings.BAYES_DIR+'HBayes.joblib')
    model_dict = HB.load_model(settings.BAYES_DIR+'HBayes.joblib')
    top_predict, second_predict = HB.predict(test_vector, model_dict)
    # 输出多分类结果
    print('第一层分类结果')
    print(classification_report(test_top_label, top_predict))
    print('第二层分类结果')
    print(classification_report(test_second_label, second_predict))



