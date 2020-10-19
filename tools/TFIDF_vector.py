from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


def feature_vector(sentences: list, dimension=-1):
    """
    用TF——IDF来表征数据
    :param sentences: 文本分词后的形式
    :param dimension: PCA降维后的维度;-1时表示不进行降维处理
    """
    # 词频矩阵 Frequency Matrix Of Words
    # sublinear_tf,是否应用子线性tf缩放，即用1 + log（tf）替换tf；
    # max_df：float in range [ 0.0，1.0 ]或int，default = 1.0；当构建词汇时，忽略文档频率严格高于给定阈值（语料库特定停止词）的术语。
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vertorizer.fit_transform(sentences)
    # Get Words Of Bag
    # words = vertorizer.get_feature_names()
    # tfidf = transformer.fit_transform(freq_words_matrix)
    # w[i][j] represents word j's weight in text class i
    weight = freq_words_matrix.toarray()
    # 将词频矩阵降维
    if dimension > 0:
        pca = PCA(n_components=dimension)
        training_data = pca.fit_transform(weight)
        return training_data


def tfidf_class_vaector(sentences: list):
    """
    根据训练集训练TF-IDF模型,并保存模型
    :param sentences:
    :return:
    """
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(sentences)
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    return train_tfidf, count_vect
