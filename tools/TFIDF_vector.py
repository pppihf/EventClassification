from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA


def feature_vector(sentences: list, dimension: int):
    """
    用TF——IDF来表征数据
    :param sentences: 文本分词后的形式
    :param dimension: PCA降维后的维度
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
    pca = PCA(n_components=dimension)
    training_data = pca.fit_transform(weight)
    return training_data