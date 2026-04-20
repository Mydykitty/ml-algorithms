import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成正弦特征
    :param dataset: 输入数据，形状为 (样本数, 特征数)
    :param sinusoid_degree: 正弦特征的最高次数
    :return: 生成的正弦特征
    """

    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids
