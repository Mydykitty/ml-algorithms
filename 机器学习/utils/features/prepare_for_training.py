"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    准备训练数据
    :param data: 原始特征数据
    :param polynomial_degree: 多项式特征次数
    :param sinusoid_degree: 正弦特征次数
    :param normalize_data: 是否归一化
    :return: 处理后的数据, 特征均值, 特征标准差
    """
    # 计算样本总数
    num_examples = data.shape[0]
    
    # 复制原始数据
    data_processed = np.copy(data)
    
    # 初始化均值和标准差
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    
    # 归一化处理
    if normalize_data:
        data_normalized, features_mean, features_deviation = normalize(data_processed)
    
    data_processed = data_normalized
    
    # 特征变换 - 正弦特征
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)
    
    # 特征变换 - 多项式特征
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)
    
    # 添加偏置列（x0 = 1）
    num_examples = data_processed.shape[0]
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))
    
    return data_processed, features_mean, features_deviation