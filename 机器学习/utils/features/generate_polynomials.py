"""生成多项式特征模块"""

import numpy as np


def generate_polynomials(data, polynomial_degree, normalize_data=False):
    """
    为数据集生成多项式特征
    
    :param data: 输入数据，形状为 (样本数, 特征数)
    :param polynomial_degree: 多项式最高次数
    :param normalize_data: 是否对生成的多项式特征进行归一化
    :return: 生成的多项式特征，形状为 (样本数, 多项式特征数)
    """
    num_examples = data.shape[0]
    num_features = data.shape[1]
    
    # 存储生成的多项式特征
    polynomials = np.empty((num_examples, 0))
    
    # 为每个原始特征生成多项式特征
    for degree in range(2, polynomial_degree + 1):
        for feature_idx in range(num_features):
            # 计算 x^degree
            poly_feature = np.power(data[:, feature_idx], degree)
            poly_feature = poly_feature.reshape(-1, 1)
            
            # 可选：对多项式特征进行归一化
            if normalize_data:
                mean_val = np.mean(poly_feature)
                std_val = np.std(poly_feature)
                std_val = std_val if std_val != 0 else 1
                poly_feature = (poly_feature - mean_val) / std_val
            
            polynomials = np.concatenate((polynomials, poly_feature), axis=1)
    
    return polynomials