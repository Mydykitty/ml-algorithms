# utils/hypothesis.py

import numpy as np

def sigmoid(z):
    """
    Sigmoid激活函数
    将输入映射到(0,1)区间
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    """
    Sigmoid函数的导数
    g'(z) = g(z) * (1 - g(z))
    用于反向传播计算梯度
    """
    g = sigmoid(z)
    return g * (1 - g)