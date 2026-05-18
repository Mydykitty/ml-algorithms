import numpy as np
from 机器学习.utils.features.prepare_for_training import prepare_for_training


class LinearRegression:
    """
    线性回归模型
    1. 对数据进行预处理操作
    2. 先得到所有的特征个数
    3. 初始化参数矩阵
    """

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        初始化线性回归模型
        :param data: 原始特征数据
        :param labels: 标签数据
        :param polynomial_degree: 多项式特征次数
        :param sinusoid_degree: 正弦特征次数
        :param normalize_data: 是否归一化
        """
        # 数据预处理
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(
            data,
            polynomial_degree=0,
            sinusoid_degree=0,
            normalize_data=True
        )

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 初始化参数矩阵
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return: 训练后的参数，成本历史记录
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):  # 梯度下降参数更新计算方法，注意是矩阵运算
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]
        # 正确方式：通过类名调用静态方法
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        # 梯度下降更新公式
        theta = self.theta - alpha * (1 / num_examples) * np.dot(self.data.T, delta)
        self.theta = theta

    def cost_function(self, data, labels):
        """
        计算成本函数（均方误差的一半）
        """
        num_examples = self.data.shape[0]
        delta = LinearRegression.hypothesis(data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        假设函数：h(x) = X·θ
        静态方法，不需要self参数
        """
        return np.dot(data, theta)

    def get_cost(self, data, labels):
        """
        计算成本函数（均方误差的一半）
        """
        data_processed = prepare_for_training(
            data,
            polynomial_degree=self.polynomial_degree,
            sinusoid_degree=self.sinusoid_degree,
            normalize_data=self.normalize_data
        )[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        使用训练好的模型进行预测
        :param data: 待预测的数据
        :return: 预测结果
        """
        # 使用保存的均值和标准差对数据进行相同的预处理
        data_processed = prepare_for_training(
            data,
            polynomial_degree=self.polynomial_degree,
            sinusoid_degree=self.sinusoid_degree,
            normalize_data=self.normalize_data
        )[0]

        # 正确方式：通过类名调用静态方法
        return LinearRegression.hypothesis(data_processed, self.theta)
