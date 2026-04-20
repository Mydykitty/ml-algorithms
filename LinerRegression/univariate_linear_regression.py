import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import numpy as np

# 读取CSV数据文件
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 划分训练集和测试集（80% 训练，20% 测试）
train_data = data.sample(frac=0.8)      # 随机抽取80%作为训练集
test_data = data.drop(train_data.index) # 剩余20%作为测试集

# 指定特征列和标签列
input_param_name = 'Economy..GDP.per.Capita.'   # 输入特征：人均GDP
output_param_name = 'Happiness.Score'          # 输出标签：幸福指数

# 提取训练数据
x_train = train_data[[input_param_name]].values  # 训练特征
y_train = train_data[[output_param_name]].values # 训练标签

# 提取测试数据
x_test = test_data[input_param_name].values    # 测试特征
y_test = test_data[output_param_name].values   # 测试标签

# 可视化：绘制散点图
plt.scatter(x_train, y_train, label='Train data', color='blue', alpha=0.7)
plt.scatter(x_test, y_test, label='Test data', color='red', alpha=0.7)
plt.xlabel(input_param_name)      # X轴标签
plt.ylabel(output_param_name)     # Y轴标签
plt.title('Happiness Score vs GDP per Capita')  # 图表标题
plt.legend()                       # 显示图例
plt.show()                         # 展示图形

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失: ', cost_history[0])
print('训练后的损失: ', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100

x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)

# 预测
y_predictions = linear_regression.predict(x_predictions)

# 可视化
plt.scatter(x_train, y_train, label='Train data', alpha=0.7)
plt.scatter(x_test, y_test, label='Test data', alpha=0.7)
plt.plot(x_predictions, y_predictions, 'r', label='Prediction', linewidth=2)
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()