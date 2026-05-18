import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置列
X_b = np.c_[np.ones((100, 1)), X]

# 用于绘图的x范围
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]

# BGD参数
eta = 0.1  # 学习率
n_iterations = 50  # 迭代次数
m = len(X_b)
theta = np.random.randn(2, 1)  # 随机初始化

# 存储损失历史
loss_history = []

# 训练
for iteration in range(n_iterations):
    # 计算预测值
    y_pred = X_b.dot(theta)

    # 计算损失（MSE）
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    # 计算梯度
    gradients = 2 / m * X_b.T.dot(y_pred - y)

    # 更新参数
    theta = theta - eta * gradients

print(f"真实值: 截距=4, 斜率=3")
print(f"BGD结果: 截距={theta[0][0]:.2f}, 斜率={theta[1][0]:.2f}")

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(X, y, 'b.')
plt.plot(X_new, X_new_b.dot(theta), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('BGD拟合结果')

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('MSE损失')
plt.title('损失下降曲线')

plt.tight_layout()
plt.show()