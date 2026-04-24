import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置列
X_b = np.c_[np.ones((100, 1)), X]

# SGD参数
m = len(X_b)
n_epochs = 50
t0, t1 = 5, 50
theta = np.random.randn(2, 1)


def learning_schedule(t):
    return t0 / (t1 + t)


# 训练
for epoch in range(n_epochs):
    for i in range(m):
        random_idx = np.random.randint(m)
        xi = X_b[random_idx:random_idx + 1]
        yi = y[random_idx:random_idx + 1]

        data = xi.dot(theta)
        gradients = 2 * xi.T.dot(data - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print(f"真实值: 截距=4, 斜率=3")
print(f"SGD结果: 截距={theta[0][0]:.2f}, 斜率={theta[1][0]:.2f}")

# 预测
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)

# 绘图
plt.plot(X_new, y_predict, 'r-', label='预测')
plt.plot(X, y, 'b.', label='数据')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()