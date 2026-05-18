import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# 添加偏置列
X_b = np.c_[np.ones((m, 1)), X]

# 学习率调度函数
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t1 + t)


# Mini-Batch参数
n_epochs = 50
minibatch = 16
theta = np.random.randn(2, 1)
theta_path_mgd = []
t = 0

# 训练
for epoch in range(n_epochs):
    # 打乱数据
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for i in range(0, m, minibatch):
        t += 1
        xi = X_b_shuffled[i:i + minibatch]
        yi = y_shuffled[i:i + minibatch]

        # 计算梯度
        gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)

        # 更新参数
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta.copy())

print(f"真实值: 截距=4, 斜率=3")
print(f"Mini-Batch结果: 截距={theta[0][0]:.2f}, 斜率={theta[1][0]:.2f}")

# 可视化参数收敛过程
theta_path_mgd = np.array(theta_path_mgd)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'b-', alpha=0.5)
plt.plot(theta_path_mgd[0, 0], theta_path_mgd[0, 1], 'go', label='起点')
plt.plot(theta_path_mgd[-1, 0], theta_path_mgd[-1, 1], 'ro', label='终点')
plt.xlabel('θ0 (截距)')
plt.ylabel('θ1 (斜率)')
plt.title('Mini-Batch 参数收敛路径')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X, y, 'b.')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
plt.plot(X_new, X_new_b.dot(theta), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mini-Batch 拟合结果')

plt.tight_layout()
plt.show()