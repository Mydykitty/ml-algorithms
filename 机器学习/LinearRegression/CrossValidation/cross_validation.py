import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========== 1. 加载手写体数据集 ==========
print("正在加载MNIST数据集...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

# 转换为数值类型
X = X.astype(np.float64)
y = y.astype(np.int64)

print(f"数据集形状: {X.shape}")
print(f"标签形状: {y.shape}")

# ========== 2. 数据预处理 ==========
# 归一化（像素值0-255 -> 0-1）
X = X / 255.0

# 简化：只使用数字5和非5的二分类问题（与你图片中的 y_train_5 对应）
y_5 = (y == 5).astype(int)  # 是5为1，不是5为0

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_5, y_test_5 = train_test_split(
    X, y_5, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"正例比例: {y_train_5.mean():.2%}")

# ========== 3. 创建分类器 ==========
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)

# ========== 4. 分层K折交叉验证 ==========
print("\n" + "=" * 50)
print("开始分层K折交叉验证...")
print("=" * 50)

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

fold_scores = []
fold_number = 1

for train_index, test_index in skfolds.split(X_train, y_train_5):
    print(f"\n第 {fold_number} 折:")

    # 克隆分类器（确保每折使用独立的模型）
    clone_clf = clone(sgd_clf)

    # 分割数据
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    # 训练
    clone_clf.fit(X_train_folds, y_train_folds)

    # 预测
    y_pred = clone_clf.predict(X_test_folds)

    # 计算准确率
    n_correct = np.sum(y_pred == y_test_folds)
    accuracy = n_correct / len(y_pred)
    fold_scores.append(accuracy)

    print(f"  预测正确数: {n_correct}/{len(y_pred)}")
    print(f"  准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\n" + "=" * 50)
print(f"交叉验证结果:")
print(f"  各折准确率: {[f'{s:.4f}' for s in fold_scores]}")
print(f"  平均准确率: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")

# ========== 5. 最终模型评估 ==========
print("\n" + "=" * 50)
print("在测试集上评估最终模型...")
print("=" * 50)

# 在整个训练集上重新训练
sgd_clf.fit(X_train, y_train_5)
y_test_pred = sgd_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_5, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


# ========== 6. 可视化一些样本 ==========
def plot_sample_images(X, y, indices, predictions=None):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        image = X[idx].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        label = "5" if y[idx] == 1 else "not 5"
        title = f"True: {label}"
        if predictions is not None:
            pred_label = "5" if predictions[idx] == 1 else "not 5"
            title += f"\nPred: {pred_label}"
            color = 'green' if y[idx] == predictions[idx] else 'red'
            axes[i].set_title(title, color=color)
        else:
            axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 显示测试集中的一些样本
sample_indices = np.random.choice(len(X_test), 10, replace=False)
plot_sample_images(X_test, y_test_5, sample_indices, y_test_pred)

# ========== 7. 混淆矩阵 ==========
print("\n混淆矩阵:")
cm = confusion_matrix(y_test_5, y_test_pred)
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Not 5', 'Is 5'])
plt.yticks([0, 1], ['Not 5', 'Is 5'])

# 添加数值标注
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

plt.show()

print("\n分类报告:")
print(classification_report(y_test_5, y_test_pred, target_names=['Not 5', 'Is 5']))

# ========== 8. Precision-Recall 曲线（不同阈值） ==========
print("\n" + "=" * 50)
print("计算不同阈值下的 Precision 和 Recall...")
print("=" * 50)

from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# 获取决策分数（置信度）
y_scores = sgd_clf.decision_function(X_test)

# 计算不同阈值下的精确率和召回率
precisions, recalls, thresholds = precision_recall_curve(y_test_5, y_scores)

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(12, 5))

# 子图1：Precision-Recall 曲线
plt.subplot(1, 2, 1)
plt.plot(recalls, precisions, 'b-', linewidth=2)
plt.xlabel('Recall (召回率)', fontsize=12)
plt.ylabel('Precision (精确率)', fontsize=12)
plt.title('Precision-Recall 曲线', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])

# 标注几个关键点
threshold_points = [0, 0.5, 1, 2]  # 示例阈值
for t in threshold_points:
    # 找到最接近的阈值
    idx = np.argmin(np.abs(thresholds - t)) if t < max(thresholds) else -1
    if idx >= 0 and idx < len(precisions):
        plt.plot(recalls[idx], precisions[idx], 'ro', markersize=8)
        plt.annotate(f'θ={t}',
                     (recalls[idx], precisions[idx]),
                     xytext=(10, 5),
                     textcoords='offset points',
                     fontsize=10)

# 子图2：阈值 vs Precision/Recall
plt.subplot(1, 2, 2)
# 只画到阈值最大值的95%，避免无穷大
plt.plot(thresholds, precisions[:-1], 'g-', linewidth=2, label='Precision')
plt.plot(thresholds, recalls[:-1], 'r-', linewidth=2, label='Recall')
plt.xlabel('Threshold (阈值)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('不同阈值下的 Precision 和 Recall', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='默认阈值(0)')

plt.tight_layout()
plt.show()

# ========== 9. 详细分析不同阈值的影响 ==========
print("\n不同阈值下的性能对比:")
print("-" * 60)
print(f"{'阈值':<10} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
print("-" * 60)

# 测试不同的阈值
test_thresholds = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0]

for threshold in test_thresholds:
    # 根据阈值进行预测
    y_pred_threshold = (y_scores >= threshold).astype(int)

    precision = precision_score(y_test_5, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test_5, y_pred_threshold, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{threshold:<10.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

# ========== 10. 寻找最佳阈值（根据F1分数） ==========
# 计算每个阈值对应的F1分数
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]
best_precision = precisions[best_threshold_idx]
best_recall = recalls[best_threshold_idx]

print("\n" + "=" * 50)
print("最佳阈值分析:")
print("=" * 50)
print(f"最佳阈值 (最大F1分数): {best_threshold:.4f}")
print(f"对应的精确率: {best_precision:.4f}")
print(f"对应的召回率: {best_recall:.4f}")
print(f"对应的F1分数: {best_f1:.4f}")

# 使用最佳阈值重新预测
y_pred_best = (y_scores >= best_threshold).astype(int)
print(f"\n使用最佳阈值的测试集准确率: {accuracy_score(y_test_5, y_pred_best):.4f}")

# ========== 11. 可视化不同阈值的效果对比 ==========
# 选择几个有代表性的样本，展示不同阈值下的预测变化
print("\n" + "=" * 50)
print("展示不同阈值对单个样本的影响:")
print("=" * 50)

# 找几个置信度不同的样本
sample_indices_for_demo = []
scores_list = y_scores.flatten()

# 找一个高置信度的正例
high_conf_pos = np.where((y_test_5 == 1) & (scores_list > 2))[0]
# 找一个低置信度的正例
low_conf_pos = np.where((y_test_5 == 1) & (scores_list > 0) & (scores_list < 1))[0]
# 找一个高置信度的负例
high_conf_neg = np.where((y_test_5 == 0) & (scores_list < -2))[0]
# 找一个低置信度的负例（容易被误判）
low_conf_neg = np.where((y_test_5 == 0) & (scores_list > -0.5) & (scores_list < 0))[0]

demo_indices = []
if len(high_conf_pos) > 0:
    demo_indices.append(high_conf_pos[0])
if len(low_conf_pos) > 0:
    demo_indices.append(low_conf_pos[0])
if len(high_conf_neg) > 0:
    demo_indices.append(high_conf_neg[0])
if len(low_conf_neg) > 0:
    demo_indices.append(low_conf_neg[0])

# 展示这些样本
fig, axes = plt.subplots(1, len(demo_indices), figsize=(15, 4))
if len(demo_indices) == 1:
    axes = [axes]

for idx, sample_idx in enumerate(demo_indices):
    # 显示图片
    axes[idx].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')

    true_label = "5" if y_test_5[sample_idx] == 1 else "not 5"
    score = y_scores[sample_idx]

    # 不同阈值下的预测
    title = f"True: {true_label}\nScore: {score:.2f}\n"
    for threshold in [-1, 0, 1]:
        pred = "5" if score >= threshold else "not 5"
        title += f"θ={threshold}: {pred}  "

    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# ========== 13. ROC 曲线（不同阈值下的 TPR 和 FPR） ==========
print("\n" + "=" * 50)
print("计算 ROC 曲线...")
print("=" * 50)

from sklearn.metrics import roc_curve, roc_auc_score

# 获取决策分数（置信度）
y_scores = sgd_clf.decision_function(X_test)

# 计算 ROC 曲线的 FPR, TPR 和阈值
fpr, tpr, roc_thresholds = roc_curve(y_test_5, y_scores)
roc_auc = roc_auc_score(y_test_5, y_scores)

print(f"ROC AUC 分数: {roc_auc:.4f}")

# ========== 14. 绘制 ROC 曲线 ==========
plt.figure(figsize=(15, 10))

# 子图1：ROC 曲线
plt.subplot(2, 2, 1)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机猜测 (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR) - 假阳性率', fontsize=12)
plt.ylabel('True Positive Rate (TPR) - 召回率', fontsize=12)
plt.title('ROC 曲线', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

# 标注几个关键阈值点
threshold_points = [-2, -1, 0, 1, 2]
for t in threshold_points:
    # 找到最接近的阈值
    idx = np.argmin(np.abs(roc_thresholds - t)) if t < max(roc_thresholds) else -1
    if idx >= 0 and idx < len(fpr):
        plt.plot(fpr[idx], tpr[idx], 'go', markersize=8)
        plt.annotate(f'θ={t}',
                     (fpr[idx], tpr[idx]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9)

# 子图2：阈值 vs TPR/FPR
plt.subplot(2, 2, 2)
# 只画有限范围内的阈值
valid_idx = roc_thresholds > -3  # 限制范围避免无穷
plt.plot(roc_thresholds[valid_idx], tpr[valid_idx], 'g-', linewidth=2, label='TPR (召回率)')
plt.plot(roc_thresholds[valid_idx], fpr[valid_idx], 'r-', linewidth=2, label='FPR (假阳性率)')
plt.xlabel('Threshold (阈值)', fontsize=12)
plt.ylabel('Rate', fontsize=12)
plt.title('不同阈值下的 TPR 和 FPR', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='默认阈值(0)')
plt.gca().invert_xaxis()  # 阈值从大到小

# 子图3：PR曲线 vs ROC曲线（对比）
plt.subplot(2, 2, 3)
# PR曲线
from sklearn.metrics import precision_recall_curve

precisions, recalls, pr_thresholds = precision_recall_curve(y_test_5, y_scores)
plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR 曲线')
plt.xlabel('Recall (召回率)', fontsize=12)
plt.ylabel('Precision (精确率)', fontsize=12)
plt.title('PR 曲线', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.legend()

# 子图4：PR曲线 vs ROC曲线（在同一图上对比思路）
plt.subplot(2, 2, 4)
# 这个图展示两种曲线关注的不同指标
text_content = """
ROC 曲线关注:
- TPR (True Positive Rate) = 召回率
- FPR (False Positive Rate) = 假阳性率

PR 曲线关注:
- Precision (精确率)
- Recall (召回率)

适用场景:
- ROC: 类别平衡时
- PR: 类别不平衡时 (如本例: 9%正例)
"""
plt.text(0.1, 0.5, text_content, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')
plt.title('ROC vs PR 曲线对比', fontsize=14)

plt.tight_layout()
plt.show()

# ========== 15. 不同阈值下的详细性能对比 ==========
print("\n不同阈值下的性能对比:")
print("-" * 90)
print(f"{'阈值':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'TPR':<10} {'FPR':<10}")
print("-" * 90)

# 测试不同的阈值
test_thresholds = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0]

for threshold in test_thresholds:
    # 根据阈值进行预测
    y_pred_threshold = (y_scores >= threshold).astype(int)

    precision = precision_score(y_test_5, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test_5, y_pred_threshold, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 计算 TPR 和 FPR
    tp = np.sum((y_test_5 == 1) & (y_pred_threshold == 1))
    fn = np.sum((y_test_5 == 1) & (y_pred_threshold == 0))
    fp = np.sum((y_test_5 == 0) & (y_pred_threshold == 1))
    tn = np.sum((y_test_5 == 0) & (y_pred_threshold == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"{threshold:<10.1f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {tpr:<10.4f} {fpr:<10.4f}")

# ========== 16. 寻找最佳阈值（根据不同的标准） ==========
print("\n" + "=" * 50)
print("最佳阈值分析（根据不同标准）:")
print("=" * 50)

# 方法1：最大 F1 分数（PR曲线）
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = pr_thresholds[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

print(f"\n1. 最大 F1 分数:")
print(f"   阈值 = {best_f1_threshold:.4f}")
print(f"   F1 分数 = {best_f1:.4f}")
print(f"   精确率 = {precisions[best_f1_idx]:.4f}")
print(f"   召回率 = {recalls[best_f1_idx]:.4f}")

# 方法2：约登指数（ROC曲线）- 最大化 TPR - FPR
youden_idx = np.argmax(tpr - fpr)
best_youden_threshold = roc_thresholds[youden_idx]
best_youden = tpr[youden_idx] - fpr[youden_idx]

print(f"\n2. 最大约登指数 (TPR - FPR):")
print(f"   阈值 = {best_youden_threshold:.4f}")
print(f"   约登指数 = {best_youden:.4f}")
print(f"   TPR (召回率) = {tpr[youden_idx]:.4f}")
print(f"   FPR = {fpr[youden_idx]:.4f}")

# 方法3：最接近左上角 (0,1) 的点
distance = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
best_dist_idx = np.argmin(distance)
best_dist_threshold = roc_thresholds[best_dist_idx]

print(f"\n3. 最接近 ROC 曲线左上角:")
print(f"   阈值 = {best_dist_threshold:.4f}")
print(f"   TPR (召回率) = {tpr[best_dist_idx]:.4f}")
print(f"   FPR = {fpr[best_dist_idx]:.4f}")

# ========== 17. 可视化不同阈值的影响（ROC视角） ==========
print("\n" + "=" * 50)
print("展示不同阈值对 ROC 的影响:")
print("=" * 50)

# 选择几个代表性的阈值
demo_thresholds = [-1.0, 0, 0.5, 1.0, 2.0]
colors = ['purple', 'blue', 'green', 'orange', 'red']

plt.figure(figsize=(12, 5))

# 子图1：ROC曲线上的点
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC 曲线')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机猜测')

for threshold, color in zip(demo_thresholds, colors):
    idx = np.argmin(np.abs(roc_thresholds - threshold))
    plt.plot(fpr[idx], tpr[idx], 'o', color=color, markersize=10)
    plt.annotate(f'θ={threshold}',
                 (fpr[idx], tpr[idx]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=10,
                 color=color)

plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('不同阈值在 ROC 曲线上的位置', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：混淆矩阵随阈值变化
plt.subplot(1, 2, 2)

# 准备数据
thresholds_display = []
tpr_values = []
fpr_values = []
precision_values = []
recall_values = []

for threshold in demo_thresholds:
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_5, y_pred).ravel()

    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_val = tpr_val

    thresholds_display.append(threshold)
    tpr_values.append(tpr_val)
    fpr_values.append(fpr_val)
    precision_values.append(prec_val)
    recall_values.append(rec_val)

x = np.arange(len(demo_thresholds))
width = 0.2

plt.bar(x - width * 1.5, tpr_values, width, label='TPR (召回率)', color='green')
plt.bar(x - width / 2, fpr_values, width, label='FPR', color='red')
plt.bar(x + width / 2, precision_values, width, label='精确率', color='blue')
plt.bar(x + width * 1.5, recall_values, width, label='召回率', color='orange')

plt.xlabel('阈值', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('不同阈值下的性能指标对比', fontsize=14)
plt.xticks(x, demo_thresholds)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ========== 18. 总结输出 ==========
print("\n" + "=" * 50)
print("ROC 曲线总结:")
print("=" * 50)
print(f"""
ROC 曲线的关键信息:
- AUC (Area Under Curve): {roc_auc:.4f}
  * AUC = 1.0: 完美分类器
  * AUC = 0.5: 随机猜测
  * AUC < 0.5: 比随机还差（可能标签反了）

- 曲线越靠近左上角，模型性能越好

ROC 曲线 vs PR 曲线的选择:
- 当前数据正例比例: {y_test_5.mean():.2%}
- 推荐使用: {'PR 曲线' if y_test_5.mean() < 0.3 else 'ROC 曲线'}
- 原因: {'数据不平衡时 PR 曲线更能反映模型性能' if y_test_5.mean() < 0.3 else '数据相对平衡'}

根据业务需求选择阈值:
- 高召回率 (找出更多正例): 降低阈值 (如 {demo_thresholds[0]})
- 高精确率 (减少误报): 提高阈值 (如 {demo_thresholds[-1]})
- 平衡方案: 使用 F1 分数最佳阈值 {best_f1_threshold:.2f}
""")