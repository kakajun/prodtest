# 1. 导入所需库
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis  # scipy内置马氏距离函数，不用自己写！

# 2. 加载数据并划分训练集/测试集（避免数据泄露）
iris = load_iris()
X = iris.data  # 特征矩阵：4个特征，150个样本
y = iris.target  # 标签：3个类别（0/1/2）
# 划分：70%训练集，30%测试集，随机种子固定（结果可复现）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 计算训练集的协方差矩阵和逆矩阵（马氏距离核心）
# rowvar=False：表示每行是一个样本，每列是一个特征
cov_mat = np.cov(X_train, rowvar=False)
# 求协方差矩阵的逆（scipy的mahalanobis会自动处理，这里先计算备用）
cov_mat_inv = np.linalg.inv(cov_mat)

# 4. 自定义马氏距离版KNN分类函数（小白重点看注释）
def mahalanobis_knn(X_train, y_train, X_test, k, cov_inv):
    y_pred = []
    # 遍历每个测试样本
    for test_sample in X_test:
        # 存储每个训练样本与测试样本的马氏距离
        distances = []
        for train_sample in X_train:
            # 用scipy内置函数计算马氏距离
            dist = mahalanobis(test_sample, train_sample, cov_inv)
            distances.append(dist)
        # 按距离从小到大排序，取前k个的索引
        k_indices = np.argsort(distances)[:k]
        # 取前k个索引对应的标签
        k_labels = y_train[k_indices]
        # 投票：取出现次数最多的标签（np.bincount统计次数，argmax取最大值索引）
        pred_label = np.bincount(k_labels).argmax()
        y_pred.append(pred_label)
    return np.array(y_pred)

# 5. 设定K值，训练并预测（小白可修改K，比如3/5/9）
k = 5
# 马氏距离KNN预测
y_pred_maha = mahalanobis_knn(X_train, y_train, X_test, k, cov_mat_inv)
# 传统欧氏距离KNN（sklearn内置）对比
knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred_euc = knn_euclidean.predict(X_test)

# 6. 计算准确率，对比效果
acc_maha = accuracy_score(y_test, y_pred_maha)  # 马氏距离KNN准确率
acc_euc = accuracy_score(y_test, y_pred_euc)    # 欧氏距离KNN准确率

# 7. 打印结果
print(f"马氏距离KNN（K={k}）准确率：{acc_maha:.2f}")
print(f"欧氏距离KNN（K={k}）准确率：{acc_euc:.2f}")
print(f"测试集真实标签：{y_test[:10]}")  # 打印前10个真实标签
print(f"马氏距离预测标签：{y_pred_maha[:10]}")  # 打印前10个马氏距离预测标签
print(f"欧氏距离预测标签：{y_pred_euc[:10]}")  # 打印前10个欧氏距离预测标签