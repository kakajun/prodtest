"""使用 Quasi-Monte Carlo Discrepancy (QMCD) 进行异常检测的示例
"""
# Author: D Kulik
# License: BSD 2 clause

from __future__ import division
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.qmcd import QMCD
import numpy as np
from __future__ import print_function

import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比
    n_train = 200  # 训练样本数
    n_test = 100  # 测试样本数

    # 生成样本数据
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # 训练 QMCD 检测器
    clf_name = 'QMCD'
    clf = QMCD()
    clf.fit(X_train, y_train)

    # 获取训练数据的预测标签和异常分数
    y_train_pred = clf.labels_  # 二进制标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 原始异常分数

    # 获取测试数据的预测结果
    y_test_pred = clf.predict(
        np.append(X_test, y_test.reshape(-1, 1), axis=1))  # 异常标签 (0 或 1)
    y_test_scores = clf.decision_function(
        np.append(X_test, y_test.reshape(-1, 1), axis=1))  # 异常分数

    # 评估并打印结果
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # 可视化结果
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
