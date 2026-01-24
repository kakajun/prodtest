# -*- coding: utf-8 -*-
"""使用中位数绝对偏差 (MAD) 进行异常检测的示例。
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.mad import MAD
from __future__ import print_function

import os
import sys
import numpy as np

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

    # 训练 MAD 检测器
    clf_name = 'MAD'
    clf = MAD(threshold=3.5)
    clf.fit(X_train)

    # 获取训练数据的预测标签和异常分数
    y_train_pred = clf.labels_  # 二进制标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 原始异常分数

    # 获取测试数据的预测结果
    y_test_pred = clf.predict(X_test)  # 异常标签 (0 或 1)
    y_test_scores = clf.decision_function(X_test)  # 异常分数

    # 评估并打印结果
    print("\n在训练数据上:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\n在测试数据上:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # 可视化结果
    # 仅为了可视化目的将维度设为 2。通过在每个维度上重复相同的数据。
    visualize(clf_name, np.hstack((X_train, X_train)), y_train, np.hstack((X_test, X_test)), y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
