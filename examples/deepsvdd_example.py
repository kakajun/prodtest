# -*- coding: utf-8 -*-
"""使用 DeepSVDD 进行异常检测的示例
"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.deep_svdd import DeepSVDD


import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比
    n_train = 20000  # 训练样本数
    n_test = 2000  # 测试样本数
    n_features = 300  # 特征数
    use_ae = False  # 是否使用 ae 架构代替简单神经网络的超参数
    random_state = 10  # 如果 C 设置为 None，则使用 random_state

    # 生成样本数据
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=n_features,
                      contamination=contamination,
                      random_state=42)

    # 训练 DeepSVDD 检测器 (Without-AE)
    clf_name = 'DeepSVDD'
    clf = DeepSVDD(n_features=n_features, use_ae=use_ae, epochs=5, contamination=contamination,
                   random_state=random_state)
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
