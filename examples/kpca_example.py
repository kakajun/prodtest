# -*- coding: utf-8 -*-
"""基于 Kernel PCA (KPCA) 的异常检测示例。
- 模型机理：KPCA用核函数把数据映射到高维特征空间，再用“重构误差”作为异常分数；核带宽不匹配会导致重构普遍偏差，正常点也被错判为异常。
- 特征尺度：KPCA对特征尺度非常敏感；未标准化时，某些维度主导核距离，造成密度估计失真，错误率升高。
"""
# Author: Akira Tamamori <tamamori5917@gmail.com>
# License: BSD 2 clause

from __future__ import division, print_function

import os
import sys

from pyod.models.kpca import KPCA
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname("__file__"), "..")))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比
    n_train = 200  # 训练样本数
    n_test = 100  # 测试样本数
    n_features = 2

    # 生成样本数据
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=2,
        contamination=contamination,
        random_state=42,
        behaviour="new",
    )

    # 训练 KPCA 检测器
    clf_name = "KPCA"
    clf = KPCA()
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
    visualize(
        clf_name,
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_pred,
        y_test_pred,
        show_figure=True,
    )
