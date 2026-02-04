# -*- coding: utf-8 -*-
"""R-graph 的示例
基于图的异常检测方法：RGraph 是一种基于重构图（Reconstruction Graph）的异常检测算法，通过构建数据点之间的重构关系来识别异常点

稀疏重构机制：利用稀疏表示理论，对每个数据点用其他数据点的线性组合来重构，正常点通常能被很好地重构，而异常点重构效果较差

高维数据适应性：从示例可以看出，该模型能够处理高维特征空间（示例中为70维特征），适合处理复杂的数据集

应用场景
无监督异常检测：适用于没有标签数据的异常检测任务
多维数据处理：特别适合处理具有较多特征的复杂数据集
鲁棒性强：基于重构误差的原理使其对噪声数据有一定的鲁棒性
"""
# Author: Michiel Bongaerts (but not author of the R-graph method)
# License: BSD 2 clause


from __future__ import division
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.rgraph import RGraph


import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":

    contamination = 0.1  # 异常值的百分比
    n_train = 100  # 训练样本数
    n_test = 100  # 测试样本数

    # 生成样本数据
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=70,
        contamination=contamination,
        behaviour="new",
        random_state=42,
    )

    # 训练 R-graph 检测器
    clf_name = 'R-graph'
    clf = RGraph(n_nonzero=100, transition_steps=20, gamma=50, blocksize_test_data=20,
                 tau=1, preprocessing=True, active_support=False, gamma_nz=False,
                 algorithm='lasso_lars', maxiter=100, verbose=1)

    clf.fit(X_train)

    # 获取训练数据的预测标签和异常分数
    y_train_pred = clf.labels_  # 二进制标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 原始异常分数

    # # 获取测试数据的预测结果
    y_test_pred = clf.predict(X_test)  # 异常标签 (0 或 1)
    y_test_scores = clf.decision_function(X_test)  # 异常分数

    # 评估并打印结果
    print("\n在训练数据上:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\n在测试数据上:")
    evaluate_print(clf_name, y_test, y_test_scores)
