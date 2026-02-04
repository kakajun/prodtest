# -*- coding: utf-8 -*-
"""使用 Sampling (采样) 进行异常检测的示例
主要特点
基于随机采样的异常检测方法：Sampling算法通过随机选择数据子集来估计数据点的异常程度，不依赖于全局数据分布假设

参数简单：模型不需要复杂的参数调整，默认配置通常就能取得不错的效果

计算效率高：由于采用采样策略，避免了对整个数据集的复杂计算，处理速度快

适用于中小规模数据集：在数据量不是特别大的情况下表现良好

技术特性
无监督学习：作为无监督异常检测算法，不需要标注的训练数据

基于局部密度：通过计算数据点相对于随机采样子集的稀有性来判断异常程度

内存友好：由于采用采样方法，内存占用相对较小

对数据分布无特殊假设：不假设数据遵循特定的概率分布

应用场景
异常检测入门工具：由于其简单性和易用性，适合作为异常检测的基线算法

快速原型开发：在需要快速验证异常检测概念时非常有用

中小型数据集：对于特征维度不太高的数据集效果较好

性能基准：可作为评估其他更复杂算法性能的基准参考
"""
# Author: Akira Tamamori <tamamori5917@gmail.com>
# License: BSD 2 clause

from __future__ import division, print_function

import os
import sys

from pyod.models.sampling import Sampling
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

    # 生成样本数据
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # 训练 Sampling 检测器
    clf_name = "Sampling"
    clf = Sampling()
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
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
