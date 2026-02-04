# -*- coding: utf-8 -*-
"""使用 ROD 进行异常检测的示例
模型特点
基于残差的异常检测：ROD通过计算数据点在子空间上的投影残差来识别异常值，对异常值具有较强的鲁棒性

三维特征处理能力：ROD专门设计用于处理低维（特别是三维及以下）数据，通过计算点到直线或平面的距离来检测异常

参数简单：相比其他异常检测算法，ROD只需要很少的参数，易于使用和调优

计算效率高：由于算法设计简洁，在处理小到中等规模数据集时速度较快

不受距离度量限制：不像基于距离的方法（如KNN）容易受到维度诅咒的影响

应用场景
低维数据分析：特别适用于二维和三维数据的异常检测，如散点图中的离群点检测

工业质量控制：在制造业中用于检测产品质量参数的异常值

金融欺诈检测：用于检测交易行为中的异常模式

生物医学数据：处理基因表达数据、生理指标等低维特征数据

传感器异常检测：物联网环境中监测传感器读数的异常情况
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.rod import ROD


import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
import time

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

    # 训练 ROD 检测器
    clf_name = 'ROD'
    clf = ROD()
    clf.fit(X_train)

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
