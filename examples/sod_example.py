# -*- coding: utf-8 -*-
"""使用 SOD 进行异常检测的示例
模型特点
子空间异常检测：SOD专门设计用于高维数据的异常检测，通过在局部邻域的子空间中计算异常分数来识别异常点

基于k-最近邻的局部方法：算法首先找到每个数据点的k个最近邻，然后在这些邻域内计算异常分数

考虑局部密度变化：不同于全局方法，SOD考虑了数据在不同子空间中的局部密度差异

适合高维数据：算法在高维空间中表现更好，代码注释中提到"d > 2"，即更适合高维数据

参数较少：相比于其他子空间方法，SOD实现较为简洁，易于使用

技术特性
局部邻域分析：通过k近邻定义局部区域，在这些区域内计算子空间异常分数

子空间权重计算：为不同维度分配不同的权重，突出在某些子空间中表现异常的维度

维度敏感性：算法性能随维度增加而提升，在低维空间中可能不如其他方法有效

计算复杂度：需要计算k近邻，时间复杂度与数据规模和维度相关

应用场景
高维数据分析：适用于基因数据、文本数据、图像特征等高维数据的异常检测

生物信息学：基因表达数据分析中检测异常基因表达模式

文本挖掘：文档向量化后检测异常文档或主题

推荐系统：检测用户行为中的异常模式

传感器网络：在多维传感器数据中检测异常节点或异常事件

金融风控：多维财务指标中检测异常账户或交易行为
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.sod import SOD


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

    # 训练 SOD 检测器
    # 注意 SOD 旨在处理高维数据 d > 2。
    # 但这里我们为了可视化目的使用 2D
    # 因此，在更高维度中预期会有更高的精度
    clf_name = 'SOD'
    clf = SOD()
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
