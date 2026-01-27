# -*- coding: utf-8 -*-
"""使用基于角度的异常检测 (ABOD) 进行异常检测的示例
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division

from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.abod import ABOD
import pandas as pd

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

    # 训练 ABOD 检测器
    clf_name = 'ABOD'
    clf = ABOD()
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

    # 导出测试数据为 CSV
    # 将 X_test 转换为 DataFrame
    test_df = pd.DataFrame(
        X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    # 添加真实标签
    test_df['ground_truth'] = y_test
    # 添加预测标签
    test_df['prediction_label'] = y_test_pred
    # 添加预测分数
    test_df['prediction_score'] = y_test_scores

    # 保存为 CSV
    test_df.to_csv('abod_test_results.csv', index=False)
    print("\n测试数据及预测结果已导出为 'abod_test_results.csv'")

    # 可视化结果
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
