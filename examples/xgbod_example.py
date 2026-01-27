# -*- coding: utf-8 -*-
"""使用 XGBOD 进行异常检测的示例
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.xgbod import XGBOD
from scipy.io import loadmat
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split


import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    # 定义数据文件并读取 X 和 y
    # 如果源数据缺失，则生成一些数据
    mat_file = 'cardio.mat'
    try:
        mat = loadmat(os.path.join('data', mat_file))

    except TypeError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    except IOError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    else:
        X = mat['X']
        y = mat['y'].ravel()
        X, y = check_X_y(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=42)

    # 训练 XGBOD 检测器
    clf_name = 'XGBOD'
    clf = XGBOD(random_state=42)
    clf.fit(X_train, y_train)

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
