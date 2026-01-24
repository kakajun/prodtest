# -*- coding: utf-8 -*-
"""组合多个基础异常分数的示例。演示了四种组合框架：

1. Average: 取所有基础检测器的平均值
2. maximization : 取所有检测器中的最大分数作为分数
3. Average of Maximum (AOM): 最大值的平均值
4. Maximum of Average (MOA): 平均值的最大值
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer
from pyod.models.combination import aom, moa, average, maximization, median
from pyod.models.knn import KNN
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from __future__ import print_function

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # 标准化数据以便处理
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    n_clf = 20  # 基础检测器的数量

    # 初始化 20 个基础检测器进行组合
    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
              150, 160, 170, 180, 190, 200]

    train_scores = np.zeros([X_train.shape[0], n_clf])
    test_scores = np.zeros([X_test.shape[0], n_clf])

    print('Combining {n_clf} kNN detectors'.format(n_clf=n_clf))

    for i in range(n_clf):
        k = k_list[i]

        clf = KNN(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)

        train_scores[:, i] = clf.decision_scores_
        test_scores[:, i] = clf.decision_function(X_test_norm)

    # 决策分数必须在组合前进行归一化
    train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                       test_scores)
    # 通过平均值组合
    y_by_average = average(test_scores_norm)
    evaluate_print('Combination by Average', y_test, y_by_average)

    # 通过最大值组合
    y_by_maximization = maximization(test_scores_norm)
    evaluate_print('Combination by Maximization', y_test, y_by_maximization)

    # 通过中位数组合
    y_by_median = median(test_scores_norm)
    evaluate_print('Combination by Median', y_test, y_by_median)

    # 通过 aom 组合
    y_by_aom = aom(test_scores_norm, n_buckets=5)
    evaluate_print('Combination by AOM', y_test, y_by_aom)

    # 通过 moa 组合
    y_by_moa = moa(test_scores_norm, n_buckets=5)
    evaluate_print('Combination by MOA', y_test, y_by_moa)
