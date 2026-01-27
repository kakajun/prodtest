# -*- coding: utf-8 -*-
"""使用 Copula Based Outlier Detector (COPOD) 进行异常检测的示例
此处提供了样本层面的解释。
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.utility import standardizer
from pyod.models.copod import COPOD
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


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

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)

    # 标准化数据以便处理
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    # 训练 COPOD 检测器
    clf_name = 'COPOD'
    clf = COPOD()

    # 您也可以尝试并行版本。
    # clf = COPOD(n_jobs=2)
    clf.fit(X_train)

    # 获取训练数据的预测标签和异常分数
    y_train_pred = clf.labels_  # 二进制标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 原始异常分数

    print('The first sample is an outlier', y_train[0])
    clf.explain_outlier(0)

    # 我们可以看到特征 7、16 和 20 高于 0.99 的截止值
    # 并且在判定其为异常值方面起着更重要的作用。
