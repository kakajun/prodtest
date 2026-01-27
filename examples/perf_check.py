# -*- coding: utf-8 -*-
"""组合多个基础异常分数的示例。演示了四种组合框架：
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from scipy.io import loadmat


import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":

    file_list = ['arrhythmia.mat', 'cardio.mat', 'ionosphere.mat',
                 'letter.mat', 'pima.mat']
    # 定义数据文件并读取 X 和 y
    # 如果源数据缺失，则生成一些数据

    for mat_file in file_list:

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

        clf = KNN()  # 您要检查的算法
        # clf = KNN_new()
        clf.fit(X)  # 拟合模型

        # 打印性能
        evaluate_print(mat_file, y, clf.decision_scores_)
