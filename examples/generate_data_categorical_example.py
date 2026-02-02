# -*- coding: utf-8 -*-
"""使用和可视化 ``generate_data_categorical`` 函数的示例。
这是一个辅助工具脚本，用于帮助用户测试那些专门处理 分类数据 的异常检测算法（虽然 PyOD 中的大多数模型是针对数值数据的，
但处理分类数据时通常需要先进行 One-Hot 编码或 Embedding，这个生成器可以直接提供测试素材）
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from pyod.utils.data import generate_data_categorical


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比

    # 生成分类样本数据
    X_train, X_test, y_train, y_test = generate_data_categorical(n_train=200, n_test=50,
                                                                 n_category_in=8, n_category_out=5,
                                                                 n_informative=1, n_features=1,
                                                                 contamination=contamination,
                                                                 shuffle=True, random_state=42)

    # 注意，可视化只能是 1 维的！
    cats = list(np.ravel(X_train))
    labels = list(y_train)
    fig, axs = plt.subplots(1, 2)
    axs[0].bar(cats, labels)
    axs[1].plot(cats, labels)
    plt.title('Synthetic Categorical Train Data')
    plt.show()

    cats = list(np.ravel(X_test))
    labels = list(y_test)
    fig, axs = plt.subplots(1, 2)
    axs[0].bar(cats, labels)
    axs[1].plot(cats, labels)
    plt.title('Synthetic Categorical Test Data')
    plt.show()
