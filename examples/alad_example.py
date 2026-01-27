# -*- coding: utf-8 -*-
"""使用对抗性学习异常检测 (ALAD) 进行异常检测的示例
对抗性学习异常检测（ALAD）作为基于双向生成对抗网络的异常检测方法，在特征提取、检测效率、适用场景等多个方面均具备显著特点
"""
from __future__ import division

from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.alad import ALAD

import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比
    n_train = 500  # 训练样本数
    n_test = 200  # 测试样本数

    # 生成样本数据
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # 训练 ALAD 检测器
    clf_name = 'ALAD'
    clf = ALAD(epochs=100, latent_dim=2,
               learning_rate_disc=0.0001,
               learning_rate_gen=0.0001,
               dropout_rate=0.2,
               add_recon_loss=False,
               lambda_recon_loss=0.05,
               add_disc_zz_loss=True,
               dec_layers=[75, 100],
               enc_layers=[100, 75],
               disc_xx_layers=[100, 75],
               disc_zz_layers=[25, 25],
               disc_xz_layers=[100, 75],
               spectral_normalization=False,
               activation_hidden_disc='tanh', activation_hidden_gen='tanh',
               preprocessing=True, batch_size=200, contamination=contamination)

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
