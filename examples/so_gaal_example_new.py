# -*- coding: utf-8 -*-
"""使用单目标生成对抗主动学习 (SO_GAAL) 进行异常检测的示例

模型特点
基于生成对抗网络(GAN)：SO_GAAL 使用生成器和判别器的对抗学习机制来检测异常点

单目标优化：与传统的多目标GAAL不同，SO_GAAL采用单目标优化策略，简化了训练过程

无监督学习：完全无监督的方法，不需要标记的异常数据进行训练

自适应阈值：能够自动确定异常检测的阈值，无需手动设定

适用于高维数据：从示例代码看，该模型能够处理高维特征空间（示例中为300维）

技术特性
生成器与判别器架构：生成器尝试生成类似正常数据的样本，判别器区分真实数据和生成数据

异常分数计算：通过生成器的重构误差或判别器的输出来计算异常分数

迭代训练：通过多轮训练(epoch_num=6)逐步优化模型性能

内存消耗较高：由于涉及神经网络训练，对计算资源有一定要求

应用场景
高维数据异常检测：适用于图像、文本、生物信息等高维数据的异常检测

工业质量控制：在制造过程中检测异常产品

网络安全：检测网络流量中的异常行为或入侵

金融欺诈检测：识别异常交易模式

医疗诊断：检测异常的医疗指标或影像数据

大规模数据集：从示例代码看出，该模型能够处理大规模数据（训练样本30000个）
"""
# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause


from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
from pyod.models.so_gaal_new import SO_GAAL
import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


if __name__ == "__main__":
    contamination = 0.1  # 异常值的百分比
    n_train = 30000  # 训练样本数
    n_test = 3000  # 测试样本数
    n_features = 300  # 特征数

    # 生成样本数据
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=n_features,
                      contamination=contamination,
                      random_state=42)

    # 训练 SO_GAAL 检测器
    clf_name = 'SO_GAAL'
    clf = SO_GAAL(epoch_num=6, contamination=contamination, verbose=2)
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
