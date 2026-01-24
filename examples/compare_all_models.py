# -*- coding: utf-8 -*-
"""通过绘制决策边界和决策边界的数量来比较所有检测算法。
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from pyod.models.lunar import LUNAR
from pyod.models.kpca import KPCA
from pyod.models.sampling import Sampling
from pyod.models.qmcd import QMCD
from pyod.models.suod import SUOD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.dif import DIF
from pyod.models.lmdd import LMDD
from pyod.models.kde import KDE
from pyod.models.gmm import GMM
from pyod.models.inne import INNE
from pyod.models.lscp import LSCP
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
import matplotlib.font_manager
import matplotlib.pyplot as plt
from numpy import percentile
import numpy as np
import warnings
from __future__ import print_function

import os
import sys

# 如果未安装 pyod，这是相对导入的临时解决方案
# 如果已安装 pyod，则无需使用以下行
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# 抑制警告以获得清晰的输出

warnings.filterwarnings("ignore")

# 导入所有模型


# TODO: add neural networks, LOCI, SOS, COF, SOD

# 定义正常值和异常值的数量
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0]

# 在给定设置下比较给定的检测器
# 初始化数据
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# 初始化 LSCP 的检测器集合
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

# 显示数据的统计信息
print('Number of inliers: %i' % n_inliers)
print('Number of outliers: %i' % n_outliers)
print(
    'Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format(
        shape=ground_truth.shape))
print(ground_truth, '\n')

random_state = 42
# 定义九个要比较的异常检测工具
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
    ABOD(contamination=outliers_fraction),
        'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction),
        'Median KNN': KNN(method='median',
                          contamination=outliers_fraction),
        'Local Outlier Factor (LOF)':
    LOF(n_neighbors=35, contamination=outliers_fraction),

        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction,
                                           random_state=random_state),
        'INNE': INNE(
        max_samples=2, contamination=outliers_fraction,
        random_state=random_state,
    ),

    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state),
        'Feature Bagging':
    FeatureBagging(LOF(n_neighbors=35),
                   contamination=outliers_fraction,
                   random_state=random_state),
        'SUOD': SUOD(contamination=outliers_fraction),

        'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),

        'Principal Component Analysis (PCA)': PCA(
        contamination=outliers_fraction, random_state=random_state),
        'KPCA': KPCA(
        contamination=outliers_fraction),

        'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
                                                    random_state=random_state),

        'LMDD': LMDD(contamination=outliers_fraction,
                     random_state=random_state),

        'Histogram-based Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),

        'Copula-base Outlier Detection (COPOD)': COPOD(
        contamination=outliers_fraction),

        'ECDF-baseD Outlier Detection (ECOD)': ECOD(
        contamination=outliers_fraction),
        'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

        'QMCD': QMCD(
        contamination=outliers_fraction),

        'Sampling': Sampling(
        contamination=outliers_fraction),

        'LUNAR': LUNAR(),

        'Cluster-based Local Outlier Factor (CBLOF)':
    CBLOF(contamination=outliers_fraction,
          check_estimator=False, random_state=random_state),

        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
}

# 显示所有检测器
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

# 使用生成的数据拟合模型并比较模型性能
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # 数据生成
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # 添加异常值
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # 拟合模型
    plt.figure(figsize=(20, 22))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print()
        print(i + 1, 'fitting', clf_name)
        # 拟合数据并标记异常值
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
        threshold = percentile(scores_pred, 100 * outliers_fraction)
        n_errors = (y_pred != ground_truth).sum()
        # 绘制等高线和点

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(5, 5, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        # a = subplot.contour(xx, yy, Z, levels=[threshold],
        #                     linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [
                # a.collections[0],
                b, c],
            [
                # 'learned decision function',
                'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    plt.suptitle("25 outlier detection algorithms on synthetic data",
                 fontsize=35)
plt.savefig('ALL.png', dpi=300, bbox_inches='tight')
plt.show()
