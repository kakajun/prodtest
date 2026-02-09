# -*- coding: utf-8 -*-
"""
使用具有马哈拉诺比斯距离的 kNN 进行风机异常检测的示例
Demo for Wind Turbine Anomaly Detection using KNN with Mahalanobis Distance
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print

# 临时添加父目录以导入 pyod (如果未安装)
# 同时也为了导入上级目录的 data_generator
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from data_generator import generate_wind_turbine_data
except ImportError:
    # Fallback if running from different location
    sys.path.append(os.path.join(os.getcwd(), 'examples/fengji'))
    from data_generator import generate_wind_turbine_data

if __name__ == "__main__":
    # 配置参数
    contamination = 0.1  # 异常比例
    n_train = 400
    n_test = 100

    # 1. 生成数据
    print("正在生成风机模拟数据...")
    # 为了保持与原代码兼容，我们只使用前两列 (风速, 功率)
    # return_dataframe=False 返回 numpy array
    X, y = generate_wind_turbine_data(n_samples=n_train + n_test,
                                      contamination=contamination,
                                      return_dataframe=False)

    # 仅保留前两列: [风速, 功率]
    # 原有的 KNN 示例是针对 2D 数据设计的，直接使用所有特征可能需要调整协方差计算和可视化逻辑
    X = X[:, :2]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]


    # 2. 训练模型 (KNN with Mahalanobis Distance)
    print("正在训练 KNN (Mahalanobis Distance) 模型...")
    clf_name = 'KNN (mahalanobis)'

    # 计算训练数据的协方差矩阵 (用于马氏距离)
    # rowvar=False 表示每一列是一个变量(特征)
    X_train_cov = np.cov(X_train, rowvar=False)

    # 初始化 KNN 模型
    # metric='mahalanobis': 使用马氏距离
    # metric_params={'V': X_train_cov}: 传入协方差矩阵 (V 是 scipy.spatial.distance.mahalanobis 的参数名)
    clf = KNN(algorithm='auto', metric='mahalanobis',
              metric_params={'V': X_train_cov}, contamination=contamination)

    clf.fit(X_train)

    # 3. 预测与评估
    # 获取训练集预测结果
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    # 获取测试集预测结果
    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    # 打印评估结果
    print("\n评估结果 (Evaluation Results):")
    print("-" * 30)
    print("在训练集上:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\n在测试集上:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # 4. 可视化结果 (保存图片)
    try:
        plt.figure(figsize=(12, 10))

        # 绘制训练数据
        plt.subplot(2, 2, 1)
        plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', s=20, label='Normal')
        plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', s=20, marker='x', label='Outlier')
        plt.title('Training Data (Ground Truth)')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power (kW)')
        plt.legend()

        # 绘制测试数据真实标签
        plt.subplot(2, 2, 2)
        plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', s=20, label='Normal')
        plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', s=20, marker='x', label='Outlier')
        plt.title('Test Data (Ground Truth)')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power (kW)')
        plt.legend()

        # 绘制测试数据预测结果
        plt.subplot(2, 2, 3)
        plt.scatter(X_test[y_test_pred==0, 0], X_test[y_test_pred==0, 1], c='blue', s=20, label='Predicted Normal')
        plt.scatter(X_test[y_test_pred==1, 0], X_test[y_test_pred==1, 1], c='red', s=20, marker='x', label='Predicted Outlier')
        plt.title('Test Data (Prediction)')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power (kW)')
        plt.legend()

        # 绘制异常分数
        plt.subplot(2, 2, 4)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_scores, cmap='viridis', s=20)
        plt.colorbar(label='Anomaly Score')
        plt.title('Test Data (Anomaly Scores)')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Power (kW)')

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'wind_turbine_anomaly_result.png')
        plt.savefig(save_path)
        print(f"\n结果图已保存至: {save_path}")

    except Exception as e:
        print(f"\n可视化失败: {e}")
