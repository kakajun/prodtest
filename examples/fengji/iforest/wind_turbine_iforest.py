# -*- coding: utf-8 -*-
"""
使用 Isolation Forest 进行风机异常检测的示例
Demo for Wind Turbine Anomaly Detection using Isolation Forest
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print

# 添加父目录以导入 data_generator 和 pyod
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # examples/fengji
sys.path.append(parent_dir)
sys.path.append(os.path.abspath(os.path.join(parent_dir, '../..'))) # prodtest root for pyod

try:
    from data_generator import generate_wind_turbine_data, get_feature_names
except ImportError:
    # Fallback if running from different location
    sys.path.append(os.path.join(os.getcwd(), 'examples/fengji'))
    from data_generator import generate_wind_turbine_data, get_feature_names

if __name__ == "__main__":
    # 配置参数
    contamination = 0.1  # 异常比例
    n_train = 500
    n_test = 200
    
    # 1. 生成数据 (使用公共模块)
    print("正在生成风机模拟数据 (包含风速、功率、转速、温度)...")
    # 返回 DataFrame，列名为 b, dc, ah, a
    X_df, y = generate_wind_turbine_data(n_samples=n_train + n_test, 
                                         contamination=contamination, 
                                         return_dataframe=True)
    
    # 获取特征映射名称
    feature_map = get_feature_names()
    print(f"特征列表: {X_df.columns.tolist()}")
    print(f"特征含义: {feature_map}")
    
    # 分割训练集和测试集
    X_train = X_df.iloc[:n_train]
    X_test = X_df.iloc[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    # 2. 训练模型 (Isolation Forest)
    print("\n正在训练 Isolation Forest 模型...")
    clf_name = 'IForest'
    # IForest 不需要计算协方差矩阵，适合高维数据，处理非线性关系能力强
    clf = IForest(contamination=contamination, random_state=42, n_estimators=100)
    
    # 使用所有 4 个特征进行训练
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
    
    # 4. 可视化结果
    # 虽然使用了4个特征训练，但为了直观展示，我们主要绘制 风速(b) vs 功率(dc)
    try:
        plt.figure(figsize=(14, 6))
        
        # 提取绘图所需的列
        x_col = 'b'   # 风速
        y_col = 'dc'  # 功率
        
        # 子图1: 真实标签 (Ground Truth)
        plt.subplot(1, 2, 1)
        # 正常点
        plt.scatter(X_test.loc[y_test==0, x_col], X_test.loc[y_test==0, y_col], 
                    c='blue', s=20, label='Normal', alpha=0.7)
        # 异常点
        plt.scatter(X_test.loc[y_test==1, x_col], X_test.loc[y_test==1, y_col], 
                    c='red', s=20, marker='x', label='Outlier', alpha=0.9)
        
        plt.title(f'Test Data Ground Truth\n({feature_map[x_col]} vs {feature_map[y_col]})')
        plt.xlabel(feature_map[x_col])
        plt.ylabel(feature_map[y_col])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 预测结果 (Prediction)
        plt.subplot(1, 2, 2)
        # 预测正常
        plt.scatter(X_test.loc[y_test_pred==0, x_col], X_test.loc[y_test_pred==0, y_col], 
                    c='blue', s=20, label='Predicted Normal', alpha=0.7)
        # 预测异常
        plt.scatter(X_test.loc[y_test_pred==1, x_col], X_test.loc[y_test_pred==1, y_col], 
                    c='orange', s=20, marker='x', label='Predicted Outlier', alpha=0.9)
        
        # 圈出被 IForest 认为异常的点 (用圆圈标记)
        # 可以在背景中画出异常分数热力图，但对于高维数据比较难画，这里仅展示散点
        
        plt.title(f'IForest Prediction Result\n({feature_map[x_col]} vs {feature_map[y_col]})')
        plt.xlabel(feature_map[x_col])
        plt.ylabel(feature_map[y_col])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(current_dir, 'wind_turbine_iforest_result.png')
        plt.savefig(save_path)
        print(f"\n结果图已保存至: {save_path}")
        
    except Exception as e:
        print(f"\n可视化失败: {e}")
        import traceback
        traceback.print_exc()
