# -*- coding: utf-8 -*-
"""
ABOD 真实场景使用示例 (带可视化)
该脚本演示了如何在真实场景中使用 ABOD (Angle-Based Outlier Detection) 算法，
并包含数据降维可视化的步骤。

真实场景通常包括以下步骤：
1. 数据加载：从文件（如 CSV）加载数据
2. 数据预处理：处理缺失值、数据标准化、划分训练/测试集
3. 模型训练：使用 ABOD 训练模型
4. 结果可视化：使用 PCA 降维到 2D 并展示
5. 结果保存：将预测结果保存为文件
"""
from __future__ import division


import os
import sys
import shutil

# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pyod.models.abod import ABOD
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print

# 临时添加 pyod 路径 (如果已安装 pyod 则不需要)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


def create_dummy_data(file_path):
    """创建一个模拟的真实数据 CSV 文件用于演示"""
    print(f"正在创建模拟数据文件: {file_path} ...")

    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 创建一些正常数据 (高斯分布)
    n_inliers = 500
    inliers = np.random.normal(loc=0, scale=1, size=(n_inliers, 5))
    inliers_labels = np.zeros(n_inliers)  # 0 表示正常

    # 创建一些异常数据 (均匀分布，远离中心)
    n_outliers = 50
    outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 5))
    outliers_labels = np.ones(n_outliers)  # 1 表示异常

    # 合并数据
    X = np.vstack([inliers, outliers])
    y = np.concatenate([inliers_labels, outliers_labels])

    # 创建 DataFrame
    # 真实场景中，数据通常包含 ID 列、特征列和标签列（如果有的话）
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
    df.insert(0, 'sample_id', [f'ID_{i}' for i in range(len(df))])
    df['label'] = y  # 添加真实标签用于验证

    # 保存为 CSV
    df.to_csv(file_path, index=False)
    print("模拟数据创建完成。\n")


def main():
    # 1. 设置文件路径
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建 'abod' 目录 (与脚本同级)
    output_dir = os.path.join(current_dir, 'abod')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    data_file = os.path.join(output_dir, 'real_world_data.csv')
    result_file = os.path.join(output_dir, 'abod_prediction_results.csv')

    # 如果数据文件不存在，则创建一个模拟文件
    if not os.path.exists(data_file):
        create_dummy_data(data_file)

    # 2. 数据加载
    print(f"正在读取数据: {data_file} ...")
    df = pd.read_csv(data_file)
    print(f"数据形状: {df.shape}")

    # 3. 数据预处理
    # 提取特征和标签
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values

    # 划分训练集和测试集 (70% 训练, 30% 测试)
    print("\n划分训练集和测试集 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 数据标准化
    print("正在进行数据标准化 ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 模型训练
    # method='fast' 推荐用于真实的大型数据集
    print("\n正在初始化并训练 ABOD 模型 (method='fast') ...")
    clf_name = 'ABOD'
    clf = ABOD(contamination=0.1, method='fast', n_neighbors=10)
    clf.fit(X_train_scaled)

    # 5. 获取预测结果
    # 训练集预测
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    # 测试集预测
    y_test_pred = clf.predict(X_test_scaled)
    y_test_scores = clf.decision_function(X_test_scaled)

    print("\n评估结果:")
    print("在训练数据上:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("在测试数据上:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # 6. 结果可视化
    # visualize() 函数要求输入数据为 2D
    # 因此我们需要使用 PCA 将 5D 数据降维到 2D
    print("\n正在进行 PCA 降维以进行可视化 ...")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("正在生成可视化图表 ...")
    # 注意：这里传入的是降维后的数据
    visualize(clf_name, X_train_pca, y_train, X_test_pca, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=True)

    # 移动生成的图片到目标目录
    source_img = f'{clf_name}.png'
    dest_img = os.path.join(output_dir, f'{clf_name}.png')
    if os.path.exists(source_img):
        # 如果目标文件已存在，先删除
        if os.path.exists(dest_img):
            os.remove(dest_img)
        shutil.move(source_img, dest_img)
        print(f"可视化图表已保存至: {dest_img}")
    else:
        print("可视化图表已保存。")

    # 7. 保存结果
    # 将测试集的预测结果保存
    # 重新构建测试集的 DataFrame
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['label'] = y_test
    test_df['is_outlier_pred'] = y_test_pred
    test_df['anomaly_score'] = y_test_scores

    # 保存到 CSV
    test_df.to_csv(result_file, index=False)
    print(f"\n测试集预测结果已保存至: {result_file}")


if __name__ == "__main__":
    main()
