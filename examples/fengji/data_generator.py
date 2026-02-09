# -*- coding: utf-8 -*-
"""
风机数据生成器模块
用于生成模拟的风力发电机 SCADA 数据
"""
import numpy as np
import pandas as pd
import json
import os

# 核心字段映射 (参考 column_mapping.json)
# b: 风速
# dc: 发电机有功功率
# ah: 风轮转速
# a: 舱外温度
FEATURE_MAPPING = {
    "b": "风速 (Wind Speed)",
    "dc": "功率 (Power)",
    "ah": "转速 (Rotor Speed)",
    "a": "温度 (Temperature)"
}

def generate_wind_turbine_data(n_samples=500, contamination=0.1, random_state=42, return_dataframe=True):
    """
    生成模拟的风机数据

    Parameters
    ----------
    n_samples : int
        样本总数
    contamination : float
        异常比例 (0 < contamination < 0.5)
    random_state : int
        随机种子
    return_dataframe : bool
        是否返回 pandas DataFrame (默认 True)，否则返回 numpy array

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        特征数据
    y : np.ndarray
        标签 (0: 正常, 1: 异常)
    """
    np.random.seed(random_state)
    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers

    # --- 1. 生成正常数据 (遵循物理规律) ---

    # 风速 (b): 3m/s 到 20m/s
    wind_speed = np.random.uniform(3, 20, n_inliers)

    # 功率 (dc): 遵循功率曲线 Power ~ v^3 (但在切入切出风速间近似 S 形)
    # Rated Power: 2000 kW
    rated_power = 2000
    # S 型曲线模拟: Power = Rated / (1 + exp(-k * (v - v0)))
    power = rated_power / (1 + np.exp(-(wind_speed - 10) * 0.8))
    # 添加噪声
    power += np.random.normal(0, 50, n_inliers)
    power = np.maximum(power, 0)

    # 转速 (ah): 与风速/功率正相关，但在额定功率后保持恒定
    # 简单模拟: 随风速线性增加，直到达到最大转速 (e.g. 15 rpm)
    rotor_speed = np.minimum(wind_speed * 1.2, 15)
    rotor_speed += np.random.normal(0, 0.5, n_inliers)

    # 温度 (a): 舱外温度，与风速弱相关或独立，这里假设为独立分布
    temperature = np.random.normal(20, 5, n_inliers)

    X_inliers = np.column_stack((wind_speed, power, rotor_speed, temperature))

    # --- 2. 生成异常数据 (违反物理规律) ---

    # 异常类型 A: 高风速低功率 (停机、限电、故障)
    outlier_wind_A = np.random.uniform(10, 20, int(n_outliers * 0.6))
    outlier_power_A = np.random.uniform(0, 500, int(n_outliers * 0.6))
    outlier_rotor_A = np.random.uniform(0, 5, int(n_outliers * 0.6)) # 转速也很低
    outlier_temp_A = np.random.normal(20, 5, int(n_outliers * 0.6))

    X_outliers_A = np.column_stack((outlier_wind_A, outlier_power_A, outlier_rotor_A, outlier_temp_A))

    # 异常类型 B: 传感器故障 (随机噪声)
    n_remain = n_outliers - len(X_outliers_A)
    outlier_wind_B = np.random.uniform(3, 20, n_remain)
    outlier_power_B = np.random.uniform(0, 2200, n_remain) # 随机功率
    outlier_rotor_B = np.random.uniform(0, 20, n_remain) # 随机转速
    outlier_temp_B = np.random.normal(20, 10, n_remain)

    X_outliers_B = np.column_stack((outlier_wind_B, outlier_power_B, outlier_rotor_B, outlier_temp_B))

    X_outliers = np.concatenate([X_outliers_A, X_outliers_B])

    # --- 3. 合并与打乱 ---
    X = np.concatenate([X_inliers, X_outliers])
    y = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    if return_dataframe:
        # 使用 column_mapping.json 中的代码作为列名
        columns = ["b", "dc", "ah", "a"]
        X_df = pd.DataFrame(X, columns=columns)
        return X_df, y
    else:
        return X, y

def get_feature_names():
    """返回特征代码到中文名称的映射"""
    return FEATURE_MAPPING
