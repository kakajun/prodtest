import taosws
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 配置
DB_URL = "taosws://root:taosdata@192.168.2.110:6041"
DB_NAME = "station_data"
TABLE_NAME = "stable_gtjjlfgdzf"
START_DATE = "2026-01-22"
END_DATE = "2026-01-29"

# 选定特征
# b: 风速
# dc: 有功功率
# a: 环境温度
# bm: 发电机定子温度 (U相)
# cd: 发电机转速
FEATURE_KEYS = ['b', 'dc', 'a', 'bm', 'cd']

def log(msg):
    print(f"[系统] {msg}")

def load_data(equ_code):
    log(f"正在从 TDengine 加载 {equ_code} 的正常数据...")
    try:
        conn = taosws.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_NAME}")

        cols = ['ts'] + FEATURE_KEYS
        cols_str = ", ".join(cols)

        query = f"SELECT {cols_str} FROM {TABLE_NAME} WHERE ts >= '{START_DATE}' AND ts < '{END_DATE}' AND equ_code = '{equ_code}'"
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()

        if not data:
            return None

        df = pd.DataFrame(data, columns=cols)

        # 数值转换
        for col in FEATURE_KEYS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        return df
    except Exception as e:
        log(f"错误: {e}")
        return None

def inject_anomalies(df):
    log("正在注入合成异常（模拟故障）...")

    # 0 = 正常
    df['label'] = 0
    df['fault_type'] = 'Normal'

    # --- 故障类型 1: 功率性能低下 (效率损失) ---
    # 逻辑: 风速大，但功率低。
    # 我们选取 10% 的数据来模拟这种情况。
    n_samples = int(len(df) * 0.1)
    indices_1 = np.random.choice(df.index, n_samples, replace=False)

    # 我们创建这些行的副本作为异常数据添加（这样我们保留了原始的正常数据）
    fault_1_df = df.loc[indices_1].copy()
    # 模拟: 功率下降到应有值的 40%
    fault_1_df['dc'] = fault_1_df['dc'] * 0.4
    fault_1_df['label'] = 1
    fault_1_df['fault_type'] = 'Power_Underperformance'

    # --- 故障类型 2: 发电机过热 ---
    # 逻辑: 功率正常，但温度过高。
    # 我们再选取 10% 的数据。
    indices_2 = np.random.choice(df.index, n_samples, replace=False)
    fault_2_df = df.loc[indices_2].copy()
    # 模拟: 温度升高 30 度
    fault_2_df['bm'] = fault_2_df['bm'] + 30.0
    fault_2_df['label'] = 2
    fault_2_df['fault_type'] = 'Generator_Overheating'

    # 合并所有数据
    final_df = pd.concat([df, fault_1_df, fault_2_df], ignore_index=True)

    # 打乱数据
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    log(f"数据集创建完成: {len(df)} 条正常, {len(fault_1_df)} 条功率偏低, {len(fault_2_df)} 条过热故障")
    return final_df

def train_classifier(df):
    X = df[FEATURE_KEYS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log("正在训练随机森林分类器...")
    # 使用随机森林，因为它提供特征重要性并且是非线性的
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    log("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '功率偏低', '发电机过热']))

    # 特征重要性
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    log("\n特征重要性 (影响分类的关键因素):")
    for f in range(X.shape[1]):
        print(f"{f+1}. {FEATURE_KEYS[indices[f]]:<10} {importances[indices[f]]:.4f}")

    return clf

def verify_with_mock_data(clf):
    log("\n=== 使用特定模拟数据进行验证 ===")

    # 手动创建特定场景来测试模型逻辑

    # 场景 1: 正常运行 (高风速, 高功率, 正常温度)
    # 风速=10m/s, 功率=2000kW, 温度=60C
    mock_normal = pd.DataFrame([[10.0, 2000.0, 25.0, 60.0, 1500.0]], columns=FEATURE_KEYS)

    # 场景 2: 功率损失 (高风速, 低功率, 正常温度)
    # 风速=10m/s, 功率=800kW (太低!), 温度=60C
    mock_fault1 = pd.DataFrame([[10.0, 800.0, 25.0, 60.0, 1500.0]], columns=FEATURE_KEYS)

    # 场景 3: 过热 (高风速, 高功率, 高温度)
    # 风速=10m/s, 功率=2000kW, 温度=95C (太热!)
    mock_fault2 = pd.DataFrame([[10.0, 2000.0, 25.0, 95.0, 1500.0]], columns=FEATURE_KEYS)

    # 预测
    p_normal = clf.predict(mock_normal)[0]
    p_fault1 = clf.predict(mock_fault1)[0]
    p_fault2 = clf.predict(mock_fault2)[0]

    labels = {0: '正常', 1: '功率偏低', 2: '发电机过热'}

    print(f"模拟 1 (风速=10, 功率=2000, 温度=60) -> 预测结果: {labels[p_normal]} (预期: 正常)")
    print(f"模拟 2 (风速=10, 功率=800,  温度=60) -> 预测结果: {labels[p_fault1]} (预期: 功率偏低)")
    print(f"模拟 3 (风速=10, 功率=2000, 温度=95) -> 预测结果: {labels[p_fault2]} (预期: 发电机过热)")

def main():
    # 1. 获取真实数据作为基准
    df = load_data('F01')
    if df is None or df.empty:
        log("未找到数据。")
        return

    # 2. 注入故障
    train_df = inject_anomalies(df)

    # 3. 训练
    model = train_classifier(train_df)

    # 4. 验证
    verify_with_mock_data(model)

    # 保存
    joblib.dump(model, 'fault_classifier.joblib')
    log("模型已保存至 fault_classifier.joblib")

if __name__ == "__main__":
    main()
