# TDengine 异常检测项目

本项目实现了一个基于 TDengine 存储的工业传感器数据的异常检测流程。它为每台设备（`equ_code`）分别训练模型，以检测运行中的异常情况。

## 项目结构

```
examples/taos/
├── detect_anomalies.py    # 主脚本：拉取数据 -> 训练 -> 绘图 -> 保存
├── column_mapping.json    # 映射文件：将数据库列名 (a, b...) 映射为可读名称
├── models/                # 模型保存目录 (脚本运行后生成)
│   ├── ECOD_F01.joblib
│   └── IForest_F01.joblib
├── *_results_*.png        # 各设备异常分数的可视化图表
├── debug.log              # 执行日志
└── README.md              # 本说明文件
```

## 方法论

### 1. 数据来源
- **数据库**: TDengine (`station_data.stable_gtjjlfgdzf`)
- **数据选择**:
  - 获取传感器读数 (`a`-`da`) 和故障代码 (`gu`-`hx`)。
  - **分设备训练**: 数据按 `equ_code`（如 F01, F02）进行划分。这确保了独立学习每台机器独特的运行特性。

### 2. 预处理
- **清洗**: 通过前向/后向填充处理缺失值。
- **缩放**: 使用 `StandardScaler` 对特征进行标准化（均值=0，标准差=1），这对 IForest 等基于距离的算法至关重要。
- **切分**:
  - **训练集**: 1月22日 - 1月27日（学习正常模式）
  - **测试集**: 1月27日 - 1月29日（评估检测效果）1

### 3. 模型
使用了 `pyod` 库中的两种无监督异常检测算法：

- **ECOD (Empirical Cumulative Distribution Functions, 经验累积分布函数)**:
  - **原理**: 估计每个维度的尾部分布。它在处理高维数据时查找离群点非常快速且有效，无需调整超参数。
  - **适用场景**: 擅长检测极值异常。

- **IForest (Isolation Forest, 孤立森林)**:
  - **原理**: 构建随机树来隔离数据点。异常点比正常点更容易被隔离（路径更短）。
  - **适用场景**: 对复杂的非线性关系具有鲁棒性。

### 4. 输出与解读
- **可视化**: `ECOD_results_{equ_code}.png`
  - **蓝线**: 异常分数 (Anomaly Score)。分数越高，代表越异常。
  - **红虚线**: 阈值 (Threshold) (由 `contamination` 污染率参数决定)。
  - **红 X**: 实际故障 (Ground truth faults) (源自故障列中的非零值)。
- **保存的模型**: 以 `.joblib` 文件格式存储在 `models/` 文件夹中。稍后可加载用于实时推理。

## 使用方法

1. **安装依赖**:
   ```bash
   pip install taos-ws-py pandas numpy matplotlib seaborn pyod scikit-learn joblib
   ```

2. **运行流程**:
   ```bash
   python detect_anomalies.py
   ```

3. **查看结果**:
   - 打开 `debug.log` 查看进度。
   - 检查 `.png` 文件进行可视化验证。
   - 使用 `models/` 中的模型进行后续部署。
