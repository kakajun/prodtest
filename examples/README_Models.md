# PyOD 示例模型总结

本文档总结了 PyOD 库 `examples` 目录中包含的异常检测模型。

| 文件名 | 模型 | 算法 | 特点 | 适用场景 | 不宜适用情况 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `abod_example.py` | ABOD | 基于角度的异常检测 (Angle-Based Outlier Detection) | 概率模型，利用角度缓解维度灾难 | 高维数据 | 极大规模数据（计算复杂度高 $O(N^3)$ 或 $O(N^2)$） |
| `ae1svm_example.py` | AE1SVM | 基于自编码器的单类支持向量机 | 深度学习，自编码器与OCSVM结合 | 复杂非线性数据 | 小数据集 |
| `alad_example.py` | ALAD | 对抗性学习异常检测 (Adversarially Learned Anomaly Detection) | 基于GAN，深度学习 | 复杂分布，高维数据 | 小数据集，训练不稳定 |
| `auto_encoder_example.py` | AutoEncoder | 全连接自编码器 | 深度学习，基于重构误差 | 图像、序列等复杂非线性数据 | 简单的线性可分数据 |
| `cblof_example.py` | CBLOF | 基于聚类的局部异常因子 | 基于聚类，将异常视为小簇 | 具有清晰簇结构的数据 | 异常值形成大簇的情况 |
| `cd_example.py` | CD | 库克距离 (Cook's Distance) | 线性模型，回归影响点分析 | 线性回归分析，多变量数据 | 非线性关系 |
| `cof_example.py` | COF | 基于连接性的异常因子 | 基于邻近度，LOF的变体，考虑连接路径 | 具有复杂几何结构的数据（如线条、曲线） | 极大规模数据（计算较慢） |
| `comb_example.py` | Combination | 模型组合 (平均、最大值等) | 集成方法，组合多个检测器 | 需要高鲁棒性和稳定性的场景 | 单一模型已足够好，或资源受限 |
| `copod_example.py` | COPOD | 基于Copula的异常检测 | 概率模型，无参数，快速 | 各种规模数据，无需调参 | 变量间存在极复杂的非线性依赖（Copula难以捕捉） |
| `deepsvdd_example.py` | DeepSVDD | 深度单类分类 (Deep One-Class Classification) | 深度学习，将正常数据映射到超球体 | 高维、复杂非线性数据 | 小数据集，多模态正态分布（可能效果受限） |
| `devnet_example.py` | DevNet | 偏差网络深度异常检测 | 深度学习，弱监督 | 小样本异常检测（有少量已知异常标签） | 完全无监督（无任何标签） |
| `dif_example.py` | DIF | 深度隔离森林 (Deep Isolation Forest) | 集成/深度学习 | 高维、非线性、大规模数据 | 小数据集 |
| `ecod_example.py` | ECOD | 基于经验累积分布函数的异常检测 | 概率模型，无参数，极快 | 大规模数据，实时检测，要求可解释性 | 变量间依赖极强且分布极不规则 |
| `feature_bagging_example.py` | Feature Bagging | 特征装袋 (Feature Bagging) | 集成，在随机特征子集上训练 | 高维数据，特征含噪 | 特征数量极少的数据 |
| `gmm_example.py` | GMM | 高斯混合模型 (Gaussian Mixture Model) | 概率模型，混合高斯分布 | 数据符合高斯混合分布假设 | 分布极不规则且不符合高斯假设 |
| `hbos_example.py` | HBOS | 基于直方图的异常评分 | 统计/邻近度，假设特征独立 | 极大规模数据，极快 | 特征间存在强相关性的数据 |
| `iforest_example.py` | IForest | 隔离森林 (Isolation Forest) | 集成，基于树结构 | 大规模、高维数据，通用性强 | 极高维稀疏数据（有时效果下降） |
| `inne_example.py` | INNE | 基于隔离的最近邻集成 | 集成，隔离+最近邻 | 类似隔离森林，但基于最近邻 | 极大规模数据（距离计算受限） |
| `kde_example.py` | KDE | 核密度估计 (Kernel Density Estimation) | 概率模型 | 低维、平滑分布的数据 | 高维数据（维度灾难） |
| `knn_example.py` | KNN | k近邻 (k-Nearest Neighbors) | 基于距离，直观 | 各种数据类型 | 极大规模数据（慢），极高维 |
| `knn_mahalanobis_example.py` | KNN (Mahalanobis) | 马氏距离k近邻 | 基于距离，考虑相关性 | 变量间存在线性相关的数据 | 高维且协方差矩阵不可逆/难以估计 |
| `kpca_example.py` | KPCA | 核主成分分析 (Kernel PCA) | 线性/非线性（核方法） | 非线性降维异常检测 | 大数据集（计算复杂度高） |
| `lmdd_example.py` | LMDD | 线性偏差检测方法 | 线性模型 | 线性相关数据 | 非线性数据 |
| `loci_example.py` | LOCI | 局部相关积分 | 基于邻近度，无需k参数 | 局部密度变化剧烈的数据 | 计算极其昂贵，大快数据 |
| `loda_example.py` | LODA | 轻量级在线异常检测 | 集成，在线学习 | 流数据，实时检测 | 特征间关系极其复杂 |
| `lof_example.py` | LOF | 局部异常因子 (Local Outlier Factor) | 基于邻近度，密度检测 | 局部密度不均匀的数据 | 极大规模数据，仅有全局异常 |
| `lscp_example.py` | LSCP | 局部选择性集成 (LSCP) | 集成选择 | 数据局部特性变化大的场景 | 计算资源受限（需训练多个基模型） |
| `lunar_example.py` | LUNAR | 基于图神经网络的统一局部异常检测 | 图模型 / GNN | 图数据，或复杂关系数据 | 简单表格数据（可能过拟合） |
| `mad_example.py` | MAD | 中位数绝对偏差 (Median Absolute Deviation) | 概率/鲁棒统计 | 单变量或简单多变量，鲁棒性要求高 | 复杂多模态分布 |
| `mcd_example.py` | MCD | 最小协方差行列式 (Minimum Covariance Determinant) | 线性模型，鲁棒协方差 | 线性相关，椭圆分布数据 | 非线性分布，极高维 |
| `mo_gaal_example.py` | MO_GAAL | 多目标生成对抗主动学习 | GAN，主动学习 | 复杂分布，生成潜在异常 | 小数据，训练不稳定 |
| `ocsvm_example.py` | OCSVM | 单类支持向量机 (One-Class SVM) | 线性/核方法 | 小到中等规模，非线性边界 | 极大规模数据（训练慢），对噪声敏感 |
| `pca_example.py` | PCA | 主成分分析 (PCA) | 线性模型 | 线性降维，全局异常检测 | 非线性数据 |
| `qmcd_example.py` | QMCD | 拟蒙特卡洛偏差 | 概率模型 | 寻找分布非均匀性 | 高度复杂的依赖关系 |
| `rgraph_example.py` | RGraph | 基于R图的异常检测 | 图方法 | 图结构数据 | 计算复杂度高 |
| `rod_example.py` | ROD | 基于旋转的异常检测 | 基于邻近度，3D旋转 | 3D或低维数据，几何直观 | 高维数据 |
| `sampling_example.py` | Sampling | 基于采样的快速距离检测 | 概率/采样 | 大规模数据快速近似 | 高精度要求 |
| `so_gaal_example.py` | SO_GAAL | 单目标生成对抗主动学习 | GAN | 复杂分布 | 模式坍塌，训练不稳定 |
| `sod_example.py` | SOD | 子空间异常检测 | 基于邻近度，子空间 | 异常仅在特定子空间可见 | 异常在全空间分布 |
| `sos_example.py` | SOS | 随机异常选择 (Stochastic Outlier Selection) | 概率，基于亲和度 | 需要概率解释的场景 | 大规模数据（亲和度矩阵计算慢） |
| `suod_example.py` | SUOD | 可扩展无监督异常检测 (SUOD) | 加速框架 | 大规模异构集成学习 | 单一小模型场景 |
| `vae_example.py` | VAE | 变分自编码器 (VAE) | 深度学习，生成模型 | 复杂非线性，概率建模 | 简单线性数据 |
| `xgbod_example.py` | XGBOD | 基于极限提升的异常检测 | 集成，半监督/有监督 | 有少量标签的半监督场景，表格数据 | 完全无监督（无标签） |
