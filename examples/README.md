### 如何运行示例？

首先应该安装 pyod，或者下载 github 仓库。
````cmd
pip install pyod
pip install --upgrade pyod # 确保安装了最新版本！
````

之后，您可以简单地复制粘贴代码或直接运行示例。

---

### 示例介绍
示例的结构如下：
- 示例命名为 XXX_example.py，其中 XXX 是模型名称。
- 对于所有示例，您可以在 pyod/models/ 下找到相应的模型。

例如：
- kNN: knn_example.py
- HBOS: hbos_example.py
- ABOD: abod_example.py
- ... 其他独立算法
- 组合框架: comb_example.py

此外，compare_all_models.py 用于比较所有已实现的算法。
一些示例有 Jupyter Notebook 版本，位于 [Jupyter Notebooks](https://github.com/yzhao062/Pyod/tree/master/notebooks)

---

### 如果看到 "xxx module could be found" 或 "Unresolved reference" 怎么办

**首先检查是否使用 pip 安装了 pyod。**

如果您没有安装，只是下载了 github 仓库，请确保代码顶部包含以下代码。
如果未安装 pyod，示例将依赖以下代码导入模型：

```python
import sys
sys.path.append("..")
```
这是在 **未安装 pyod** 的情况下进行相对导入的 **临时解决方案**。

如果使用 pip 安装了 pyod，则无需导入 sys 和 sys.path.append("..")
可以随意删除这些行并直接导入 pyod 模型。
