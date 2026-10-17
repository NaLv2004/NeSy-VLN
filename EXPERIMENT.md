# 多样化训练轨迹采集

分支：`exp/v70-diverse-collection`。

从训练池固定抽取 1200 个普通场景短中距离任务，预先隔离 960 个拟合任务与 240 个留出任务。采集实际轨迹及候选动作结果，不纳入导航验证任务。此分支是数据采集流程。

## 主要入口

- [`worker.py`](predictive_rules/diverse_history_v70/worker.py)

完整源码位于 `predictive_rules/diverse_history_v70/`。共 17 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
