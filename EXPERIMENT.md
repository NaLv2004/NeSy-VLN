# 视觉证据与规则继续训练

分支：`exp/v72-evidence-continuation`。

从 44/80 基线继续拟合视觉证据与转移规则，使用固定的 288 条新增数据子集。训练 1000 次更新，选中第 600 次检查点；同一 80 任务为 35/80，未优于基线。

## 主要入口

- [`train_joint.py`](predictive_rules/continued_evidence_v72/train_joint.py)
- [`evaluate.py`](predictive_rules/continued_evidence_v72/evaluate.py)

完整源码位于 `predictive_rules/continued_evidence_v72/`。共 13 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
