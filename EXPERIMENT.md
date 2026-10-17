# 扩大数据后的视觉与动力学训练

分支：`exp/v71-expanded-dynamics`。

在新采集数据上训练视觉残差和观测转移解码器，保留上下文、定义及目标模块。包含采集完成后的训练衔接、大批量选择和 GPU 符号规划。本次源码归档不声明最终导航成绩。

## 主要入口

- [`train_joint.py`](predictive_rules/expanded_dynamics_v71/train_joint.py)
- [`handoff.py`](predictive_rules/expanded_dynamics_v71/handoff.py)
- [`evaluate.py`](predictive_rules/expanded_dynamics_v71/evaluate.py)

完整源码位于 `predictive_rules/expanded_dynamics_v71/`。共 17 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
