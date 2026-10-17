# 八步离散历史预测

分支：`exp/v69-history8`。

用历史观测编号和动作预测下一观测状态，保持符号规划。独立状态预测留出集表现退化，未通过候选筛选，没有宣称新的 80 任务成功率。

## 主要入口

- [`train_history.py`](predictive_rules/history_transition_v69/train_history.py)

完整源码位于 `predictive_rules/history_transition_v69/`。共 22 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
