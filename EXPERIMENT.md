# 反事实状态转移学习

分支：`exp/v63-counterfactual-closure`。

从同一实际观测采集不同候选动作的结果，监督离散状态转移。当前普通场景 80 任务为 39/80。

## 主要入口

- [`train_closure.py`](predictive_rules/counterfactual_closure_v63/train_closure.py)
- [`closure_objective.py`](predictive_rules/counterfactual_closure_v63/closure_objective.py)
- [`planner.py`](predictive_rules/counterfactual_closure_v63/planner.py)

完整源码位于 `predictive_rules/counterfactual_closure_v63/`。共 35 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
