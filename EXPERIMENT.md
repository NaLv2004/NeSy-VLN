# Blocks World 验证

分支：`exp/blocks-world-baseline`。

在积木世界中测试匿名状态、生成转移规则与子目标规划。此任务不同于 ObjectNav，不将其成绩与导航 80 任务结果混用。

## 主要入口

- [`run.py`](predictive_rules/blocks_world_v1/run.py)
- [`model.py`](predictive_rules/blocks_world_v1/model.py)
- [`legal_plan.py`](predictive_rules/blocks_world_v1/legal_plan.py)

完整源码位于 `predictive_rules/blocks_world_v1/`。共 6 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
