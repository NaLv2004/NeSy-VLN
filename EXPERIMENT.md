# GPU 符号推导与批处理优化

分支：`perf/v74-gpu-symbolic`。

将相同有限状态 Bellman 递推移到 GPU，并测量批量大小。8 个程序、32 步组件测试约 27.9 倍加速；524288 个动作选择一致。该数值不是端到端训练加速倍数。

## 主要入口

- [`gpu_planner.py`](predictive_rules/gpu_efficiency_v74/gpu_planner.py)
- [`autobatch.py`](predictive_rules/gpu_efficiency_v74/autobatch.py)
- [`benchmark.py`](predictive_rules/gpu_efficiency_v74/benchmark.py)

完整源码位于 `predictive_rules/gpu_efficiency_v74/`。共 11 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
