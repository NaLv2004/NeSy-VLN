# 远端环境与外部资产

原实验根目录：`/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015`。

策略 Python：`/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python`。

每条路线通常把 `predictive_rules/<name>/*.py` 部署到远端 `src_<name>/`，输出保存在 `run_<name>/`。脚本可能显式约束 CUDA_VISIBLE_DEVICES 和 WORLD_SIZE；不要在已有任务占用的 GPU 上直接运行历史启动脚本。

已确认 44/80 基线：`/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_goal_probability_v68/checkpoint_000300.pt`。

SHA-256：`dba453f4772d7a201ba9b595edd5952d24dd4221bb3b0f41c7d253a15c96f3ed`。

新增采集目录：`/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diverse_history_v70`。1200 任务的拟合/留出划分在采样前固定；训练样本、图像和 manifest 数据本体未上传。v72/v73 使用其中固定的 288 任务子集；v75/v76 使用 768 任务子集。模型选择用的状态预测留出集与 80 任务导航开发集是不同层次的评估。

`predictive_rules/predictive_nesy/environment.py` 还依赖实验父目录的 `original_ppo_resume_v99/src`、`symbolic_execution_v38/configs/grounded_outcome_feedback.json`、历史视觉编码器权重、词表及其 COMMITTED 记录。运行时通过 `nesy_vla.runtime.setup` 接入外部 `stripsnav` 模拟器工具。部分 Blocks World 脚本依赖 `/home/user/xjy/strips_wm_navigation_20261009` 的任务环境及数据。

环境包含 PyTorch、NumPy、Pillow、Paramiko，模拟器侧需要 Habitat 及对应场景资产。本次未伪造未经验证的锁定版本或依赖安装清单；迁移运行环境时应从现有远端导出真实版本。

SSH 凭证从环境变量读取；本仓库不提供凭证。本地辅助脚本中的服务器地址、目录和网络接口是部署配置，换机器需要调整。

固定验证任务及过去最终测试集继续独立保存，禁止加入采集、SFT 或 RL 的训练池。已有 checkpoint 和数据不会因本次 Git 导出被移动、删除或重写。
