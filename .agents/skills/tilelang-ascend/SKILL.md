---
name: tilelang-ascend
description: 面向 Ascend950DT_9582 的 TileLang ascend 设计、检视、开发、调优、诊断和 TileOPs 集成。
---

读取 [平台规则](../../platforms/toolchains/ascend/workflow.md) 和平台配置给出的容量与证据。沿用调用方定义的阶段产物、章节和退出信号。

- Stage 0：按平台规则准备目标 TileOPs wrapper 和 .migration_meta.json，记录每个提取函数的源码路径和迁移输入。检查目标 manifest 键及 benchmark 入口，返回逐函数迁移任务。
- Stage 1：从算子语义、shape/dtype 和参考实现推导算法、搬运与分核。比较支持的编程模式，按任务容量核算 UB/L1/L0C 与多版本缓冲。在 DESIGN.md 记录物理核数查询和逻辑到物理核的映射。
- Stage 2：按调用方规定的维度检视 DESIGN.md。优先检查语义、资源预算、尾块、写入范围及同步，返回 REVIEW.md 和检视结论。
- Stage 3：从平台 kernel 模板实现算子和独立 PyTorch golden，执行 L0、代表 shape、多 dtype 和边界用例。失败时保留日志和最小复现，按环境、编译、运行或精度分类。
- Stage 4：按平台规则枚举 workload，先验精度，再使用同一官方 profiler 协议对比基线与候选。按 Stage 4 契约保存 workload inventory、原始采集、性能记录和最终版本，使用调用方给出的实验预算。
- Stage 5：按平台规则接线并运行单算子验收。排错使用上游 Ascend wrapper、manifest 和 runner 的真实日志，返回调用方定义的集成结果。
