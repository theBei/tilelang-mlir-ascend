---
id: PL-950-vector-execution
kind: pattern
device: Ascend950DT_9582
apis: [T.SimdVF, T.SimtVF, T.copy]
---

# 950 向量执行和尾块

TileOPs 的 Ascend Abs 实现对固定 SIMD 向量宽度的路径在 UB 内做尾块填充，并按有效元素数搬入、写回 GM。该实现也包含 SIMT 分支。根据 dtype 与数据访问方式选择路径，记录填充后的 UB 占用。

证据见 [TileOPs Ascend Abs](https://github.com/yyttt6/TileOPs/blob/ascend950/src/tileops/kernels/ascend/elementwise/abs.py)。
