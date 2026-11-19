# Ascend 950 知识索引

条目规范见 [.agents/skills/_shared/standards/knowledge-entry-schema.md](../../../../skills/_shared/standards/knowledge-entry-schema.md)。

适用于 Ascend950DT_9582 与 ascend 工具链。阶段技能按此索引读取已有条目，并核对来源及任务工具链版本。

- [PL-950-vector-execution：950 向量执行和尾块](PL-950-vector-execution.md)

- [PL-950-row-primitives：单行转换、归约与广播](PL-950-row-primitives.md)
- [PL-950-cube-tile：显式 L0 操作数的单 tile GEMM](PL-950-cube-tile.md)
- [ALG-950-formulas：CPU 数学参考，非 950 kernel 认证](ALG-950-formulas.md)
- [接口与资源规则](../references/programming.md)
- [迁移与诊断](../references/migration-diagnostics.md)
- [核验来源与范围](../references/evidence.md)

- [算法候选](algorithm-candidates.md)
- [瓶颈模式](bottlenecks.md)
- [能力缺口](capability-gaps.md)
- [PL-950-normalization：带权重的 LayerNorm 与 RMSNorm](operator-patterns.md)
- [PL-950-attention：单头 dense attention 的 SIMT 正确性参考](operator-patterns.md)
- [PL-950-gemm-tiles：GEMM 输入布局、K 分块与尾块](gemm-tiles.md)
- [PL-950-mixed-epilogue：Cube GEMM 与 Vector ReLU 的混合 kernel](execution-patterns.md)
- [PL-950-manual-sync：关闭自动调度后的单 tile Cube 管线同步](execution-patterns.md)
- [PL-950-atomic-splitk：两个 Cube 核的 split-K 原子累加](execution-patterns.md)
- [PL-950-conversion-layout：SIMT 转换特殊值与带广播的转置](numerical-boundaries.md)
- [PL-950-simd-reduce：显式 SIMD 寄存器行归约](numerical-boundaries.md)
- [PL-950-softmax-boundaries：FP32 softmax 的有限极值与非有限输入](numerical-boundaries.md)
- [TRAP-950-simd-shared-reduce：高层 SIMD 二维 shared→shared 归约的输出差异](numerical-boundaries.md)
