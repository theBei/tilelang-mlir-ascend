# TileLang ascend 与 Ascend950DT_9582

## 环境与硬件

以 [TileOPs ascend950 工具链说明](https://github.com/yyttt6/TileOPs/blob/ascend950/docs/ascend950/toolchain.md) 和同目录的锁文件为准。使用 ARM64、官方 TileLang 0.1.15 wheel、CANN 9.2.0 release 编译 SDK、独立 CANN 9.1.0 runtime/OPP，以及锁定的 Python、PyTorch、torch-npu 和 C++ 编译器。按上游文档设置 ASCEND_HOME_PATH、ASCEND_RUNTIME_ROOT、ASCEND_NPU_ARCH=dav-3510 以及动态库路径。

用以下命令在 CPU 上检查软件版本和文件指纹。真机上改用 --probe，查询 torch.npu.get_device_properties 并验证设备型号和工具链。正式运行前用同一组合初始化或验证任务状态。

    python3 .agents/tools/statectl.py platform --hardware ascend-950dt-9582 --toolchain ascend --check-environment

## 设计、实现与排错

从任务的平台配置读取容量预算。记录 torch.npu.get_device_properties 返回的 Cube 和 Vector 核数，根据实际执行路径选择 SIMD、SIMT、Cube 或混合模式。向量操作参考上游 elementwise 的 T.SimdVF、T.simd.* 或 T.SimtVF。Cube 设计对照上游 GEMM 检查布局、累加缓冲和数据搬运。ascend.batch_schedule 的跨 block barrier 依赖尚未通过该工具链认证。设计评审核对实际 tile 占用、双缓冲、尾块保护和同步区间。

使用 import tilelang.ascend.language as T、@T.prim_func 和 tilelang.compile(..., target="ascend", out_idx=...)。参照平台 kernel 模板。用 PyTorch 参考实现和 --level L0 / --level all 执行分层精度验证。排错先检查锁定环境与目标架构，再用最小 shape 复现并保留编译器日志、生成代码和异常输入。

## 性能采集

950 使用上游 guarded torch_npu.profiler 协议。各次设备执行区间并集的算术均值为 mean_device_busy_union_us，原始目录记入 ascend_profile_path。先通过精度验证，使用相同 workload、输入规格和站点时钟对比候选与 current best。

- 独立算子公开模板中的 profile_workload(**params)，返回构建好的 callable 和输入 tuple。运行下列命令，guard 成功退出后才会生成 measurement.json。从中读取 duration_us，并确认 guard_exit_code 为 0。原始数据在结果目录的 profiles/。

      python3 .agents/tools/toolchains/ascend_profile.py --tileops-root <950仓> --kernel <候选.py> --workload '<JSON参数>' --clock-config <时钟配置> --run-dir <新目录> --owner <占卡标识>

- TileOPs 用以下命令。上游的 --implementation-factory 模块:函数 可在同轮对比实现。替换 benchmark 子命令为 summary 并传入 --run-dir <结果目录> 可读取各 workload 和实现的均值。--case-regex 和 --verify-only 用于单用例或正确性冒烟。

      python3 .agents/tools/toolchains/ascend_tileops.py benchmark --tileops-root <950仓> --op <manifest键> --run-dir <新目录> --clock-config <时钟配置> --owner <占卡标识>

- 按 Stage 4 工作负载协议填写 inventory、perf_records.jsonl 和 winner。TileOPs 的完整 workload 从上游 benchmark cases 枚举。保留真实运行的原始目录和当前任务的 statectl platform --task-dir 证据。

## CG-950-batch-barrier

工具链锁记录的未认证功能为 ascend.batch_schedule 的跨 block barrier 依赖。需要该依赖的设计在取得独立验证前应选择已有支持路径。

## TileOPs 工程迁移

使用指定的独立 TileOPs ascend950 checkout。开发规则、manifest、benchmark cases 与 backend 注册参考 [上游开发说明](https://github.com/yyttt6/TileOPs/blob/ascend950/docs/ascend950/development.md)。

Stage 0 提取源函数，在 Agent 仓的 examples/{op_slug}/ 下保存 .migration_meta.json。保留现有 op_name（manifest 键）、op_slug、family、extracted_functions 及源代码定位。增加 tileops_root（目标 950 checkout）和 wrapper_path（相对于目标 src/tileops/kernels/ascend/ 的 Python 文件路径）。按目标仓布局准备该算子的 manifest、cases 和 Ascend wrapper，沿用已有的上游实现作为范例。

Stage 5 在 Agent 仓根执行配置中的 integration.script --meta <meta_path>，把 Stage 3 kernel 和 DESIGN.md 快照复制到 Ascend wrapper 同目录的 {op_slug}_kernel/。在该算子的 wrapper 中用 from .{op_slug}_kernel.{func} import ... 导入各集成实现，保持原有调用签名、dtype 和参数映射，按上游 register_kernel_type 注册 ascend backend。先进行单用例冒烟，然后执行完整算子正式 benchmark。将 receipt.json、result.json 和原始采集保留在 examples/{op_slug}/ 下本次运行目录。在集成包写 integration_log.md 和 integration_report.json，报告引用格式为 {"run_dir": "<相对于 examples/{op_slug} 的本次运行目录>"}。Stage 5 使用完整用例的正式运行及源码指纹验收。会话耗时记录在任务原有 Markdown 中。
