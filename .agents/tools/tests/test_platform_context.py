"""Independent model selection, task pinning and old-state compatibility."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import platform_context as pc
from design_calc_check import check_ub_budget
from test_statectl import (
    _make_harness_repo,
    _ready_stage5_gate,
    _write_stage5_report,
    sc,
)


def test_same_toolchain_uses_model_specific_budget():
    b2c = pc.resolve_platform("ascend-910b2c", "npuir")
    b3 = pc.resolve_platform("ascend-910b3", "npuir")
    assert type(b2c.adapter) is type(b3.adapter)
    assert b2c.adapter.precision_command("op.py", "L0") == b3.adapter.precision_command(
        "op.py", "L0"
    )
    assert b2c.adapter.profiler_args(
        "main", "output", 15, 5
    ) == b3.adapter.profiler_args("main", "output", 15, 5)
    design = "## 4.5 内存\n**UB**\n| buffer | 128KB |\n"
    assert check_ub_budget(design, b2c.design_budget())[0]["status"] == "pass"
    assert b3.design_budget()["ub_inflation"] is None
    assert pc.resolve_platform(state={}).design_budget()["ub_inflation"] == 1.7


def test_measurements_match_hardware_toolchain_stamp_and_workload(monkeypatch):
    source = pc.read_json
    measurement = {
        "hardware_id": "ascend-910b2c",
        "toolchain_id": "npuir",
        "toolchain_stamp": "fixture-build",
        "workload": "fixture",
        "metric": "ub_inflation_factor",
        "unit": "ratio",
        "value": 1.25,
        "source": "fixture",
    }
    monkeypatch.setattr(
        pc,
        "read_json",
        lambda path: (
            {"measurements": [measurement]}
            if path.name == "index.json"
            else source(path)
        ),
    )
    b2c = pc.resolve_platform("ascend-910b2c", "npuir")
    b3 = pc.resolve_platform("ascend-910b3", "npuir")
    assert b2c.measurements("fixture-build", "fixture") == [measurement]
    assert b2c.measurements("other-build", "fixture") == []
    assert b2c.measurements("fixture-build", "other-workload") == []
    assert b3.measurements("fixture-build", "fixture") == []


def test_probe_rejects_different_device_or_runtime(monkeypatch):
    context = pc.resolve_platform("ascend-910b3", "npuir")
    monkeypatch.setattr(
        type(context.adapter), "probe_device", lambda self: {"name": "Ascend910B2C"}
    )
    with pytest.raises(pc.PlatformError, match="active device reports"):
        context.probe()
    monkeypatch.setattr(
        type(context.adapter),
        "probe_device",
        lambda self: {"name": "Ascend910B3", "cores": {"cube": 20, "vector": 40}},
    )
    monkeypatch.setattr(
        type(context.adapter),
        "toolchain_stamp",
        lambda *a, **k: {
            "raw": "current",
            "tilelang": "1",
            "cann": "9",
            "compiler": "fixture",
        },
    )
    context = context.verified()
    assert context.runtime["device"]["cores"]["vector"] == 40
    monkeypatch.setattr(
        type(context.adapter),
        "toolchain_stamp",
        lambda *a, **k: {
            "raw": "updated",
            "tilelang": "1",
            "cann": "9",
            "compiler": "fixture",
        },
    )
    with pytest.raises(pc.PlatformError, match="differ from the task record"):
        context.probe()


def test_migration_children_inherit_selection(tmp_path):
    op_dir = tmp_path / "op"
    assert (
        sc(
            "migration",
            "init",
            "--dir",
            str(op_dir),
            "--op-slug",
            "op",
            "--family",
            "test",
            "--meta-path",
            "meta.json",
            "--functions",
            "func",
            "--hardware",
            "ascend-910b3",
            "--toolchain",
            "npuir",
        )[0]
        == 0
    )
    child = op_dir / "func"
    args = (
        "init",
        "--dir",
        str(child),
        "--project",
        "op",
        "--op",
        "func",
        "--scenario",
        "migration",
        "--migration-mode",
        "harness",
        "--task-dir",
        str(op_dir),
    )
    assert sc(*args, "--hardware", "ascend-910b2c")[0] == 1
    assert sc(*args)[0] == 0
    saved = json.loads((child / ".stage_state.json").read_text())
    aggregate = json.loads((op_dir / ".migration_state.json").read_text())
    assert saved["hardware_id"] == aggregate["hardware_id"] == "ascend-910b3"
    assert saved["toolchain_id"] == aggregate["toolchain_id"] == "npuir"
    assert saved["platform_config"] == aggregate["platform_config"]
    code, result = sc("platform", "--task-dir", str(child))
    assert code == 0, result
    assert result["platform"]["hardware_id"] == saved["hardware_id"]
    assert result["platform"]["toolchain_id"] == saved["toolchain_id"]
    assert (
        sc("platform", "--task-dir", str(child), "--hardware", "ascend-910b2c")[0] == 1
    )


def test_harness_scaffold_and_aggregate_state_handoff(tmp_path):
    repo, op_dir, meta = _make_harness_repo(tmp_path)
    directory = ("--dir", str(op_dir))
    platform = ("--hardware", "ascend-910b3", "--toolchain", "npuir")
    assert (
        sc(
            "init",
            *directory,
            "--project",
            "mop",
            "--op",
            "mop",
            "--scenario",
            "migration",
            "--migration-mode",
            "harness",
            "--stage-plan",
            "0",
            *platform,
        )[0]
        == 0
    )
    assert sc("start", "0", *directory)[0] == 0
    original_meta = meta.read_text()
    meta.write_text(
        json.dumps({**json.loads(original_meta), "extracted_functions": "funcA"})
    )
    rc, result = sc("gate", "0", *directory, "--meta", str(meta.relative_to(repo)))
    assert rc == 1 and "S0-META-FUNC" in {f["rule_id"] for f in result["failures"]}
    meta.write_text(original_meta)
    for relative in (
        "tileops/manifest/reduction.yaml",
        "tileops/workloads/reduction.py",
        "tileops/kernels/reduction/mop/mop.py",
        "tests/ops/test_mop.py",
        "benchmarks/ops/bench_mop.py",
    ):
        path = repo / "examples/TileOPs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# scaffold\n")
    meta_arg = str(meta.relative_to(repo))
    assert sc("complete", "0", *directory, "--meta", meta_arg)[0] == 0
    assert (
        sc(
            "migration",
            "init",
            *directory,
            "--op-slug",
            "mop",
            "--family",
            "reduction",
            "--meta-path",
            str(meta),
            "--functions",
            "funcA",
            "funcB",
            *platform,
        )[0]
        == 0
    )
    aggregate_path = op_dir / ".migration_state.json"
    assert pc.task_state_path(op_dir) == aggregate_path
    assert sc("gate", "0", *directory)[0] == 0
    rc, result = sc("gate", "5", "--migration-dir", str(op_dir), *platform)
    assert rc == 1 and "S5-PKG" in {f["rule_id"] for f in result["failures"]}


def test_wrong_task_directory_and_malformed_state_are_rejected(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    selected = pc.resolve_platform("ascend-910b3", "npuir").selection
    path = task / ".stage_state.json"
    path.write_text(json.dumps(selected))
    code, result = sc("platform", "--task-dir", str(task / "typo"))
    assert code == 1 and result["errors"][0]["code"] == "E-PLATFORM"
    path.write_text(json.dumps({**selected, "platform_runtime": []}))
    code, result = sc("platform", "--task-dir", str(task))
    assert code == 1 and "invalid platform_runtime" in result["errors"][0]["message"]


def test_stage5_gate_reads_explicit_aggregate_and_report(tmp_path):
    repo, directory, package, _ = _ready_stage5_gate(tmp_path)
    _write_stage5_report(repo, package)
    path = directory / ".migration_state.json"
    selection = pc.resolve_platform("ascend-910b3", "npuir").selection
    path.write_text(json.dumps({**json.loads(path.read_text()), **selection}))
    args = ("gate", "5", "--migration-dir", str(directory))
    rc, result = sc(*args, "--hardware", "ascend-910b3", "--toolchain", "npuir")
    assert rc == 0, result
    rc, result = sc(*args, "--hardware", "ascend-910b2c", "--toolchain", "npuir")
    assert rc == 1 and result["errors"][0]["code"] == "E-PLATFORM"
