"""Toolchain rules and tools for the existing NPUIR workflow."""

from __future__ import annotations

import ast
import csv
import glob
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .adapter import ToolchainAdapter

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / ".agents/platforms/toolchains/npuir/profile.json"
PROFILER = "msprof"
DURATION_FIELD = "Task Duration(us)"
METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)


class NpuirToolchain(ToolchainAdapter):
    profiler = PROFILER
    duration_label = "Task Duration"

    @property
    def raw_profile_field(self) -> str:
        return self.workflow_context()["measurement"]["raw_evidence_field"]

    measurement_label = "msprof 口径证据（Task Duration，唯一 kernel 时延口径）"

    def workflow_context(self) -> dict:
        return json.loads(PROFILE.read_text(encoding="utf-8"))

    def has_matrix_path(self, design: str) -> bool:
        return bool(
            re.search(
                r"^[ \t]*(?:(?:T\.gemm|(?:T\.)?(?:load_nd2nz|store_fixpipe))[ \t]*\("
                r"""|(?:with[ \t]+)?(?:T\.)?Scope\(\s*["']Cube["']\s*\))""",
                design,
                re.MULTILINE,
            )
        )

    def matrix_path_required_message(self) -> str:
        return (
            "no Cube-path call (T.gemm/load_nd2nz/store_fixpipe/"
            "Scope(Cube)) found in the design pseudocode"
        )

    def design_modes(self) -> tuple[str, str]:
        return r"[Dd]eveloper|[Ee]xpert|混合", "Developer/Expert/混合"

    def core_evidence(self) -> tuple[str, str]:
        return (
            "get_aicore_num",
            "② 物理核数依据（NPUUtils.get().get_aicore_num() 实查记录）",
        )

    def probe_device(self) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            import torch
            import torch_npu  # noqa: F401 -- registers torch.npu

            properties = torch.npu.get_device_properties(torch.npu.current_device())
        return {
            "name": properties.name,
            "cores": {
                "cube": properties.cube_core_num,
                "vector": properties.vector_core_num,
            },
        }

    def device_cores(self) -> dict:
        from tilelang.utils import NPUUtils

        count = NPUUtils.get().get_aicore_num()
        return {"cube": count, "vector": count * 2, "mixed": count}

    def precision_command(self, kernel: str, level: str) -> list[str]:
        return [sys.executable, kernel, "--level", level]

    def profiler_args(self, kernel, output, launch_count, warm_up) -> list[str]:
        return [
            PROFILER,
            "op",
            f"--kernel-name={kernel}",
            f"--output={output}",
            f"--launch-count={launch_count}",
            f"--warm-up={warm_up}",
            "--dump=off",
            f"--aic-metrics={METRICS}",
        ]

    def profile_samples(self, output) -> list[float]:
        samples = []
        pattern = os.path.join(output, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv")
        for path in sorted(glob.glob(pattern)):
            with open(path) as handle:
                for row in csv.DictReader(handle):
                    if row.get(DURATION_FIELD):
                        samples.append(float(row[DURATION_FIELD]))
        return samples

    def migration_mode(self, source_repo: str, operator: str) -> str:
        root = Path(source_repo)
        directories = ("tileops/manifest", "tests/ops", "benchmarks/ops")
        if not all((root / directory).is_dir() for directory in directories):
            return "plain"
        key = re.compile(f"^{re.escape(operator)}:", re.M)
        manifests = (root / directories[0]).glob("*.yaml")
        return (
            "harness"
            if any(key.search(_read_text(str(path)) or "") for path in manifests)
            else "plain"
        )

    def scaffold_checks(self, meta_path: str, repo_root: str):
        return meta_lint(meta_path, repo_root)

    def integration_checks(self, migration_state: dict, repo_root: str):
        return integration_lint(migration_state, repo_root)

    def toolchain_stamp(self, repo_root: str, *, active: bool = False) -> dict:
        return detect_current_stamp(repo_root, active=active)

    def kernel_checks(self, tree, path: str, prefix: str):
        failures, warnings = [], []
        backend = self.workflow_context()["backend"]
        jit_found = False
        for _dec, has_target, target_val in _jit_decorators(tree):
            jit_found = True
            if has_target and target_val != backend:
                failures.append(
                    _fail(
                        f"{prefix}-JIT",
                        path,
                        f'@tilelang.jit target="{target_val}"，必须为 target="{backend}"',
                    )
                )
            elif not has_target:
                warnings.append(
                    _warn(
                        f"{prefix}-JIT",
                        path,
                        f'@tilelang.jit 未显式声明 target="{backend}"',
                    )
                )
        if not jit_found:
            failures.append(
                _fail(f"{prefix}-JIT", path, "未找到 @tilelang.jit 装饰的 kernel 函数")
            )

        return failures, warnings

    def kernel_entry_checks(self, tree, path: str, prefix: str):
        failures, warnings = [], []
        # --- main with --level ---
        has_level = any(
            isinstance(node, ast.Constant) and node.value == "--level"
            for node in ast.walk(tree)
        )
        if not has_level:
            failures.append(
                _fail(
                    f"{prefix}-MAIN-LEVEL",
                    path,
                    "main 入口缺少 --level 参数（L0/all 分层测试入口）",
                )
            )

        # --- T.prim_func usage ---
        has_prim = any(
            (isinstance(node, ast.Attribute) and node.attr == "prim_func")
            or (isinstance(node, ast.Name) and node.id == "prim_func")
            for node in ast.walk(tree)
        )
        if not has_prim:
            warnings.append(
                _warn(
                    f"{prefix}-PRIM-FUNC",
                    path,
                    "未检测到 T.prim_func（请确认 kernel 结构）",
                )
            )
        return failures, warnings


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def _fail(rule_id: str, file: str, message: str) -> dict:
    return {"rule_id": rule_id, "file": file, "message": message}


_warn = _fail


def _jit_decorators(tree: ast.AST):
    """Yield (call_node, has_target, target_value) for tilelang.jit decorators."""
    tilelang_aliases, jit_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "tilelang":
                    tilelang_aliases.add(item.asname or "tilelang")
        elif isinstance(node, ast.ImportFrom) and node.module == "tilelang":
            for item in node.names:
                if item.name == "jit":
                    jit_names.add(item.asname or "jit")

    def is_jit(func) -> ast.expr | None:
        for dec in func.decorator_list:
            target = None
            if isinstance(dec, ast.Call):
                target = dec.func
            elif isinstance(
                dec, (ast.Attribute, ast.Name)
            ):  # bare @tilelang.jit / @jit
                target = dec
            if target is None:
                continue
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                if isinstance(target.value, ast.Name) and (
                    target.value.id in tilelang_aliases
                ):
                    return dec
            elif isinstance(target, ast.Name) and target.id in jit_names:
                return dec
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dec = is_jit(node)
            if dec is not None:
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg == "target":
                            val = getattr(kw.value, "value", None)
                            yield dec, True, val
                            break
                    else:
                        yield dec, False, None
                else:
                    yield dec, False, None


def meta_lint(meta_path: str, repo_root: str):
    """Stage 0 gate: .migration_meta.json schema + TileOPs 7-file scaffold."""
    failures, warnings = [], []
    raw = _read_text(meta_path)
    if raw is None or not raw.strip():
        failures.append(
            _fail("S0-META", meta_path, ".migration_meta.json 不存在或为空")
        )
        return failures, warnings
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S0-META", meta_path, f".migration_meta.json 非法 JSON：{exc}")
        )
        return failures, warnings

    if not isinstance(meta, dict):
        return [
            _fail("S0-META", meta_path, ".migration_meta.json 必须是 JSON 对象")
        ], warnings
    for field in ("op_slug", "family"):
        if not isinstance(meta.get(field), str) or not meta[field].strip():
            failures.append(
                _fail(
                    "S0-META-FIELD",
                    meta_path,
                    f".migration_meta.json 缺有效字段 {field}",
                )
            )
    funcs = meta.get("extracted_functions")
    if (
        not isinstance(funcs, list)
        or not funcs
        or any(not isinstance(name, str) or not name.strip() for name in funcs)
    ):
        failures.append(
            _fail("S0-META-FUNC", meta_path, "extracted_functions 必须是非空函数名数组")
        )
    if failures:
        return failures, warnings

    family = meta.get("family")
    op_slug = meta.get("op_slug")
    if family and op_slug:
        checks = [
            ("S0-SCAFFOLD", f"tileops/manifest/{family}.yaml"),
            ("S0-SCAFFOLD", f"tileops/workloads/{family}.py"),
            ("S0-SCAFFOLD", f"tileops/kernels/{family}/{op_slug}/{op_slug}.py"),
        ]
        for test_slug in [meta.get("test_slug")] if meta.get("test_slug") else []:
            checks.append(("S0-SCAFFOLD", f"tests/ops/test_{test_slug}.py"))
        for bench_slug in [meta.get("bench_slug")] if meta.get("bench_slug") else []:
            checks.append(("S0-SCAFFOLD", f"benchmarks/ops/bench_{bench_slug}.py"))
        for rule_id, rel in checks:
            abs_path = os.path.join(repo_root, "examples/TileOPs", rel)
            if not os.path.exists(abs_path):
                failures.append(
                    _fail(
                        rule_id,
                        abs_path,
                        f"TileOPs 脚手架文件缺失：examples/TileOPs/{rel}",
                    )
                )
    return failures, warnings


WRAPPER_PERF_IMPORT_RE = re.compile(r"^\s*#\s*(?:from|import)\s+.*perf_opt", re.M)


def _literal_ascend_mode(source: str) -> str | None:
    """Find the final kernel's explicit mode (legacy setdefault is accepted)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    declared, defaults = [], []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ASCEND_MODE"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            declared.append(node.value.value)
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "setdefault"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "environ"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "TILELANG_ASCEND_MODE"
        ):
            value = call.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                defaults.append(value.value)
            elif isinstance(value, ast.Name) and value.id == "ASCEND_MODE" and declared:
                defaults.append(declared[-1])
    modes = {mode.capitalize() for mode in declared + defaults}
    return modes.pop() if len(modes) == 1 and modes <= {"Developer", "Expert"} else None


def integration_lint(migration_state: dict, repo_root: str):
    """Stage 5 gate: integration package, wrapper, and single-op report.

    ``migration_state`` is the parsed .migration_state.json (needs meta_path,
    family, op_slug, functions).
    """
    failures, warnings = [], []
    family = migration_state.get("family")
    op_slug = migration_state.get("op_slug")
    meta_path = migration_state.get("meta_path")
    if not (family and op_slug):
        failures.append(
            _fail("S5-STATE", str(meta_path), ".migration_state.json 缺 family/op_slug")
        )
        return failures, warnings
    pkg_dir = os.path.join(
        repo_root,
        f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}_kernel",
    )
    if not os.path.isdir(pkg_dir):
        failures.append(_fail("S5-PKG", pkg_dir, "集成包目录不存在"))
        return failures, warnings
    for required in ("__init__.py", "integration_log.md", "integration_report.json"):
        if not os.path.exists(os.path.join(pkg_dir, required)):
            failures.append(
                _fail(
                    "S5-PKG", os.path.join(pkg_dir, required), f"集成包缺少 {required}"
                )
            )
    for func in migration_state.get("functions") or {}:
        for suffix in (".py", "_DESIGN.md"):
            path = os.path.join(pkg_dir, f"{func}{suffix}")
            if not os.path.exists(path):
                failures.append(
                    _fail("S5-FUNCS", path, f"集成包缺少函数 {func} 的 {suffix} 文件")
                )
    wrapper = os.path.join(
        repo_root, f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}.py"
    )
    wtext = _read_text(wrapper)
    source_modes = set()
    for func in migration_state.get("functions") or {}:
        kernel_path = os.path.join(pkg_dir, f"{func}.py")
        kernel_text = _read_text(kernel_path)
        if kernel_text is None:
            continue
        mode = _literal_ascend_mode(kernel_text)
        if mode is None:
            failures.append(
                _fail(
                    "S5-ASCEND-MODE",
                    kernel_path,
                    "kernel 缺少明确且一致的 Developer/Expert 模式声明",
                )
            )
        else:
            source_modes.add(mode)
    if len(source_modes) > 1:
        failures.append(
            _fail(
                "S5-ASCEND-MODE",
                pkg_dir,
                "同一 Kernel class 集成了不同编程模式，需逐调用限定模式",
            )
        )
    if wtext is None:
        failures.append(_fail("S5-WRAPPER", wrapper, "wrapper 文件不存在或不可读"))
    else:
        try:
            wrapper_tree = ast.parse(wtext)
            classes = [
                node
                for node in wrapper_tree.body
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id == "Kernel"
                    for base in node.bases
                )
            ]
            class_modes = [
                stmt.value.value
                for cls in classes
                for stmt in cls.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ascend_mode"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
            ]
            if (
                len(classes) != 1
                or len(class_modes) != 1
                or set(class_modes) != source_modes
            ):
                failures.append(
                    _fail(
                        "S5-ASCEND-MODE",
                        wrapper,
                        "wrapper 的 Kernel.ascend_mode 与集成 kernel 模式不一致",
                    )
                )
        except SyntaxError:
            failures.append(
                _fail(
                    "S5-ASCEND-MODE",
                    wrapper,
                    "wrapper Python 语法错误，无法检查编程模式",
                )
            )
        if "perf_opt" not in wtext:
            failures.append(
                _fail(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 未生成 baseline/perf_opt 双 import 切换块（未检测到 perf_opt 引用）",
                )
            )
        elif not WRAPPER_PERF_IMPORT_RE.search(wtext):
            warnings.append(
                _warn(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 含 perf_opt 引用但未检测到注释态的 perf_opt import（切换块形态请复核）",
                )
            )
    report_ref_path = os.path.join(pkg_dir, "integration_report.json")
    report_ref_text = _read_text(report_ref_path)
    if report_ref_text is None:
        return failures, warnings
    try:
        report_ref = json.loads(report_ref_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT-REF", report_ref_path, f"报告引用不是合法 JSON：{exc}")
        )
        return failures, warnings
    run_rel = report_ref.get("run_json") if isinstance(report_ref, dict) else None
    if not isinstance(run_rel, str) or not run_rel.endswith("/run.json"):
        failures.append(
            _fail(
                "S5-REPORT-REF",
                report_ref_path,
                "run_json 必须是仓库相对的报告 run.json 路径",
            )
        )
        return failures, warnings
    report_root = os.path.realpath(
        os.path.join(repo_root, "examples", "TileOPs", "reports", "tileops")
    )
    run_path = os.path.realpath(os.path.join(repo_root, run_rel))
    if (
        not run_rel.startswith("examples/TileOPs/reports/tileops/")
        or os.path.commonpath((report_root, run_path)) != report_root
    ):
        failures.append(
            _fail(
                "S5-REPORT-REF", report_ref_path, "run_json 路径不在 TileOPs 报告目录内"
            )
        )
        return failures, warnings
    run_text = _read_text(run_path)
    if run_text is None:
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 不存在或不可读"))
        return failures, warnings
    try:
        run = json.loads(run_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT", run_path, f"报告 run.json 不是合法 JSON：{exc}")
        )
        return failures, warnings
    if not isinstance(run, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 顶层必须为对象"))
        return failures, warnings

    meta_file = (
        meta_path
        if os.path.isabs(str(meta_path))
        else os.path.join(repo_root, str(meta_path))
    )
    try:
        meta = json.loads(_read_text(meta_file) or "")
    except (json.JSONDecodeError, TypeError):
        meta = None
    if not isinstance(meta, dict) or not meta.get("op_name"):
        failures.append(
            _fail("S5-REPORT-META", meta_file, "迁移元数据缺少 manifest 算子名 op_name")
        )
        return failures, warnings
    if run.get("operator") != meta["op_name"]:
        failures.append(
            _fail("S5-REPORT-OP", run_path, "报告算子与迁移元数据 op_name 不一致")
        )
    metadata = run.get("metadata")
    summary = run.get("summary")
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告缺少 metadata/summary 对象"))
        return failures, warnings
    expected_test = (
        meta.get("test_path") or f"tests/ops/test_{meta.get('test_slug')}.py"
    )
    expected_bench = (
        meta.get("bench_path") or f"benchmarks/ops/bench_{meta.get('bench_slug')}.py"
    )
    if (
        metadata.get("test_file") != expected_test
        or metadata.get("benchmark_file") != expected_bench
    ):
        failures.append(
            _fail(
                "S5-REPORT-TARGET",
                run_path,
                "报告 test/benchmark 目标与迁移元数据不一致",
            )
        )
    if metadata.get("prof_mode_requested") != "msprof":
        failures.append(_fail("S5-REPORT-PROF", run_path, "报告未请求 msprof 模式"))
    correctness_passed = summary.get("correctness_passed") is True
    benchmark_requested = summary.get("benchmark_requested") is True
    benchmark_passed = summary.get("benchmark_passed") is True
    status = run.get("status")
    correctness_tests = summary.get("correctness_tests")
    if (
        not correctness_passed
        or not isinstance(correctness_tests, int)
        or isinstance(correctness_tests, bool)
        or correctness_tests < 1
    ):
        failures.append(
            _fail("S5-REPORT-TEST", run_path, "报告缺少通过的全量正确性用例")
        )
    if not benchmark_requested:
        failures.append(_fail("S5-REPORT-BENCH", run_path, "报告未实际运行 benchmark"))
    if status not in ("passed", "partial") or (status == "passed") != benchmark_passed:
        failures.append(
            _fail("S5-REPORT-STATUS", run_path, "报告状态与 benchmark 结论不一致")
        )
    if status == "partial" and correctness_passed and benchmark_requested:
        warnings.append(
            _warn(
                "S5-REPORT-BENCH",
                run_path,
                "正确性通过，但 benchmark 未得到有效结果；仅记录，不阻断集成",
            )
        )
    for name in ("report.md", "report.html"):
        path = os.path.join(os.path.dirname(run_path), name)
        if not os.path.isfile(path):
            failures.append(_fail("S5-REPORT-FILE", path, f"缺少 {name}"))
    sources = [wrapper, os.path.join(pkg_dir, "__init__.py")]
    sources.extend(
        os.path.join(pkg_dir, f"{func}.py")
        for func in migration_state.get("functions") or {}
    )
    if all(os.path.isfile(path) for path in sources) and os.path.getmtime(
        run_path
    ) < max(os.path.getmtime(path) for path in sources):
        failures.append(
            _fail(
                "S5-REPORT-STALE",
                run_path,
                "报告早于当前 wrapper/kernel 集成文件，需重新运行",
            )
        )
    return failures, warnings


# ---------------------------------------------------------------------------
# .stage_state.json schema lint (consumed by verify/repair/every mutation)


def detect_current_stamp(repo_root: str, *, active: bool = False) -> dict:
    stamp = {"tilelang": None, "tilelang_commit": None, "cann": None, "raw": ""}
    source_root = repo_root
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            import tilelang  # noqa: PLC0415
        stamp["tilelang"] = getattr(tilelang, "__version__", None)
        if active:
            package_root = Path(tilelang.__file__).resolve().parent.parent
            source_root = package_root if (package_root / ".git").exists() else None
    except Exception:  # noqa: BLE001 -- advisory tool
        if active:
            source_root = None
    if source_root is not None:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--short=10", "HEAD"],
                cwd=source_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if head.returncode == 0:
                stamp["tilelang_commit"] = head.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    candidates = [
        Path(ascend_home) / ".." / "version.cfg"
        if ascend_home
        else Path("/../version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
    ]
    if active and ascend_home:
        home = Path(ascend_home)
        candidates = [
            home / "version.cfg",
            home / f"{os.uname().machine}-linux/ascend_toolkit_install.info",
            *candidates,
        ]
    for candidate in candidates:
        try:
            text = candidate.read_text()
        except OSError:
            continue
        match = re.search(r"Version\s*=\s*(\S+)", text, re.I if active else 0)
        if match:
            stamp["cann"] = match.group(1)
            break
    if active:
        try:
            compiler = subprocess.run(
                ["bishengir-compile", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            stamp["compiler"] = next(
                (line.strip() for line in compiler.splitlines() if line.strip()), None
            )
        except (OSError, subprocess.SubprocessError):
            stamp["compiler"] = None
    stamp["raw"] = "tilelang {} (commit {}) + CANN {}".format(
        stamp["tilelang"] or "?", stamp["tilelang_commit"] or "?", stamp["cann"] or "?"
    )
    if active:
        stamp["raw"] += " + " + (stamp["compiler"] or "unknown bishengir-compile")
    return stamp
