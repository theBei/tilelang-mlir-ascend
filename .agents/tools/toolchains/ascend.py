"""TileLang ascend backend and the locked ARM64 compiler/runtime environment."""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import ascend_profile, ascend_tileops
from .adapter import ToolchainAdapter

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / ".agents/platforms/toolchains/ascend/profile.json"


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class AscendToolchain(ToolchainAdapter):
    profiler = "torch_npu.profiler"
    raw_profile_field = "ascend_profile_path"
    measurement_label = "torch_npu.profiler 设备执行区间并集的算术均值"
    duration_label = "mean_device_busy_union_us"

    def workflow_context(self):
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        profile["environment_lock_revision"] = file_digest(
            ROOT / profile["environment_lock"]
        )
        return profile

    def has_matrix_path(self, design):
        return bool(re.search(r"^[ \t]*T\.gemm\s*\(", design, re.M))

    def matrix_path_required_message(self):
        return "no Ascend Cube GEMM call found in the design pseudocode"

    def design_modes(self):
        return r"SIMD|SIMT|Cube|混合", "SIMD/SIMT/Cube/混合"

    def core_evidence(self):
        return "get_device_properties", "② torch.npu.get_device_properties 实查记录"

    def probe_device(self):
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

    def device_cores(self):
        return self.probe_device()["cores"]

    def kernel_checks(self, tree, path, prefix):
        nodes = list(ast.walk(tree))
        ascend_namespace = any(
            (
                isinstance(node, ast.Import)
                and any(n.name == "tilelang.ascend.language" for n in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "tilelang.ascend"
                and any(n.name == "language" for n in node.names)
            )
            for node in nodes
        )
        entries = [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tilelang"
            and node.func.attr in {"compile", "jit"}
        ]
        failures = []
        if not ascend_namespace:
            failures.append(
                {
                    "rule_id": f"{prefix}-ASCEND-LANGUAGE",
                    "file": path,
                    "message": "使用 tilelang.ascend.language 声明 Ascend kernel",
                }
            )
        if not entries or any(
            not any(
                kw.arg == "target"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "ascend"
                for kw in entry.keywords
            )
            for entry in entries
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-JIT",
                    "file": path,
                    "message": 'tilelang.compile 或 tilelang.jit 必须使用 target="ascend"',
                }
            )
        return failures, []

    def kernel_entry_checks(self, tree, path, prefix):
        nodes = list(ast.walk(tree))
        failures = []
        if not any(
            isinstance(node, ast.Constant) and node.value == "--level" for node in nodes
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-MAIN-LEVEL",
                    "file": path,
                    "message": "main 入口缺少 --level 参数",
                }
            )
        if not any(
            isinstance(node, ast.Attribute) and node.attr == "prim_func"
            for node in nodes
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-PRIM-FUNC",
                    "file": path,
                    "message": "缺少 Ascend T.prim_func",
                }
            )
        return failures, []

    def precision_command(self, kernel, level):
        return [sys.executable, kernel, "--level", level]

    def profiler_args(self, kernel, output, launch_count, warm_up):
        raise RuntimeError(
            "ascend measurements use the guarded Ascend runner in the platform rules"
        )

    def profile_samples(self, output):
        output = Path(output)
        standalone = output / "measurement.json"
        if standalone.is_file():
            measurement = ascend_tileops.read_json(standalone)
            if measurement.get("guard_exit_code") != 0:
                raise RuntimeError(
                    "standalone measurement has no successful guard completion"
                )
            if measurement["protocol"] != ascend_tileops.PROTOCOL:
                raise RuntimeError("unexpected standalone timing protocol")
            return [sample * 1000 for sample in measurement["samples_ms"]]
        if (output / ascend_profile.PENDING_MEASUREMENT).is_file():
            raise RuntimeError(
                "standalone measurement was not published after guard completion"
            )
        rows = ascend_tileops.measurements(output)
        if len(rows) != 1:
            raise RuntimeError(
                "select one case and one implementation for a sample series"
            )
        return [sample * 1000 for sample in rows[0]["samples_ms"]]

    def toolchain_stamp(self, repo_root, *, active=False):
        stamp, _ = self._environment(repo_root)
        return stamp

    def check_environment(self, repo_root):
        stamp, errors = self._environment(repo_root)
        if errors:
            raise RuntimeError("Ascend 950 toolchain mismatch: " + " | ".join(errors))
        return stamp

    def _environment(self, repo_root):
        profile = self.workflow_context()
        lock_path = Path(repo_root) / profile["environment_lock"]
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        errors = []
        cxx = os.environ.get("CXX", "g++")
        try:
            compiler = subprocess.check_output(
                [cxx, "-dumpfullversion"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            compiler = None
        for name, actual, expected in (
            ("machine", platform.machine(), lock["machine"]),
            ("python", platform.python_version(), lock["python"]),
            ("CXX", compiler, lock["cxx_version"]),
            (
                "ASCEND_NPU_ARCH",
                os.environ.get("ASCEND_NPU_ARCH"),
                profile["architecture"],
            ),
        ):
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")

        packages = {}
        for name, expected in lock["packages"].items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                actual = None
            packages[name] = actual
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")

        files = {}
        for item in lock["files"]:
            name, package = item["name"], item.get("package")
            try:
                if package:
                    path = Path(
                        metadata.distribution(package).locate_file(item["path"])
                    )
                else:
                    env_key = item.get("root_env", "ASCEND_HOME_PATH")
                    root = os.environ.get(env_key)
                    if not root:
                        errors.append(f"{name}: {env_key} is unset")
                        continue
                    path = Path(root) / item["path"]
                files[name] = file_digest(path)
                if files[name] != item["sha256"]:
                    errors.append(f"{name}: fingerprint differs from {lock['profile']}")
            except (OSError, metadata.PackageNotFoundError) as exc:
                errors.append(f"{name}: {exc}")
        evidence = {
            "lock": file_digest(lock_path),
            "packages": packages,
            "files": files,
            "compiler": compiler,
        }
        return {
            "raw": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest(),
            "tilelang": packages["tilelang"],
            "tilelang_commit": None,
            "cann": lock["compiler_sdk"] if not errors else None,
            "runtime_opp": lock["runtime_opp"] if not errors else None,
            "compiler": compiler,
        }, errors

    def migration_mode(self, source_repo, operator):
        return (
            "harness"
            if ascend_tileops.has_operator(Path(source_repo), operator)
            else "plain"
        )

    def scaffold_checks(self, meta_path, repo_root):
        return ascend_tileops.scaffold_checks(meta_path, repo_root)

    def integration_checks(self, migration_state, repo_root):
        return ascend_tileops.integration_checks(migration_state, repo_root)
