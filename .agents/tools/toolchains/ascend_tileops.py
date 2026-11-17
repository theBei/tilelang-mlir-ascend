"""Stage 5 integration and official timing entry for TileOPs ascend950."""

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "ascend950-profiler-locked-v1"


def manifest_has_operator(root, operator):
    key = re.compile(r"^" + re.escape(operator) + r":", re.M)
    for layout in ("src/tileops/manifest/spec", "tileops/manifest"):
        for path in (root / layout).glob("*.yaml"):
            if key.search(path.read_text(encoding="utf-8")):
                return True
    return False


def has_operator(root, operator):
    return (
        (root / "benchmarks").is_dir()
        and (root / "tests").is_dir()
        and manifest_has_operator(root, operator)
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty path")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"path is outside {root}: {value}")
    return path


def read_meta(meta_path, repo_root):
    meta = read_json((repo_root / meta_path).resolve())
    if not isinstance(meta, dict):
        raise ValueError("migration meta must be an object")
    for key in ("op_name", "op_slug", "family"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.isidentifier():
            raise ValueError(f"invalid migration field {key}")
    names = meta.get("extracted_functions")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name.isidentifier() for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("extracted_functions must contain distinct Python identifiers")
    root_value = meta.get("tileops_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("tileops_root must point to the TileOPs ascend950 checkout")
    tileops = (repo_root / root_value).resolve()
    native_root = tileops / "src/tileops/kernels/ascend"
    wrapper = project_path(native_root.resolve(), meta.get("wrapper_path"))
    if not wrapper.is_file() or wrapper.suffix != ".py":
        raise ValueError(f"Ascend wrapper does not exist: {wrapper}")
    if not (tileops / "scripts/benchmark_ascend.py").is_file():
        raise ValueError(f"TileOPs ascend950 benchmark entry missing: {tileops}")
    if not manifest_has_operator(tileops, meta["op_name"]):
        raise ValueError(f"TileOPs manifest missing {meta['op_name']}")
    return meta, tileops, wrapper


def lint_error(rule, path, error):
    return [{"rule_id": rule, "file": str(path), "message": str(error)}], []


def scaffold_checks(meta_path, repo_root):
    try:
        read_meta(meta_path, Path(repo_root).resolve())
    except (OSError, TypeError, ValueError) as exc:
        return lint_error("S0-ASCEND-SCAFFOLD", meta_path, exc)
    return [], []


def artifact_paths(meta, repo_root, wrapper):
    task = repo_root / "examples" / meta["op_slug"]
    package = wrapper.parent / f"{meta['op_slug']}_kernel"
    return task, package


def stage_files(meta, task):
    for name in meta["extracted_functions"]:
        yield task / name / f"{name}.py", f"{name}.py"
        yield task / name / "DESIGN.md", f"{name}_DESIGN.md"


def integrate(meta, repo_root, wrapper):
    task, package = artifact_paths(meta, repo_root, wrapper)
    artifacts = list(stage_files(meta, task))
    for source, _ in artifacts:
        if not source.is_file():
            raise ValueError(f"Stage 3 artifact is missing: {source}")
    package.mkdir(exist_ok=True)
    for source, name in artifacts:
        shutil.copyfile(source, package / name)
    (package / "__init__.py").touch()
    print(
        json.dumps(
            {
                "package": str(package),
                "wrapper": str(wrapper),
                "design_docs": [name for _, name in artifacts if name.endswith(".md")],
            }
        )
    )


def passed_run(run_dir):
    receipt = read_json(run_dir / "receipt.json")
    result = read_json(run_dir / "result.json")
    if (
        receipt.get("status") != "passed"
        or receipt.get("guard_exit_code") != 0
        or receipt.get("result") != result
        or result.get("status") != "passed"
        or not isinstance(result.get("passed"), int)
        or result["passed"] < 1
        or result.get("failed") != 0
        or result.get("source_changed") is not False
    ):
        raise ValueError(f"TileOPs Ascend run did not pass: {run_dir}")
    return result


def measurements(run_dir):
    result = passed_run(run_dir)
    if result["verify_only"]:
        raise ValueError(f"verification-only run has no measurements: {run_dir}")
    rows = []
    case_count = 0
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            case_count += 1
            if case["status"] != "measured":
                raise ValueError(f"TileOPs case was not measured: {run_dir}")
            for name, entry in case["implementations"].items():
                measurement = entry["measurement"]
                if measurement["protocol"] != PROTOCOL:
                    raise ValueError(
                        f"unexpected timing protocol: {measurement['protocol']}"
                    )
                rows.append(
                    {
                        "op": case["op"],
                        "case": case["case_id"],
                        "implementation": name,
                        "latency_us": measurement["latency_ms"] * 1000,
                        "samples_ms": measurement["samples_ms"],
                    }
                )
    if not rows or case_count != result["passed"]:
        raise ValueError(f"incomplete TileOPs timing rows: {run_dir}")
    return rows


def integration_checks(state, repo_root):
    root = Path(repo_root).resolve()
    meta_path = state.get("meta_path")
    try:
        meta, tileops, wrapper = read_meta(meta_path, root)
        task, package = artifact_paths(meta, root, wrapper)
        for source, name in stage_files(meta, task):
            integrated = package / name
            if not integrated.is_file():
                raise ValueError(f"missing integration artifact: {integrated}")
            if name.endswith(".md") and source.read_bytes() != integrated.read_bytes():
                raise ValueError(f"integrated design differs from Stage 1: {name}")
        imports = {
            node.module
            for node in ast.walk(ast.parse(wrapper.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        for name in meta["extracted_functions"]:
            module = f"{package.name}.{name}"
            if not any(
                path == module or path.endswith(f".{module}") for path in imports
            ):
                raise ValueError(f"Ascend wrapper does not import {module}")
        for name in ("__init__.py", "integration_log.md"):
            if not (package / name).is_file():
                raise ValueError(f"missing integration artifact: {package / name}")
        ref = read_json(package / "integration_report.json")
        run_dir = project_path(task.resolve(), ref["run_dir"])
        result = passed_run(run_dir)
        environment = read_json(run_dir / "environment.json")
        receipt = read_json(run_dir / "receipt.json")
        if (
            environment["ops"] != [meta["op_name"]]
            or result["verify_only"]
            or "--case-regex" in receipt["command"]
        ):
            raise ValueError("Stage 5 needs a formal run for the selected operator")
        hashes = read_json(run_dir / "sources.json")
        for path in (
            wrapper,
            *(package / n for _, n in stage_files(meta, task) if n.endswith(".py")),
        ):
            if (
                hashes.get(str(path.relative_to(tileops)))
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    f"benchmark did not use current integrated source: {path}"
                )
        if any(row["op"] != meta["op_name"] for row in measurements(run_dir)):
            raise ValueError("benchmark row belongs to a different operator")
    except (OSError, KeyError, SyntaxError, TypeError, ValueError) as exc:
        return lint_error("S5-ASCEND-INTEGRATION", meta_path, exc)
    return [], []


def benchmark_command(tileops, args):
    command = [
        sys.executable,
        str(tileops / "scripts/benchmark_ascend.py"),
        "--run-dir",
        str(args.run_dir.resolve()),
        "--owner",
        args.owner,
        "--op",
        args.op,
    ]
    if args.case_regex:
        command.extend(["--case-regex", args.case_regex])
    if args.implementation_factory:
        command.extend(["--implementation-factory", args.implementation_factory])
    if args.verify_only:
        command.append("--verify-only")
    else:
        command.extend(["--clock-config", str(args.clock_config.resolve())])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    bench = commands.add_parser("benchmark", help="run the upstream guarded benchmark")
    bench.add_argument("--tileops-root", type=Path, required=True)
    bench.add_argument("--run-dir", type=Path, required=True)
    bench.add_argument("--op", required=True)
    bench.add_argument("--owner", required=True)
    bench.add_argument("--case-regex")
    bench.add_argument("--implementation-factory")
    bench.add_argument("--verify-only", action="store_true")
    bench.add_argument("--clock-config", type=Path)
    summary = commands.add_parser("summary", help="read an existing formal benchmark")
    summary.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--meta", help="copy Stage 3 artifacts to the Ascend wrapper package"
    )
    args = parser.parse_args()

    try:
        if args.command == "benchmark":
            if args.op == "all":
                parser.error("select one operator with --op")
            if not args.verify_only and args.clock_config is None:
                parser.error("formal benchmark requires --clock-config")
            tileops = args.tileops_root.resolve()
            return subprocess.run(
                benchmark_command(tileops, args), cwd=tileops
            ).returncode
        if args.command == "summary":
            print(json.dumps(measurements(args.run_dir), indent=2))
            return 0
        if not args.meta:
            parser.error("specify --meta or a subcommand")
        meta, _, wrapper = read_meta(args.meta, ROOT)
        integrate(meta, ROOT, wrapper)
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
