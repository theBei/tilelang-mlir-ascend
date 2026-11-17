"""Measure one standalone workload with the TileOPs Ascend profiler protocol."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PENDING_MEASUREMENT = ".measurement.json"


def profile(args):
    from benchmarks.ascend.clocks import ClockPolicy
    from benchmarks.ascend.guard import require_reservation
    from benchmarks.ascend.timing import ProfilerTimer

    require_reservation()
    spec = importlib.util.spec_from_file_location("candidate", args.kernel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn, inputs = module.profile_workload(**json.loads(args.workload))
    with ClockPolicy(args.clock_config) as clocks:
        measurement = ProfilerTimer(args.run_dir / "profiles", clocks).measure(
            fn, inputs
        )
    measurement["duration_us"] = measurement["latency_ms"] * 1000
    (args.run_dir / PENDING_MEASUREMENT).write_text(
        json.dumps(measurement, indent=2) + "\n", encoding="utf-8"
    )


def guarded_profile(args):
    tileops = args.tileops_root.resolve()
    guard = Path(os.environ.get("TILEOPS_NPU_GUARD", tileops / "scripts/npu_guard.py"))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tileops / "src"), str(tileops)))
    child = [
        sys.executable,
        str(guard.resolve(strict=True)),
        "run",
        "--owner",
        args.owner,
        "--wait",
        "600",
        "--timeout",
        "1800",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--tileops-root",
        str(tileops),
        "--kernel",
        str(args.kernel),
        "--workload",
        args.workload,
        "--clock-config",
        str(args.clock_config),
        "--run-dir",
        str(args.run_dir),
    ]
    result = subprocess.run(child, cwd=tileops, env=env)
    if result.returncode != 0:
        return result.returncode

    pending = args.run_dir / PENDING_MEASUREMENT
    measurement = json.loads(pending.read_text(encoding="utf-8"))
    measurement["guard_exit_code"] = result.returncode
    pending.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
    pending.replace(args.run_dir / "measurement.json")
    print(
        json.dumps(
            {
                "duration_us": measurement["duration_us"],
                "ascend_profile_path": str(args.run_dir),
            }
        )
    )
    return 0


def main():
    mode = argparse.ArgumentParser(add_help=False)
    mode.add_argument("command", nargs="?", choices=("worker",))
    parser = argparse.ArgumentParser(description=__doc__, parents=[mode])
    parser.add_argument("--tileops-root", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--workload", required=True, help="JSON keyword args for profile_workload"
    )
    parser.add_argument("--clock-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner")
    args = parser.parse_args()
    if not isinstance(json.loads(args.workload), dict):
        parser.error("--workload must be a JSON object")
    args.kernel = args.kernel.resolve(strict=True)
    args.clock_config = args.clock_config.resolve(strict=True)
    args.run_dir = args.run_dir.resolve()
    if args.command == "worker":
        profile(args)
        return 0
    if not args.owner:
        parser.error("--owner is required")
    return guarded_profile(args)


if __name__ == "__main__":
    sys.exit(main())
