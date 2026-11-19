"""Read the task selection already used by TileOPs reporting."""
import argparse
import os
import sys
from pathlib import Path


def get_task_platform():
    task_dir = os.environ.get("TILEOPS_TASK_DIR")
    tools = Path(__file__).resolve().parents[4] / ".agents/tools"
    sys.path.insert(0, str(tools))
    from platform_context import command_platform, get_platform

    if not task_dir:
        return get_platform()

    return command_platform(
        argparse.Namespace(task_dir=task_dir, hardware=None, toolchain=None)
    )


def shared_memory_budget(device=None):
    context = get_task_platform()
    if context.hardware is not None:
        return context.design_budget()["capacity_bytes"]["UB"]
    return context.adapter.shared_memory_budget(device)
