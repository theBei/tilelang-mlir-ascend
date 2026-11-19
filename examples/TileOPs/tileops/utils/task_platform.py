"""Read the selected Agent task when TileOPs runs inside a task."""

import argparse
import os
import sys
from pathlib import Path


def get_task_platform():
    task_dir = os.environ.get("TILEOPS_TASK_DIR")
    tools = Path(__file__).resolve().parents[4] / ".agents/tools"
    if not task_dir:
        context_module = sys.modules.get("platform_context")
        return context_module.ACTIVE.get() if context_module else None
    sys.path.insert(0, str(tools))
    from platform_context import command_platform

    return command_platform(argparse.Namespace(task_dir=task_dir, hardware=None, toolchain=None))


def shared_memory_budget(device=None):
    context = get_task_platform()
    if context is None:
        from tileops.device import get_device_backend

        return get_device_backend().shared_memory_budget(device)
    if context.hardware is not None:
        budget = context.design_budget()
        return budget["capacity_bytes"][budget["inflation_level"]]
    return context.adapter.shared_memory_budget(device)
