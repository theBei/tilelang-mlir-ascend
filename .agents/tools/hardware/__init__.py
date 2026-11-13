"""Import compatibility for experiment scripts copied before the split."""

from pathlib import Path

from platform_context import PlatformError, get_platform, read_json, task_state_path


def get_adapter():
    state_path = task_state_path(Path.cwd())
    if state_path is not None and read_json(state_path).get("hardware_id"):
        raise PlatformError(
            "use the current experiment template for tasks with an explicit platform"
        )
    return get_platform().adapter
