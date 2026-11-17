"""Toolchain implementations used by the agent workflow."""

from .ascend import AscendToolchain
from .npuir import NpuirToolchain

TOOLCHAINS = {"npuir": NpuirToolchain, "ascend": AscendToolchain}
