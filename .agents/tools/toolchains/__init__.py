"""Toolchain implementations used by the agent workflow."""

from .npuir import NpuirToolchain

TOOLCHAINS = {"npuir": NpuirToolchain}
