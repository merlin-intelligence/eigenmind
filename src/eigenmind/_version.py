"""Installed package version, shared by the package and the result payloads."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eigenmind")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"
