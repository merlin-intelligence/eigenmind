"""Helpers for deprecating the pre-1.8 API (removal planned for 2.0)."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

REMOVAL_VERSION = "2.0"


def warn_deprecated(name: str, replacement: str, stacklevel: int = 3) -> None:
    """Emit the standard DeprecationWarning for a pre-1.8 name."""
    warnings.warn(
        f"{name} is deprecated since eigenmind 1.8 and will be removed in "
        f"{REMOVAL_VERSION}; {replacement}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(name: str, replacement: str) -> Callable[[F], F]:
    """Wrap ``fn`` so that each call emits a DeprecationWarning."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(name, replacement)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
