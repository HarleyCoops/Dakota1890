"""Dakota grammar RL environment package.

``load_environment`` stays lazy so train-reward, split, and judge helpers can
be imported without the optional ``verifiers`` extra.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load_environment"]


def load_environment(*args: Any, **kwargs: Any) -> Any:
    from .environment import load_environment as _load_environment

    return _load_environment(*args, **kwargs)
