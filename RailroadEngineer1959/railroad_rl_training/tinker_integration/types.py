from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RailroadExample:
    example_id: str
    prompt: str
    answer: str
    info: Dict[str, Any]

