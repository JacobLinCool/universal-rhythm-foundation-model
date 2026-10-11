"""Presentation and scoring controls that are not player actions.

Scroll speed, slider velocity, gogo/kiai time, bar-line visibility and similar
switches live on their own timeline so that gameplay events stay purely about
interaction. ``kind`` is either a shared name (see :data:`SHARED_KINDS`) or a
namespaced game-specific one (``taiko:...``).
"""

from __future__ import annotations

from typing import Any

import msgspec

from urfm.ir.events import BranchRef
from urfm.ir.time import Time

SHARED_KINDS = frozenset(
    {
        "scroll",  # visual scroll-speed multiplier
        "highlight",  # gogo time / kiai time on/off
        "barline",  # bar-line visibility on/off
        "section",  # named section marker
    }
)

Value = bool | int | float | str | None


class Control(msgspec.Struct, frozen=True, kw_only=True):
    time: Time
    kind: str
    value: Value = None
    branch: BranchRef | None = None
    native: dict[str, Any] = {}
