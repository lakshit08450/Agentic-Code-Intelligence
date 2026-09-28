"""The only place the repo starts subprocesses (Windows: never open a console window).

Every call gets CREATE_NO_WINDOW on Windows. Background jobs (scripts/bg.py) additionally
get CREATE_NEW_PROCESS_GROUP. DETACHED_PROCESS is never used: a process without a console
makes each console child it starts (git, pip, workers, sandbox runs) open its own window.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
BACKGROUND = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0


def _flags(kwargs: dict[str, Any], base: int) -> dict[str, Any]:
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | base
    return kwargs


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **_flags(kwargs, NO_WINDOW))


def popen(cmd: list[str], *, background: bool = False, **kwargs: Any) -> subprocess.Popen:
    return subprocess.Popen(cmd, **_flags(kwargs, BACKGROUND if background else NO_WINDOW))
