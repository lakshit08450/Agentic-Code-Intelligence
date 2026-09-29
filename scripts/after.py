"""Wait until the process in a bg.py .pid file exits, then run a command in the foreground
(itself launched with bg.py, so the chain stays detached and window-free).

    .venv/Scripts/python scripts/bg.py logs/next.log -- .venv/Scripts/python scripts/after.py logs/prev.log.pid -- <cmd...>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import psutil

from codeintel.common import proc


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        sys.exit("usage: after.py <pidfile> -- <command...>")
    pid = int(Path(argv[0]).read_text(encoding="utf-8").strip())
    print(f"after.py: waiting for pid {pid}", flush=True)
    while psutil.pid_exists(pid):
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(30)
    print(f"after.py: pid {pid} exited at {time.strftime('%H:%M:%S')}; starting {argv[2:]}", flush=True)
    return proc.run(argv[2:]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
