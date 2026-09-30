"""Hard cutoff: at HH:MM, stop a bg.py job cleanly (bg.py --stop) if it is still running.

    .venv/Scripts/python scripts/bg.py logs/cutoff.log -- .venv/Scripts/python scripts/stop_at.py 15:00 logs/job.log
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

from codeintel.common import proc

BG = Path(__file__).resolve().with_name("bg.py")


def status(log: str) -> str:
    return proc.run([sys.executable, str(BG), "--status", log], capture_output=True, text=True).stdout.strip()


def main(hhmm: str, log: str) -> None:
    h, m = map(int, hhmm.split(":"))
    now = dt.datetime.now()
    cutoff = now.replace(hour=h, minute=m, second=0, microsecond=0)
    print(f"stop_at: watching {log} until {cutoff:%Y-%m-%d %H:%M}", flush=True)
    while dt.datetime.now() < cutoff:
        if "EXITED" in status(log):
            print(f"stop_at: job finished before the cutoff ({dt.datetime.now():%H:%M}); nothing to do", flush=True)
            return
        time.sleep(30)
    s = status(log)
    if "RUNNING" in s:
        out = proc.run([sys.executable, str(BG), "--stop", log], capture_output=True, text=True).stdout.strip()
        print(f"stop_at: cutoff reached, {out}", flush=True)
    else:
        print(f"stop_at: cutoff reached, job already {s}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
