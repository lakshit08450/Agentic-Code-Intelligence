"""Detached background job launcher for Windows (replaces nohup).

    .venv/Scripts/python scripts/bg.py logs/job.log -- .venv/Scripts/python scripts/encode.py --config ...
    .venv/Scripts/python scripts/bg.py --status logs/job.log

Starts the command with DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, sends stdout and
stderr to the log file, and writes the PID to <log>.pid. The job survives the parent shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def launch(log_path: Path, cmd: list[str]) -> int:
    # CreateProcess does not apply PATHEXT or resolve forward-slash relative paths reliably
    # (Python 3.11's shutil.which ignores PATHEXT for paths containing a directory)
    exe = shutil.which(cmd[0])
    for cand in (Path(cmd[0]), Path(cmd[0] + ".exe")):
        if exe is None and cand.is_file():
            exe = str(cand.resolve())
    if exe is None:
        sys.exit(f"bg.py: executable not found: {cmd[0]}")
    cmd = [exe, *cmd[1:]]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with open(log_path, "ab") as log:
        log.write(f"### bg.py: {subprocess.list2cmdline(cmd)}\n".encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, close_fds=True, env=env, start_new_session=os.name != "nt",
        )
    log_path.with_suffix(log_path.suffix + ".pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def status(log_path: Path) -> str:
    import psutil

    pid_file = log_path.with_suffix(log_path.suffix + ".pid")
    if not pid_file.exists():
        return "no pid file"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    return f"pid {pid} {'RUNNING' if alive else 'EXITED'}"


def main(argv: list[str]) -> None:
    if len(argv) == 2 and argv[0] == "--status":
        print(status(Path(argv[1])))
        return
    if len(argv) < 3 or argv[1] != "--":
        sys.exit("usage: bg.py <logfile> -- <command...>   |   bg.py --status <logfile>")
    pid = launch(Path(argv[0]), argv[2:])
    print(f"started pid {pid}, log {argv[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
