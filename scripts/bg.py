"""Background job launcher for Windows (replaces nohup). Opens no console windows.

    .venv/Scripts/python scripts/bg.py logs/job.log -- .venv/Scripts/python scripts/encode.py --config ...
    .venv/Scripts/python scripts/bg.py --status logs/job.log
    .venv/Scripts/python scripts/bg.py --stop logs/job.log

Starts the command with CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP (never DETACHED_PROCESS,
see codeintel.common.proc), sends stdout and stderr to the log and writes the PID to <log>.pid.
Do not launch jobs with `start` or `cmd /c start`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from codeintel.common import proc


def _pid_file(log_path: Path) -> Path:
    return log_path.with_suffix(log_path.suffix + ".pid")


def launch(log_path: Path, cmd: list[str]) -> int:
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
    with open(log_path, "ab") as log:
        log.write(f"### bg.py: {subprocess.list2cmdline(cmd)}\n".encode("utf-8"))
        log.flush()
        p = proc.popen(
            cmd, background=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, env=env, start_new_session=os.name != "nt",
        )
    _pid_file(log_path).write_text(str(p.pid), encoding="utf-8")
    return p.pid


def _process(log_path: Path):
    import psutil

    pid_file = _pid_file(log_path)
    if not pid_file.exists():
        return None, "no pid file"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return None, f"pid {pid} EXITED"
        return p, f"pid {pid} RUNNING"
    except psutil.NoSuchProcess:
        return None, f"pid {pid} EXITED"


def status(log_path: Path) -> str:
    return _process(log_path)[1]


def stop(log_path: Path, timeout: float = 15.0) -> str:
    """Terminate the job and its children; kill whatever is still alive after `timeout`."""
    import psutil

    p, msg = _process(log_path)
    if p is None:
        return msg
    tree = p.children(recursive=True) + [p]
    for q in tree:
        try:
            q.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(tree, timeout=timeout)
    for q in alive:
        q.kill()
    with open(log_path, "ab") as log:
        log.write(f"### bg.py: stopped pid {p.pid} ({len(tree)} processes)\n".encode("utf-8"))
    return f"pid {p.pid} STOPPED ({len(tree)} processes, {len(alive)} force-killed)"


def main(argv: list[str]) -> None:
    if len(argv) == 2 and argv[0] in {"--status", "--stop"}:
        print((status if argv[0] == "--status" else stop)(Path(argv[1])))
        return
    if len(argv) < 3 or argv[1] != "--":
        sys.exit("usage: bg.py <logfile> -- <command...>  |  bg.py --status <logfile>  |  bg.py --stop <logfile>")
    pid = launch(Path(argv[0]), argv[2:])
    print(f"started pid {pid}, log {argv[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
