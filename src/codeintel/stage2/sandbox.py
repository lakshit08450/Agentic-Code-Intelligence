"""Windows sandbox for untrusted APPS programs (plan Section 11.3, R4).

Layers: (1) dedicated embeddable interpreter tools/sandbox-python/python.exe, never the venv;
(2) outbound firewall block on that exe (one-time admin rule); (3) audit-hook guard in runner.py;
(4) a Job Object per run: 512 MB process memory, 1 active process, per-process user-time limit,
kill-on-close, die-on-unhandled-exception, UI restrictions; (5) per run: wall timeout, 1 MB stdout
cap, fresh cwd under cache/tmp/ deleted afterwards, minimal environment, no console window.

`run_program` returns a RunResult with status OK / ERROR / TIMEOUT / UNKNOWN. Results are cached in
cache/exec/exec.sqlite keyed by sha1(doc) | sha1(stdin) | SANDBOX_VERSION.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import win32api
import win32con
import win32job

from codeintel.common import proc
from codeintel.common.config import REPO_ROOT
from codeintel.common.hashing import sha1_text

PY = REPO_ROOT / "tools" / "sandbox-python" / "python.exe"
RUNNER = Path(__file__).resolve().with_name("runner.py")
TMP_ROOT = REPO_ROOT / "cache" / "tmp"
EXEC_DB = REPO_ROOT / "cache" / "exec" / "exec.sqlite"

T_WALL = 2.0
MEM_BYTES = 512 * 1024 * 1024
OUT_CAP = 1 << 20
ERR_KEEP = 4096
SANDBOX_VERSION = f"v1-t{T_WALL}-m{MEM_BYTES >> 20}-o{OUT_CAP}"

EXIT_SYNTAX, EXIT_NONSTDLIB, EXIT_NOJOB = 90, 91, 93
ERROR_NOT_ENOUGH_QUOTA = 1816  # exit status when the job's per-process CPU time limit kills a process
MARK = b"__PRISM_SANDBOX_GUARD__:"


@dataclass
class RunResult:
    status: str          # OK | ERROR | TIMEOUT | UNKNOWN
    reason: str          # e.g. exit_0, exception, guard, output_limit, wall, cpu, py2_syntax, nonstdlib, nojob
    stdout: str
    stderr_tail: str
    wall_s: float
    exit_code: int | None


def _make_job(t_wall: float):
    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    basic = info["BasicLimitInformation"]
    basic["LimitFlags"] = (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | win32job.JOB_OBJECT_LIMIT_PROCESS_TIME
        | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | win32job.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    )
    basic["ActiveProcessLimit"] = 1
    basic["PerProcessUserTimeLimit"] = int((t_wall + 1.0) * 10_000_000)  # 100 ns units
    info["ProcessMemoryLimit"] = MEM_BYTES
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    win32job.SetInformationJobObject(
        job, win32job.JobObjectBasicUIRestrictions, {"UIRestrictionsClass": win32job.JOB_OBJECT_UILIMIT_ALL}
    )
    return job


def _reader(stream, keep: int, cap: int, state: dict, key: str, on_cap) -> None:
    buf = bytearray()
    total = 0
    while True:
        chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if key == "err" and MARK in chunk:
            state["guard"] = True
        if len(buf) < keep:
            buf += chunk[: keep - len(buf)]
        elif key == "err":
            buf = (buf + chunk)[-keep:]
        if total > cap:
            state[f"{key}_capped"] = True
            on_cap()
            break
    state[key] = bytes(buf)
    if key == "err" and MARK in state[key]:
        state["guard"] = True


def _rmtree(path: Path) -> None:
    for _ in range(20):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.05)


def run_program(program: str, stdin_text: str, t_wall: float = T_WALL) -> RunResult:
    """Run one untrusted program on one input. Never raises for anything the program does."""
    if not PY.exists():
        raise RuntimeError(f"sandbox interpreter missing: {PY} (run scripts/setup_sandbox.ps1)")
    tmp = TMP_ROOT / uuid.uuid4().hex
    tmp.mkdir(parents=True)
    prog = tmp / "solution.py"
    prog.write_bytes(program.encode("utf-8", "surrogatepass"))
    env = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "TEMP": str(tmp), "TMP": str(tmp), "PYTHONIOENCODING": "utf-8",
    }
    job = _make_job(t_wall)
    state: dict = {}
    t0 = time.perf_counter()
    try:
        p = proc.popen(
            [str(PY), "-I", "-B", "-X", "utf8", str(RUNNER), str(prog), str(tmp)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(tmp), env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            hproc = win32api.OpenProcess(
                win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE | win32con.PROCESS_QUERY_INFORMATION,
                False, p.pid,
            )
            win32job.AssignProcessToJobObject(job, hproc)
            win32api.CloseHandle(hproc)
        except Exception:
            p.kill()  # never let it run outside the job; runner would exit EXIT_NOJOB anyway
            raise

        def kill():
            try:
                win32job.TerminateJobObject(job, 1)
            except Exception:
                pass

        readers = [
            threading.Thread(target=_reader, args=(p.stdout, OUT_CAP, OUT_CAP, state, "out", kill), daemon=True),
            threading.Thread(target=_reader, args=(p.stderr, ERR_KEEP, OUT_CAP, state, "err", kill), daemon=True),
        ]
        for r in readers:
            r.start()

        def feed():
            try:
                p.stdin.write(stdin_text.encode("utf-8"))
                p.stdin.close()
            except OSError:
                pass

        writer = threading.Thread(target=feed, daemon=True)
        writer.start()
        try:
            rc = p.wait(timeout=t_wall)
            timed_out = False
        except subprocess.TimeoutExpired:
            kill()
            rc = p.wait(timeout=5)
            timed_out = True
        for r in readers:
            r.join(timeout=5)
        wall = time.perf_counter() - t0
    finally:
        win32api.CloseHandle(job)  # KILL_ON_JOB_CLOSE: nothing survives
        _rmtree(tmp)

    out = state.get("out", b"").decode("utf-8", "replace").replace("\r\n", "\n")
    err = state.get("err", b"").decode("utf-8", "replace").replace("\r\n", "\n")[-ERR_KEEP:]
    if state.get("guard"):
        status, reason = "ERROR", "guard"
    elif state.get("out_capped") or state.get("err_capped"):
        status, reason = "ERROR", "output_limit"
    elif timed_out:
        status, reason = "TIMEOUT", "wall"
    elif rc == ERROR_NOT_ENOUGH_QUOTA:
        status, reason = "TIMEOUT", "cpu"
    elif rc == EXIT_SYNTAX:
        status, reason = "UNKNOWN", "py2_syntax"
    elif rc == EXIT_NONSTDLIB:
        status, reason = "UNKNOWN", "nonstdlib"
    elif rc == EXIT_NOJOB:
        status, reason = "UNKNOWN", "nojob"
    elif rc != 0:
        status, reason = "ERROR", "memory" if "MemoryError" in err else f"exit_{rc}"
    else:
        status, reason = "OK", "exit_0"
    return RunResult(status, reason, out, err, round(wall, 4), rc)


class ExecCache:
    """Thread-safe sqlite cache of RunResults."""

    def __init__(self, path: Path = EXEC_DB) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS runs(key TEXT PRIMARY KEY, status TEXT, reason TEXT, stdout TEXT,"
            " stderr_tail TEXT, wall_s REAL, exit_code INTEGER)"
        )
        self.lock = threading.Lock()

    @staticmethod
    def key(program: str, stdin_text: str) -> str:
        return f"{sha1_text(program)}|{sha1_text(stdin_text)}|{SANDBOX_VERSION}"

    def get(self, k: str) -> RunResult | None:
        with self.lock:
            row = self.db.execute(
                "SELECT status, reason, stdout, stderr_tail, wall_s, exit_code FROM runs WHERE key=?", (k,)
            ).fetchone()
        return RunResult(*row) if row else None

    def put(self, k: str, r: RunResult) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)", (k, *asdict(r).values()))
            self.db.commit()

    def __len__(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def run_cached(cache: ExecCache | None, program: str, stdin_text: str) -> tuple[RunResult, bool]:
    """Returns (result, was_cached)."""
    if cache is not None:
        k = cache.key(program, stdin_text)
        hit = cache.get(k)
        if hit is not None:
            return hit, True
    r = run_program(program, stdin_text)
    if cache is not None and r.reason != "nojob":
        cache.put(k, r)
    return r, False
