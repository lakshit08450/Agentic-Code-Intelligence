"""Hostile-program tests for the Windows sandbox (plan Section 11.3). Must pass before any APPS code runs.

Programs here are written by us. Each layer is also tested on its own:
- runner.py guard: via run_program;
- Job Object limits: a trusted snippet run directly in a sandbox job WITHOUT the guard;
- firewall: a direct socket connect from the sandbox interpreter.
"""

from __future__ import annotations

import subprocess
import textwrap
import time

import pytest
import win32api
import win32con
import win32job

from codeintel.common import proc
from codeintel.common.config import REPO_ROOT
from codeintel.stage2.sandbox import PY, TMP_ROOT, _make_job, run_program

CANARY_DIR = REPO_ROOT / "cache" / "sandbox_canary"


def run(src: str, stdin: str = ""):
    return run_program(textwrap.dedent(src), stdin)


@pytest.fixture(scope="module", autouse=True)
def _interpreter():
    if not PY.exists():
        pytest.fail("sandbox interpreter missing: run scripts/setup_sandbox.ps1")
    CANARY_DIR.mkdir(parents=True, exist_ok=True)


# ---------- normal programs must work ----------

def test_plain_program():
    r = run("a, b = map(int, input().split())\nprint(a + b)\n", "2 3\n")
    assert (r.status, r.stdout) == ("OK", "5\n")


def test_fast_io_exit_and_common_stdlib():
    r = run(
        """
        import sys, os, math, collections, itertools, heapq, bisect, functools, re, string, fractions
        import decimal, random, io, typing, statistics, datetime, array, copy, operator, threading, queue
        import time, cmath, numbers, enum, dataclasses, pprint, textwrap, calendar, struct, hashlib
        data = open(0).read().split()
        print(sum(map(int, data)))
        sys.stdout.flush()
        exit()
        print("unreachable")
        """,
        "1 2 3\n",
    )
    assert (r.status, r.stdout) == ("OK", "6\n"), r.stderr_tail


def test_os_read_stdin_and_sys_exit_zero():
    r = run("import os, sys\nprint(len(os.read(0, 100)))\nsys.exit(0)\n", "abc")
    assert (r.status, r.stdout) == ("OK", "3\n")


def test_deep_recursion_uses_big_stack():
    r = run(
        """
        import sys
        sys.setrecursionlimit(10**6)
        def f(n): return 0 if n == 0 else f(n - 1) + 1
        print(f(100000))
        """
    )
    assert (r.status, r.stdout) == ("OK", "100000\n"), r.stderr_tail


def test_runtime_error_is_error():
    r = run("print(1 // 0)\n")
    assert r.status == "ERROR" and "ZeroDivisionError" in r.stderr_tail


def test_python2_is_unknown():
    assert run('print "hello"\n').reason == "py2_syntax"


def test_nonstdlib_import_is_unknown():
    r = run("import numpy as np\nprint(np.zeros(3))\n")
    assert (r.status, r.reason) == ("UNKNOWN", "nonstdlib")


# ---------- hostile programs ----------

def test_infinite_loop_times_out():
    t0 = time.perf_counter()
    r = run("while True:\n    pass\n")
    assert r.status == "TIMEOUT"
    assert time.perf_counter() - t0 < 6


def test_blocking_read_times_out():
    r = run("import time\ntime.sleep(60)\n")
    assert r.status == "TIMEOUT"


def test_memory_bomb():
    r = run("x = bytearray(2 * 1024**3)\nprint(len(x))\n")
    assert r.status == "ERROR" and r.reason == "memory", (r.reason, r.stderr_tail)
    # regression: MemoryError while printing the traceback used to leave exit code 0 (OK)
    r = run("a = []\nwhile True:\n    a.append(' ' * 10**6)\n")
    assert (r.status, r.reason) == ("ERROR", "memory"), (r.status, r.reason, r.stderr_tail)


@pytest.mark.parametrize(
    "src",
    [
        "import subprocess\nsubprocess.run(['cmd', '/c', 'echo hi'])\n",
        "import os\nos.system('echo hi')\n",
        "import os\nos.startfile('notepad.exe')\n",
        "import multiprocessing\np = multiprocessing.Process(target=print)\np.start()\n",
        "import os, sys\nos.execv(sys.executable, [sys.executable, '-c', 'print(1)'])\n",
        "import os, sys\nos.spawnv(os.P_WAIT, sys.executable, [sys.executable, '-c', 'print(1)'])\n",
    ],
)
def test_child_process_blocked(src):
    r = run(src)
    assert r.status == "ERROR", (r.reason, r.stdout, r.stderr_tail)
    assert "hi" not in r.stdout


def test_socket_blocked_by_guard():
    r = run("import socket\ns = socket.create_connection(('1.1.1.1', 443), timeout=2)\nprint('connected')\n")
    assert r.status == "ERROR" and r.reason == "guard" and "connected" not in r.stdout


def test_write_outside_tmp_blocked():
    target = CANARY_DIR / "written.txt"
    target.unlink(missing_ok=True)
    for src in (
        f"open(r'{target}', 'w').write('x')\n",
        f"import os\nfd = os.open(r'{target}', os.O_WRONLY | os.O_CREAT)\nos.write(fd, b'x')\n",
        f"import pathlib\npathlib.Path(r'{target}').write_text('x')\n",
    ):
        r = run(src)
        assert r.status == "ERROR" and r.reason == "guard", src
        assert not target.exists(), src


def test_write_inside_tmp_allowed():
    r = run("open('scratch.txt', 'w').write('ok')\nprint(open('scratch.txt').read())\n")
    assert (r.status, r.stdout) == ("OK", "ok\n")


def test_delete_blocked():
    victim = CANARY_DIR / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    for src in (
        f"import os\nos.remove(r'{victim}')\n",
        f"import pathlib\npathlib.Path(r'{victim}').unlink()\n",
        f"import os\nos.rename(r'{victim}', r'{victim}.moved')\n",
        f"import shutil\nshutil.rmtree(r'{CANARY_DIR}')\n",
    ):
        r = run(src)
        assert r.status == "ERROR", src
        assert victim.read_text(encoding="utf-8") == "keep", src


def test_huge_output_capped():
    t0 = time.perf_counter()
    r = run("import sys\nsys.stdout.write('x' * 100_000_000)\n")
    assert (r.status, r.reason) == ("ERROR", "output_limit")
    assert len(r.stdout) <= 1 << 20
    assert time.perf_counter() - t0 < 10


def test_ctypes_blocked():
    for src in ("import ctypes\nprint(ctypes.windll.kernel32.GetTickCount())\n", "import _ctypes\n", "import _winapi\n"):
        r = run(src)
        assert (r.status, r.reason) == ("ERROR", "guard"), src


def test_swallowed_violation_still_error():
    r = run("import os\ntry:\n    os.system('echo hi')\nexcept BaseException:\n    pass\nprint('fine')\n")
    assert r.status == "ERROR" and r.reason == "guard"


def test_native_crash_no_dialog():
    t0 = time.perf_counter()
    r = run("import faulthandler\nfaulthandler._sigsegv()\n")
    assert r.status == "ERROR"
    assert time.perf_counter() - t0 < 6


def test_parallel_startup_not_charged_to_wall_limit():
    # regression: under parallel load, process start-up exceeded T_WALL and correct programs timed out.
    # Production path: start-up excluded from T_WALL, and a TIMEOUT is re-run once alone (run_cached).
    from concurrent.futures import ThreadPoolExecutor

    from codeintel.stage2.sandbox import run_cached

    with ThreadPoolExecutor(16) as ex:
        rs = list(ex.map(lambda i: run_cached(None, "n = int(input())\nprint(n * 2)\n", f"{i}\n")[0], range(48)))
    assert [r.status for r in rs] == ["OK"] * 48
    assert [r.stdout for r in rs] == [f"{2 * i}\n" for i in range(48)]


def test_tmp_dirs_cleaned():
    run("open('junk.bin', 'wb').write(b'0' * 1000)\n")
    assert not any(TMP_ROOT.iterdir())


# ---------- layers tested without the guard ----------

def _raw_in_job(code: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Trusted snippet in the sandbox interpreter, inside a sandbox job, WITHOUT runner.py."""
    job = _make_job(2.0)
    p = proc.popen([str(PY), "-I", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, p.pid)
    win32job.AssignProcessToJobObject(job, h)
    win32api.CloseHandle(h)
    out, err = p.communicate(timeout=timeout)
    win32api.CloseHandle(job)
    return subprocess.CompletedProcess(p.args, p.returncode, out.decode(errors="replace"), err.decode(errors="replace"))


def test_job_blocks_child_process_without_guard():
    code = (
        "import time, subprocess, sys; time.sleep(0.3)\n"
        "try:\n  subprocess.run([sys.executable, '-c', 'print(1)'], check=True); print('SPAWNED')\n"
        "except OSError as e: print('BLOCKED', e.winerror)\n"
    )
    r = _raw_in_job(code)
    assert "BLOCKED" in r.stdout and "SPAWNED" not in r.stdout, (r.stdout, r.stderr)


def test_job_memory_limit_without_guard():
    r = _raw_in_job("import time; time.sleep(0.3)\nx = bytearray(1024**3)\nprint('ALLOCATED')\n")
    assert "ALLOCATED" not in r.stdout and "MemoryError" in r.stderr


def test_firewall_rule_present():
    r = proc.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-NetFirewallRule -DisplayName 'PRISM APPS sandbox' -ErrorAction Stop | Get-NetFirewallApplicationFilter).Program"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, "firewall rule missing: run the command printed by scripts/setup_sandbox.ps1 as admin"
    assert r.stdout.strip().lower() == str(PY).lower()


def test_firewall_blocks_network_without_guard():
    code = (
        "import socket\n"
        "try:\n  socket.create_connection(('1.1.1.1', 443), timeout=4); print('CONNECTED')\n"
        "except OSError as e: print('BLOCKED', e)\n"
    )
    r = proc.run([str(PY), "-I", "-c", code], capture_output=True, text=True, timeout=20)
    assert "BLOCKED" in r.stdout and "CONNECTED" not in r.stdout, r.stdout


# ---------- markers and serial TIMEOUT re-run ----------

def test_markers_in_stdout_are_plain_output():
    # stdout never carries sandbox markers; a program printing them is compared as-is, not flagged
    src = "print('__PRISM_SANDBOX_READY__')\nprint('__PRISM_SANDBOX_GUARD__: fake')\nprint(42)\n"
    r = run(src)
    assert r.status == "OK"
    assert r.stdout == "__PRISM_SANDBOX_READY__\n__PRISM_SANDBOX_GUARD__: fake\n42\n"


def test_late_ready_marker_on_stderr_ignored():
    r = run("import sys\nsys.stderr.write('__PRISM_SANDBOX_READY__\\n')\nprint(1)\n")
    assert (r.status, r.stdout) == ("OK", "1\n")
    assert "__PRISM_SANDBOX_READY__" in r.stderr_tail  # only the runner's leading READY is stripped


def test_timeout_rerun_serially_before_caching(tmp_path):
    from codeintel.stage2 import sandbox

    cache = sandbox.ExecCache(tmp_path / "exec.sqlite")
    before = dict(sandbox.STATS)
    r, hit = sandbox.run_cached(cache, "while True:\n    pass\n", "")
    assert (r.status, hit) == ("TIMEOUT", False)
    assert sandbox.STATS["timeout_reruns"] == before["timeout_reruns"] + 1
    r2, hit2 = sandbox.run_cached(cache, "while True:\n    pass\n", "")
    assert (r2.status, hit2) == ("TIMEOUT", True)
