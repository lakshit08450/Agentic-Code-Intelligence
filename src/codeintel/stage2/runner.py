"""Trusted runner, executed by the sandbox interpreter only (plan Section 11.3, layer 3):

    tools/sandbox-python/python.exe -I -B -X utf8 runner.py <program.py> <tmpdir>

Standard library only (the embeddable package has nothing else). Order of operations:
1. Before any untrusted code: disable Windows error dialogs and wait (max 1 s) until this process
   sits in a Job Object carrying the sandbox limits (1 active process, 512 MB); exit EXIT_NOJOB otherwise.
2. Drop ctypes from sys.modules, install the audit-hook guard (cannot be removed once installed).
3. Compile the program (SyntaxError -> EXIT_SYNTAX, usually Python 2) and exec it as __main__ in a
   thread with a large stack (Linux judges give ~8 MB+ of stack; Windows' main thread has 1 MB).

A guard violation writes MARK to stderr through the raw fd (so user code cannot swallow it by
catching the exception) and raises PermissionError. Audit hooks are defense in depth, not a
security boundary; layers 1, 2 and 4 are the boundary.
"""

import builtins
import os
import sys
import threading
import time

EXIT_SYNTAX = 90
EXIT_NONSTDLIB = 91
EXIT_NOJOB = 93
MARK = b"\n__PRISM_SANDBOX_GUARD__:"
MEM_LIMIT = 512 * 1024 * 1024
STACK_BYTES = 128 * 1024 * 1024  # Windows rejects >= 256 MB


def wait_for_job(timeout: float = 1.0) -> bool:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetErrorMode(0x0001 | 0x0002 | 0x8000)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rt", "wt", "ot")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
    ]
    info = EXTENDED()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # NULL handle = the job this process belongs to (innermost); fails if in no job
        ok = k32.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        if ok and info.Basic.ActiveProcessLimit == 1 and info.ProcessMemoryLimit == MEM_LIMIT:
            return True
        time.sleep(0.002)
    return False


def install_guard(tmpdir: str) -> None:
    tmp = os.path.normcase(os.path.abspath(tmpdir)) + os.sep
    exact = {
        "subprocess.Popen", "os.system", "os.startfile", "os.posix_spawn", "os.remove", "os.unlink",
        "os.rmdir", "os.rename", "shutil.rmtree",
    }
    prefixes = ("socket.", "ctypes.", "winreg.", "os.exec", "os.spawn", "_winapi.")
    banned_imports = {"ctypes", "_ctypes", "_winapi"}
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    write_fd = os.write

    def violation(event: str, detail: str = "") -> None:
        write_fd(2, MARK + f"{event} {detail}".encode("utf-8", "replace")[:200] + b"\n")
        raise PermissionError(f"sandbox: {event} blocked")

    def is_write(mode, flags) -> bool:
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return True
        return isinstance(flags, int) and bool(flags & write_flags)

    def hook(event, args):
        if event in exact or event.startswith(prefixes):
            violation(event)
        elif event == "import":
            if args and str(args[0]).split(".")[0] in banned_imports:
                violation(event, str(args[0]))
        elif event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if not is_write(mode, flags):
                return
            if isinstance(path, int):
                if path not in (1, 2):
                    violation(event, f"fd {path}")
                return
            try:
                p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
            except Exception:
                violation(event, "unresolvable path")
            if not p.startswith(tmp):
                violation(event, p)

    sys.addaudithook(hook)


def run_program(code, stdout) -> int:
    """exec in a big-stack thread; returns the process exit code."""
    result = {"code": 0}

    def target():
        try:
            exec(code, {"__name__": "__main__", "__builtins__": builtins})
        except SystemExit as e:
            c = e.code
            if c is None or isinstance(c, int):
                result["code"] = c or 0
            else:
                sys.stderr.write(f"{c}\n")
                result["code"] = 1
        except ModuleNotFoundError as e:
            top = (e.name or "").split(".")[0]
            result["code"] = EXIT_NONSTDLIB if top and top not in sys.stdlib_module_names else 1
            import traceback
            traceback.print_exc()
        except BaseException:
            import traceback
            traceback.print_exc()
            result["code"] = 1

    threading.stack_size(STACK_BYTES)
    t = threading.Thread(target=target, name="solution")
    t.start()
    t.join()
    return result["code"]


def _exit(code=None):
    raise SystemExit(code)


def main() -> None:
    prog_path, tmpdir = sys.argv[1], sys.argv[2]
    with open(prog_path, "rb") as f:
        source = f.read()
    if not wait_for_job():
        os._exit(EXIT_NOJOB)
    # already-loaded modules never fire the "import" audit event, so unload the banned ones first
    for name in [m for m in sys.modules if m in ("ctypes", "_winapi") or m.startswith(("ctypes.", "_ctypes"))]:
        del sys.modules[name]
    builtins.exit = builtins.quit = _exit  # normally added by `site`, which -I/._pth skip
    sys.argv = [prog_path]
    install_guard(tmpdir)
    try:
        code = compile(source, "solution.py", "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        os._exit(EXIT_SYNTAX)
    rc = run_program(code, sys.stdout)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(rc)


if __name__ == "__main__":
    main()
