"""Trusted runner, executed by the sandbox interpreter only (plan Section 11.3, layer 3):

    tools/sandbox-python/python.exe -I -B -X utf8 runner.py <program.py> <tmpdir> [--call]

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
READY = b"__PRISM_SANDBOX_READY__\n"
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
    result = {"code": 1}  # anything but a clean finish is a failure

    def report() -> None:
        # best effort: printing itself can fail (e.g. under MemoryError)
        try:
            import traceback
            traceback.print_exc()
        except BaseException:
            pass

    def target():
        try:
            exec(code, {"__name__": "__main__", "__builtins__": builtins})
            result["code"] = 0
        except SystemExit as e:
            c = e.code
            result["code"] = (c or 0) if c is None or isinstance(c, int) else 1
            if result["code"] == 1 and not isinstance(c, int):
                try:
                    sys.stderr.write(f"{c}\n")
                except BaseException:
                    pass
        except MemoryError:
            result["code"] = 1
            os.write(2, b"MemoryError\n")
        except ModuleNotFoundError as e:
            top = (e.name or "").split(".")[0]
            result["code"] = EXIT_NONSTDLIB if top and top not in sys.stdlib_module_names else 1
            report()
        except BaseException:
            result["code"] = 1
            report()

    threading.stack_size(STACK_BYTES)
    t = threading.Thread(target=target, name="solution")
    t.start()
    t.join()
    return result["code"]


# ---------------- call-based mode (`--call`): plan step 5 harness ----------------
# The spec (JSON) is read from stdin by this trusted runner BEFORE any untrusted code runs:
# {"nonce": str, "func": str | null, "nargs": int, "calls": [repr(args tuple), ...]}.
# The program is executed as module "solution" (not __main__), the target callable is found,
# and each call's result is written to fd 1 as one line: nonce + JSON. The program's own prints
# during calls go to a discarded buffer.

UNSUPPORTED_TYPES = ("TreeNode", "ListNode", "Node", "NestedInteger", "Employee", "MountainArray")


def norm_value(x, depth: int = 0):
    """JSON-safe canonical form used on both sides of the comparison."""
    if depth > 50:
        return {"__repr__": "deep"}
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else {"__float__": repr(x)}
    if isinstance(x, (list, tuple)):
        return [norm_value(v, depth + 1) for v in x]
    if isinstance(x, (set, frozenset)):
        items = [norm_value(v, depth + 1) for v in x]
        try:
            return {"__set__": sorted(items, key=repr)}
        except Exception:
            return {"__set__": items}
    if isinstance(x, dict):
        return {"__dict__": sorted(([repr(k), norm_value(v, depth + 1)] for k, v in x.items()), key=lambda kv: kv[0])}
    return {"__repr__": repr(x)[:200]}


def _fuzzy(name: str) -> str:
    return name.replace("_", "").lower()


def _nparams(fn) -> int | None:
    import inspect

    try:
        ps = [p for p in inspect.signature(fn).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(ps)
    except (TypeError, ValueError):
        return None


def find_callable(g: dict, func, nargs: int):
    """Returns (callable | None, reason)."""
    import types

    sol = g.get("Solution")
    if isinstance(sol, type):
        try:
            inst = sol()
        except Exception:
            return None, "init_error"
        names = [n for n in dir(inst) if not n.startswith("_") and callable(getattr(inst, n, None))]
        if func:
            hit = [n for n in names if n == func] or [n for n in names if _fuzzy(n) == _fuzzy(func)]
            if hit:
                return getattr(inst, hit[0]), "ok"
        if len(names) == 1:
            return getattr(inst, names[0]), "ok"
        by_n = [n for n in names if _nparams(getattr(inst, n)) == nargs]
        return (getattr(inst, by_n[-1]), "ok") if len(by_n) >= 1 else (None, "no_function")
    fns = [(k, v) for k, v in g.items() if isinstance(v, types.FunctionType) and v.__module__ == "solution"]
    if func:
        hit = [v for k, v in fns if k == func] or [v for k, v in fns if _fuzzy(k) == _fuzzy(func)]
        return (hit[0], "ok") if hit else (None, "no_function")
    public = [(k, v) for k, v in fns if not k.startswith("_")]
    if len(public) == 1:
        return public[0][1], "ok"
    by_n = [v for k, v in public if _nparams(v) == nargs]
    return (by_n[-1], "ok") if by_n else (None, "no_function")


def run_calls(code, spec: dict) -> int:
    import ast
    import copy
    import io
    import json

    nonce = spec["nonce"].encode()
    result = {"code": 1}

    def emit(obj) -> None:
        os.write(1, nonce + json.dumps(obj).encode("utf-8") + b"\n")

    def target():
        g = {"__name__": "solution", "__builtins__": builtins}
        try:
            exec(code, g)
        except MemoryError:
            os.write(2, b"MemoryError\n")
            return
        except ModuleNotFoundError as e:
            top = (e.name or "").split(".")[0]
            result["code"] = EXIT_NONSTDLIB if top and top not in sys.stdlib_module_names else 1
            return
        except BaseException:
            return
        fn, why = find_callable(g, spec.get("func"), spec.get("nargs", 0))
        if fn is None:
            emit({"status": why})
            result["code"] = 0
            return
        ann = str(getattr(fn, "__annotations__", ""))
        if any(t in ann for t in UNSUPPORTED_TYPES):
            emit({"status": "unsupported"})
            result["code"] = 0
            return
        for c in spec["calls"]:
            args = ast.literal_eval(c)
            sink, old = io.StringIO(), sys.stdout
            sys.stdout = sink
            try:
                out = {"status": "ok", "value": norm_value(fn(*copy.deepcopy(args)))}
            except MemoryError:
                out = {"status": "error", "error": "MemoryError"}
            except RecursionError:
                out = {"status": "error", "error": "RecursionError"}
            except BaseException as e:  # includes SystemExit from the solution
                out = {"status": "error", "error": type(e).__name__}
            finally:
                sys.stdout = old
            emit(out)
        result["code"] = 0

    threading.stack_size(STACK_BYTES)
    t = threading.Thread(target=target, name="solution")
    t.start()
    t.join()
    return result["code"]


def _exit(code=None):
    raise SystemExit(code)


def main() -> None:
    prog_path, tmpdir = sys.argv[1], sys.argv[2]
    call_mode = len(sys.argv) > 3 and sys.argv[3] == "--call"
    with open(prog_path, "rb") as f:
        source = f.read()
    spec = None
    if call_mode:  # the call spec arrives on stdin; read it before any untrusted code runs
        import json

        spec = json.loads(sys.stdin.buffer.read().decode("utf-8"))
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
    os.write(2, READY)  # parent starts the T_WALL clock here (start-up time excluded)
    rc = run_calls(code, spec) if call_mode else run_program(code, sys.stdout)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(rc)


if __name__ == "__main__":
    main()
