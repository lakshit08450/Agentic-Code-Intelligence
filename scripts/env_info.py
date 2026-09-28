"""Write results/env.json: OS, CPU, cores, RAM, GPU, torch/CUDA and key package versions."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        return platform.processor()


def main() -> None:
    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "docker": False,
        "sandbox": "native Windows (plan Section 11.3)",
        "packages": {p: version(p) for p in ["mteb", "sentence-transformers", "transformers", "datasets", "numpy"]},
    }
    out = ROOT / "results" / "env.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
