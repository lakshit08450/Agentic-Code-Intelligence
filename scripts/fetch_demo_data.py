"""One-line demo data install: download (or copy) prism_demo_data.zip and extract demo_data/ into the repo.

    .venv\\Scripts\\python scripts\\fetch_demo_data.py <https-url-or-local-path-to-prism_demo_data.zip>
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(src: str) -> None:
    tmp = Path(tempfile.mkdtemp())
    z = tmp / "prism_demo_data.zip"
    if src.startswith(("http://", "https://")):
        print(f"downloading {src} ...", flush=True)
        urllib.request.urlretrieve(src, z)
    else:
        shutil.copy(src, z)
    target = ROOT / "demo_data"
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(z) as zf:
        bad = [n for n in zf.namelist() if not n.startswith("demo_data/") or ".." in n]
        if bad:
            sys.exit(f"refusing to extract unexpected paths: {bad[:3]}")
        zf.extractall(ROOT)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"demo data ready in {target} ({sum(1 for _ in target.rglob('*'))} files)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
