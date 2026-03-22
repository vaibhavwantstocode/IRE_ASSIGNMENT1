#!/usr/bin/env python
"""
Project entrypoint: run CLI tools without installing the package (adds `src/` to PYTHONPATH).

Usage:
  python main.py query -x 4 -y 1 -z 1 -q T -optim 0 --query "machine learning"
  python main.py build -x 4 -y 1 -z 1 -optim 0 --limit 100
  python main.py evaluate -x 4 -y 1 -z 1 -optim 0
"""

from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_COMMANDS = {
    "build": "build.py",
    "query": "query.py",
    "evaluate": "evaluate.py",
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <build|query|evaluate> [arguments passed to that script]")
        print("Examples:")
        print('  python main.py query -x 4 -y 1 -z 1 -q T -optim 0 --query "test"')
        print("  python main.py build -x 4 -y 1 -z 1 -optim 0 --limit 50")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd!r}. Choose from: {', '.join(_COMMANDS)}")
        sys.exit(1)
    script = os.path.join(_ROOT, "cli", _COMMANDS[cmd])
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
