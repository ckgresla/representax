"""Run Representax's fast, import-free static-analysis gate."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run every static check and report all failures in one invocation."""

    github = os.environ.get("GITHUB_ACTIONS") == "true"
    commands = (
        ("ruff", "format", "--check", "."),
        (
            "ruff",
            "check",
            *(("--output-format=github",) if github else ()),
            ".",
        ),
        (
            "ty",
            "check",
            "--python",
            sys.executable,
            *(("--output-format=github",) if github else ()),
        ),
    )
    failed = False
    for command in commands:
        completed = subprocess.run(command, check=False)
        failed = completed.returncode != 0 or failed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
