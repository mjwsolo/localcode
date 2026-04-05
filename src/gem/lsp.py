"""Code intelligence — diagnostics, imports, all offline.

Runs LOCAL tools only (ruff, pyflakes, py_compile).
No network, no language server process needed.
"""
from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Diagnostic:
    file: str
    line: int
    severity: str  # error | warning | info
    message: str

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.severity, "?")
        return f"  {icon} {self.file}:{self.line} {self.message}"


def get_diagnostics(file_path: Path) -> list[Diagnostic]:
    """Get diagnostics for a file using available tools.

    Tries in order: ruff, pyflakes, python -m py_compile
    """
    diagnostics: list[Diagnostic] = []
    rel = file_path.name

    if file_path.suffix != ".py":
        return diagnostics

    # Try ruff (fast Python linter)
    if shutil.which("ruff"):
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                for item in json.loads(result.stdout):
                    diagnostics.append(Diagnostic(
                        file=rel,
                        line=item.get("location", {}).get("row", 0),
                        severity="warning",
                        message=f"{item.get('code', '')}: {item.get('message', '')}",
                    ))
            return diagnostics
        except Exception:
            pass

    # Try pyflakes
    if shutil.which("pyflakes"):
        try:
            result = subprocess.run(
                ["pyflakes", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        lineno = 0
                    diagnostics.append(Diagnostic(
                        file=rel, line=lineno, severity="warning", message=parts[2].strip(),
                    ))
            return diagnostics
        except Exception:
            pass

    # Fallback: py_compile
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "syntax error"
            diagnostics.append(Diagnostic(file=rel, line=0, severity="error", message=err))
    except Exception:
        pass

    return diagnostics


def get_imports_for_symbol(symbol: str, file_path: Path) -> list[str]:
    """Suggest import statements for an undefined symbol."""
    common_imports = {
        "Path": "from pathlib import Path",
        "os": "import os",
        "sys": "import sys",
        "json": "import json",
        "re": "import re",
        "time": "import time",
        "datetime": "import datetime",
        "typing": "from typing import",
        "List": "from typing import List",
        "Dict": "from typing import Dict",
        "Optional": "from typing import Optional",
        "dataclass": "from dataclasses import dataclass",
        "field": "from dataclasses import field",
        "requests": "import requests",
        "numpy": "import numpy as np",
        "np": "import numpy as np",
        "pd": "import pandas as pd",
        "pandas": "import pandas as pd",
        "plt": "import matplotlib.pyplot as plt",
        "torch": "import torch",
        "nn": "from torch import nn",
        "Flask": "from flask import Flask",
        "FastAPI": "from fastapi import FastAPI",
        "pytest": "import pytest",
    }
    if symbol in common_imports:
        return [common_imports[symbol]]
    return []
