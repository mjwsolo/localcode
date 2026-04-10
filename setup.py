"""Build script — compiles all .py to .so via Cython for distribution."""
import os
from pathlib import Path
from setuptools import setup, find_packages
from Cython.Build import cythonize

# Find all .py files in gem/ (excluding __init__.py and __main__.py which must stay as .py)
source_dir = Path("src/gem")
py_files = []
for f in sorted(source_dir.rglob("*.py")):
    name = f.name
    # Keep these as pure Python (needed for package/entry discovery)
    if name in ("__init__.py", "__main__.py"):
        continue
    py_files.append(str(f))

# Also handle tui subpackage
# __init__.py files must stay as .py

if py_files:
    ext_modules = cythonize(
        py_files,
        compiler_directives={
            "language_level": "3",
        },
        quiet=True,
    )
else:
    ext_modules = []

setup(
    ext_modules=ext_modules,
)
