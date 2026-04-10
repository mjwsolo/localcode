"""Build script — compiles all .py to .so via Cython for distribution."""
import os
from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize

# Find all .py files in gem/ (excluding __init__.py and __main__.py which must stay as .py)
source_dir = Path("src/gem")
py_files = []
for f in sorted(source_dir.rglob("*.py")):
    name = f.name
    if name in ("__init__.py", "__main__.py"):
        continue
    py_files.append(str(f))

if py_files:
    ext_modules = cythonize(
        py_files,
        compiler_directives={"language_level": "3"},
        quiet=True,
    )
else:
    ext_modules = []


class StripSourceBuildWheel:
    """Remove .py and .c source files from the wheel, keeping only .so"""
    pass


setup(
    ext_modules=ext_modules,
    options={
        "bdist_wheel": {},
    },
    exclude_package_data={
        "gem": ["*.c"],
    },
)
