"""Build script. Cython compilation is OPT-IN, not the default.

Cython used to run by default on `pip install -e .`, producing 92 MB of
.c files plus ~35 .so files that shadowed .py edits. That caused two
real problems: (1) every local .py fix stayed invisible until the user
knew to purge .so files manually, and (2) VSCode's Python indexer
burned a full CPU core indexing the generated .c files.

Now: no Cython unless the builder explicitly opts in via env var
`LOCALCODE_WITH_CYTHON=1`. Wheels for distribution can set that.
Editable dev installs run the source directly — faster to install,
edits take effect immediately, no hidden shadowing.
"""
import os
from pathlib import Path
from setuptools import setup

ext_modules = []

if os.environ.get("LOCALCODE_WITH_CYTHON"):
    try:
        from Cython.Build import cythonize
        source_dir = Path("src/localcode")
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
    except ImportError:
        pass


from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    """Include binary and data files that setuptools normally skips."""
    def build_package_data(self):
        super().build_package_data()
        # Copy llama-server binary
        src = Path("src/localcode/bin/llama-server")
        if src.exists():
            dst_dir = Path(self.build_lib) / "localcode" / "bin"
            dst_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(src), str(dst_dir / "llama-server"))

setup(
    ext_modules=ext_modules,
    cmdclass={"build_py": CustomBuildPy},
    options={
        "bdist_wheel": {},
    },
    package_data={
        "localcode": ["bin/llama-server", "bin/*.dylib", "**/*.tcss"],
    },
    exclude_package_data={
        "localcode": ["*.c"],
    },
)
