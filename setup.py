"""Build script — compiles all .py to .so via Cython for distribution.

Skip Cython for editable/dev installs: LOCALCODE_NO_CYTHON=1 pip install -e .
"""
import os
from pathlib import Path
from setuptools import setup

ext_modules = []

# Only compile Cython for wheel builds, not editable/dev installs
if not os.environ.get("LOCALCODE_NO_CYTHON") and not os.environ.get("SKIP_CYTHON"):
    try:
        from Cython.Build import cythonize
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
    except ImportError:
        pass


class StripSourceBuildWheel:
    """Remove .py and .c source files from the wheel, keeping only .so"""
    pass


from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    """Include binary and data files that setuptools normally skips."""
    def build_package_data(self):
        super().build_package_data()
        # Copy llama-server binary
        src = Path("src/gem/bin/llama-server")
        if src.exists():
            dst_dir = Path(self.build_lib) / "gem" / "bin"
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
        "gem": ["bin/llama-server", "bin/*.dylib", "**/*.tcss"],
    },
    exclude_package_data={
        "gem": ["*.c"],
    },
)
