__all__ = ["__version__"]

# Read the version from installed package metadata so it can never drift from
# pyproject.toml (it was hardcoded and went ~12 releases stale). Falls back to a
# hardcoded value only when running from a source tree that isn't installed.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("localcode")
except PackageNotFoundError:  # not installed (e.g. raw source checkout)
    __version__ = "0.3.46"
