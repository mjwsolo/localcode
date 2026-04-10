"""Offline package management — install common packages without internet.

When user's code needs a package and they're offline, we:
1. Check if the package is in our local cache
2. If yes, install from cache
3. If no, tell user which packages are needed

Cache is built with: localcode cache-build
"""
from __future__ import annotations

import subprocess
from .config import ensure_home_dirs
from .network import is_online


CACHE_DIR = ensure_home_dirs() / "package_cache"

# Tier 1: Essential, tiny (<1MB each) — always cache (~3MB total)
TIER1_PACKAGES = [
    "requests", "click", "pyyaml", "toml", "httpx",
    "beautifulsoup4", "python-dotenv", "colorama",
]

# Tier 2: Common, medium (1-10MB each) — cache on demand (~25MB total)
TIER2_PACKAGES = [
    "flask", "fastapi", "uvicorn", "pytest",
    "black", "pillow",
]

# Tier 3: Heavy (10MB+) — only cache if user asks (~100MB total)
TIER3_PACKAGES = [
    "numpy", "pandas", "matplotlib", "pygame",
    "ruff", "lxml",
]

COMMON_PACKAGES = TIER1_PACKAGES  # default: just the essentials


def install_package(package: str) -> tuple[bool, str]:
    """Install a package — from cache if offline, from PyPI if online."""
    if is_online():
        result = subprocess.run(
            ["pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stdout + result.stderr

    # Offline — try cache
    cache = CACHE_DIR / package
    if cache.exists():
        wheels = list(cache.glob("*.whl"))
        if wheels:
            result = subprocess.run(
                ["pip", "install", "--no-index", "--find-links", str(cache), package],
                capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0, result.stdout + result.stderr

    return False, f"Offline and {package} not in cache. Run `localcode cache-build` while online."


def build_cache(packages: list[str] | None = None) -> tuple[int, str]:
    """Download packages to local cache for offline use."""
    if not is_online():
        return 0, "Cannot build cache — no internet connection."

    pkgs = packages or COMMON_PACKAGES
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cached = 0
    for pkg in pkgs:
        pkg_dir = CACHE_DIR / pkg
        pkg_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            ["pip", "download", "-d", str(pkg_dir), pkg],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            cached += 1

    return cached, f"Cached {cached}/{len(pkgs)} packages to {CACHE_DIR}"


def is_package_cached(package: str) -> bool:
    cache = CACHE_DIR / package
    return cache.exists() and any(cache.glob("*.whl"))
