"""Create sparse files of a given size portably (NTFS needs the sparse flag)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _set_sparse_windows(f) -> bool:
    import ctypes
    import msvcrt
    from ctypes import wintypes
    FSCTL_SET_SPARSE = 0x000900C4
    returned = wintypes.DWORD(0)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    ]
    handle = msvcrt.get_osfhandle(f.fileno())
    return bool(kernel32.DeviceIoControl(
        handle, FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(returned), None))


def make_sparse(path, size: int) -> None:
    path = Path(path)
    if not path.exists():
        path.write_bytes(b"")
    with open(path, "r+b") as f:
        if sys.platform == "win32" and not _set_sparse_windows(f):
            import pytest
            pytest.skip("filesystem has no sparse-file support")
        f.truncate(size)
