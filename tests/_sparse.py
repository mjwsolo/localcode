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


def _set_end_of_file_windows(f, size: int) -> None:
    # file.truncate() on Windows zero-fills when extending, even on a sparse
    # file; SetEndOfFile just moves the end marker.
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    f.flush()
    handle = msvcrt.get_osfhandle(f.fileno())
    if not kernel32.SetFilePointerEx(handle, size, None, 0) or not kernel32.SetEndOfFile(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def make_sparse(path, size: int) -> None:
    path = Path(path)
    if not path.exists():
        path.write_bytes(b"")
    with open(path, "r+b") as f:
        if sys.platform == "win32" and not _set_sparse_windows(f):
            import pytest
            pytest.skip("filesystem has no sparse-file support")
        if sys.platform == "win32":
            _set_end_of_file_windows(f, size)
        else:
            f.truncate(size)
