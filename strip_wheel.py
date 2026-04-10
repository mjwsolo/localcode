"""Strip .py and .c source files from a wheel, keeping only compiled .so files.

Usage: python strip_wheel.py dist/localcode-*.whl
"""
import hashlib
import base64
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

def strip_wheel(whl_path: str) -> None:
    whl = Path(whl_path)
    tmp = tempfile.mkdtemp()
    out = whl.parent / whl.name

    kept_files = []
    with zipfile.ZipFile(whl, 'r') as zin:
        with zipfile.ZipFile(f"{tmp}/stripped.whl", 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                # Keep: .so files, __init__.py, __main__.py, metadata, licenses
                # Remove: .py (source), .c (cython intermediate)
                if name.endswith('.py'):
                    base = Path(name).name
                    if base not in ('__init__.py', '__main__.py'):
                        print(f"  strip: {name}")
                        continue
                if name.endswith('.c'):
                    print(f"  strip: {name}")
                    continue
                data = zin.read(name)
                zout.writestr(item, data)
                # Track for RECORD (skip RECORD itself)
                if not name.endswith('/RECORD'):
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                    kept_files.append(f"{name},sha256={digest},{len(data)}")

            # Rewrite RECORD with correct hashes
            record_name = [n for n in zin.namelist() if n.endswith('/RECORD')][0]
            kept_files.append(f"{record_name},,")
            record_data = "\n".join(kept_files) + "\n"
            zout.writestr(record_name, record_data)

    shutil.move(f"{tmp}/stripped.whl", str(out))
    shutil.rmtree(tmp)
    print(f"Done: {out}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"Stripping {path}")
        strip_wheel(path)
