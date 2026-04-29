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

    record_name = None
    kept_files = []
    with zipfile.ZipFile(whl, 'r') as zin:
        with zipfile.ZipFile(f"{tmp}/stripped.whl", 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                # Skip directories and RECORD (we'll rewrite it)
                if name.endswith('/') or name.endswith('/RECORD') or name == 'RECORD':
                    if 'RECORD' in name:
                        record_name = name
                    continue
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
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                kept_files.append(f"{name},sha256={digest},{len(data)}")

            # Write fresh RECORD
            if record_name:
                kept_files.append(f"{record_name},,")
                zout.writestr(record_name, "\n".join(kept_files) + "\n")

    shutil.move(f"{tmp}/stripped.whl", str(out))
    shutil.rmtree(tmp)
    print(f"Done: {out}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"Stripping {path}")
        strip_wheel(path)
