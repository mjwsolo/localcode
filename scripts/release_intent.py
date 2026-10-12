#!/usr/bin/env python3
"""Publish approved main version bumps automatically; skip non-release pushes."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--before', default='')
    args = parser.parse_args()
    publish = args.event in ('workflow_dispatch', 'release') or args.ref.startswith('refs/tags/')
    if args.event == 'push' and args.ref == 'refs/heads/main':
        current = tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
        if not args.before or set(args.before) == {'0'}:
            publish = True
        else:
            previous = subprocess.check_output(['git', 'show', f'{args.before}:pyproject.toml'], text=True)
            publish = tomllib.loads(previous)['project']['version'] != current
    print(f'publish={str(publish).lower()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
