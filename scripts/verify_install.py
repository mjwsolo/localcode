#!/usr/bin/env python3
"""Install a wheel or pinned PyPI release outside the checkout; optionally upgrade.

Uses disposable pip/uv tool environments, never the user's installed LocalCode.
Example: python scripts/verify_install.py --target dist/localcode-*.whl
         --expected 0.5.9 --installer uv --from-version 0.4.9
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def run(args, cwd, env):
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                            text=True, capture_output=True, timeout=900)
    if result.returncode:
        # Pip/uv can print proxy/index credentials. Keep raw logs local, not in CI.
        raise RuntimeError(f'{Path(str(args[0])).name} failed (exit {result.returncode}); '
                           'check package compatibility and approved network/index settings')
    return result.stdout.strip() or result.stderr.strip()


def verify(target: str, expected: str, installer: str, previous: str | None):
    if Path(target).exists():
        target = str(Path(target).resolve())
    with tempfile.TemporaryDirectory(prefix='localcode-install-') as tmp:
        work = Path(tmp)
        env = {k: v for k, v in os.environ.items() if k not in (
            'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'LOCALCODE_UI_BIN', 'LOCALCODE_HOME')}
        env.update(HOME=str(work / 'home'), LOCALCODE_HOME=str(work / 'state'),
                   XDG_CONFIG_HOME=str(work / 'config'), XDG_DATA_HOME=str(work / 'data'),
                   UV_TOOL_DIR=str(work / 'tools'), UV_TOOL_BIN_DIR=str(work / 'bin'),
                   UV_CACHE_DIR=str(work / 'cache'), PIP_DISABLE_PIP_VERSION_CHECK='1')
        Path(env['HOME']).mkdir()
        if installer == 'pip':
            venv.EnvBuilder(with_pip=True).create(work / 'venv')
            py = work / 'venv/bin/python'
            bindir = py.parent
            def install(spec):
                run([py, '-m', 'pip', 'install', '--upgrade', spec], work, env)
            run([py, '-m', 'pip', 'install', '--upgrade', 'pip'], work, env)
        else:
            uv = shutil.which('uv')
            if not uv:
                raise RuntimeError('uv is required for the uv install test')
            bindir = work / 'bin'
            py = work / 'tools/localcode/bin/python'
            def install(spec):
                run([uv, 'tool', 'install', '--upgrade', '--python', sys.executable, spec], work, env)
        env['PATH'] = str(bindir) + os.pathsep + env.get('PATH', '')
        if previous:
            install('localcode==' + previous)
            assert run([py, '-c', 'from importlib.metadata import version; print(version("localcode"))'], work, env) == previous
        install(target)
        cli = bindir / 'localcode'
        assert shutil.which('localcode', path=env['PATH']) == str(cli), 'wrong executable on PATH'
        assert run([cli, '--version'], work, env) == expected, 'stale CLI version'
        assert run([py, '-c', 'from importlib.metadata import version; print(version("localcode"))'], work, env) == expected
        assert 'LocalCode' in run([cli, '--help'], work, env)
        # Verify installed package data, not the source checkout or an editable install.
        check = ('from pathlib import Path; import localcode; '
                 'p=Path(localcode.__file__).parent; '
                 'assert all((p/x).is_file() for x in '
                 '["bin/llama-server","bin/localcode-ui","ui/plugin/localcode.ts"]); print(p)')
        package = Path(run([py, '-c', check], work, env))
        assert work in package.parents, 'package was imported from outside isolated install'
        for binary in ('llama-server', 'localcode-ui'):
            # llama-server may silence version logs via inherited logging settings.
            # A successful native process exit is the compatibility assertion.
            run([package / 'bin' / binary, '--version'], work, env)
        # Some previously published versions do not yet have doctor; only test it when advertised.
        if 'doctor' in run([cli, '--help'], work, env):
            assert json.loads(run([cli, 'doctor'], work, env))['version'] == expected
        print(json.dumps({'installer': installer, 'from': previous, 'version': expected,
                          'installed_binaries': 'passed', 'result': 'passed'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--installer', choices=['pip', 'uv'], required=True)
    p.add_argument('--from-version')
    a = p.parse_args()
    verify(a.target, a.expected, a.installer, a.from_version)


if __name__ == '__main__':
    main()
