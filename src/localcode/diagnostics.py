"""Offline installation/platform diagnostics; never prints credential values."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version


def sysctl(name: str) -> str:
    try:
        return subprocess.check_output(['sysctl', '-n', name], stderr=subprocess.DEVNULL,
                                       text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def platform_problem() -> str | None:
    if platform.system() != 'Darwin':
        return 'This release requires an Apple silicon Mac with macOS 13 or newer.'
    if platform.machine().lower() != 'arm64':
        if sysctl('sysctl.proc_translated') == '1' or sysctl('hw.optional.arm64') == '1':
            return ('Python is running under Rosetta. Open a native Apple silicon terminal, '
                    'install an arm64 Python or uv, then reinstall LocalCode there.')
        return 'Intel Macs are not supported by the bundled Apple silicon binaries.'
    release = platform.mac_ver()[0]
    if not release or not release.split('.')[0].isdigit():
        return 'Could not determine macOS version. LocalCode requires macOS 13 or newer.'
    if int(release.split('.')[0]) < 13:
        return f'macOS {release} is too old. Upgrade to macOS 13 or newer.'
    return None


def report() -> dict:
    from localcode import __version__
    try:
        installed = version('localcode')
    except Exception:
        installed = __version__
    return {
        'version': installed, 'python': sys.version.split()[0],
        'python_executable': sys.executable, 'localcode_on_path': shutil.which('localcode'),
        'system': platform.system(), 'architecture': platform.machine(),
        'macos': platform.mac_ver()[0], 'chip': sysctl('machdep.cpu.brand_string') if platform.system() == 'Darwin' else '',
        'ram_bytes': sysctl('hw.memsize') if platform.system() == 'Darwin' else '',
        'platform_problem': platform_problem(),
        'network_configuration_present': {name: bool(os.environ.get(name)) for name in (
            'HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy', 'SSL_CERT_FILE',
            'REQUESTS_CA_BUNDLE', 'PIP_INDEX_URL', 'UV_INDEX_URL', 'UV_DEFAULT_INDEX')},
    }


def main() -> int:
    print(json.dumps(report(), indent=2))
    return 0
