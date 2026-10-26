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
        return subprocess.check_output(['/usr/sbin/sysctl', '-n', name], stderr=subprocess.DEVNULL,
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
        'local_services': service_report(),
    }


def service_report() -> dict:
    """Read-only local diagnosis: no API keys, prompts or full process arguments."""
    from .model_files import model_files
    from .models_catalog import model_dir
    from .ui.ports import find_running
    result = {'context': os.environ.get('LOCALCODE_CONTEXT', 'minimal'),
              'models_dir': str(model_dir()), 'installed_models': len(model_files(model_dir()))}
    try:
        processes = subprocess.check_output(['ps', '-axo', 'pid,ppid,rss,comm'], text=True, timeout=3)
        servers = []
        for line in processes.splitlines():
            fields = line.strip().split(None, 3)
            if len(fields) == 4 and os.path.basename(fields[3]) == 'llama-server':
                servers.append({'pid': int(fields[0]), 'parent_pid': int(fields[1]), 'rss_kb': int(fields[2])})
        result['model_processes'] = servers
    except (OSError, ValueError, subprocess.SubprocessError):
        result['model_processes'] = None
    try:
        running = find_running()
        result['supervisor_found'] = running is not None
        if running:
            result['control_port'] = running[0]
            result['server'] = {key: running[1][key] for key in (
                'version', 'state', 'current', 'port', 'inference_port', 'server_running',
                'server_ready', 'process_id', 'exit_code', 'ctx', 'warmup',
            ) if key in running[1]}
    except (OSError, ValueError):
        result['supervisor_found'] = False
    return result


def main() -> int:
    print(json.dumps(report(), indent=2))
    return 0
