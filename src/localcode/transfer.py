"""Resumable HTTP ranges. Only fsynced, checksummed chunks are checkpointed."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import ssl
import threading
import time
import urllib.request

CHUNK_BYTES = 8 * 1024 * 1024


class RangeUnsupported(OSError):
    pass


def download(url: str, dest: Path, *, workers=16, progress=None, cancel=None):
    from .bootstrap import DownloadCancelled
    import fcntl
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    dest = Path(dest)
    part = dest.with_name(dest.name + '.part')
    state = dest.with_name(dest.name + '.resume.json')
    lockpath = dest.with_name(dest.name + '.download.lock')
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Never follow a planted sidecar symlink or let two writers share a partial.
    for p in (part, state, state.with_suffix('.tmp'), lockpath):
        if p.is_symlink():
            raise OSError('Download sidecar must not be a symlink')
    lockfd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OSError('A download of this file is already in progress') from None
        def cancelled():
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled()
        cancelled()
        headers = {'Accept-Encoding': 'identity'}
        def request(extra=None, method=None):
            return urllib.request.urlopen(urllib.request.Request(url, headers={**headers, **(extra or {})}, method=method), context=ctx, timeout=30)
        # A probe validates ranges instead of trusting Accept-Ranges alone.
        with request({'Range': 'bytes=0-0'}) as response:
            cr = response.headers.get('Content-Range', '')
            ranged = response.status == 206 and cr.startswith('bytes 0-0/')
            size = int(cr.split('/')[-1]) if ranged else int(response.headers.get('Content-Length') or 0)
            etag = response.headers.get('ETag', '')
            if etag.startswith('W/'):
                etag = ''
        identity = {'url': hashlib.sha256(url.encode()).hexdigest(), 'size': size, 'etag': etag, 'chunk': CHUNK_BYTES}
        done = {}
        if ranged and etag and part.is_file() and part.stat().st_size == size:
            try:
                saved = json.loads(state.read_text())
                if saved.get('identity') == identity:
                    with part.open('rb') as f:
                        for key, digest in saved.get('done', {}).items():
                            i = int(key)
                            if i < 0 or i * CHUNK_BYTES >= size:
                                continue
                            f.seek(i * CHUNK_BYTES)
                            if hashlib.sha256(f.read(min(CHUNK_BYTES, size-i*CHUNK_BYTES))).hexdigest() == digest:
                                done[str(i)] = digest
            except (OSError, ValueError, TypeError, AttributeError):
                done = {}
        def checkpoint():
            temp = state.with_suffix('.tmp')
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump({'identity': identity, 'done': done}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, state)
        def sequential():
            state.unlink(missing_ok=True)
            cancelled()
            with request() as response:
                if response.status != 200:
                    raise OSError('Expected a full HTTP response')
                expected = int(response.headers.get('Content-Length') or 0)
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                count = 0
                with os.fdopen(fd, 'wb') as f:
                    while True:
                        cancelled()
                        data = response.read(1024 * 1024)
                        if not data:
                            break
                        f.write(data)
                        count += len(data)
                        if progress and expected:
                            progress(f'Downloading: {count//1048576}/{expected//1048576} MB ({min(100,count*100//expected)}%)')
                    f.flush()
                    os.fsync(f.fileno())
                if expected and count != expected:
                    raise OSError('Incomplete HTTP download')
            cancelled()
            part.replace(dest)
        if not ranged or size <= 0:
            sequential()
            return
        fd = os.open(part, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'r+b') as f:
            if not done:
                f.truncate(0)
            f.truncate(size)
        checkpoint()
        mutex = threading.Lock()
        stopped = threading.Event()
        transferred = {}
        initial = sum(min(CHUNK_BYTES, size-int(i)*CHUNK_BYTES) for i in done)
        started = time.monotonic()
        def chunk(i):
            if stopped.is_set():
                return
            cancelled()
            start = i * CHUNK_BYTES
            end = min(size, start + CHUNK_BYTES) - 1
            extra = {'Range': f'bytes={start}-{end}'}
            if etag:
                extra['If-Range'] = etag
            try:
                with request(extra) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RangeUnsupported('Server did not honor the requested byte range')
                    if etag and response.headers.get('ETag', etag) != etag:
                        raise RangeUnsupported('Remote file changed during download')
                    digest = hashlib.sha256()
                    count = 0
                    fd = os.open(part, os.O_RDWR | os.O_NOFOLLOW)
                    with os.fdopen(fd, 'r+b') as f:
                        f.seek(start)
                        while count < end-start+1:
                            cancelled()
                            if stopped.is_set():
                                return
                            data = response.read(min(1024*1024, end-start+1-count))
                            if not data:
                                raise OSError('Truncated HTTP byte range')
                            f.write(data)
                            digest.update(data)
                            count += len(data)
                            with mutex:
                                transferred[i] = count
                        if response.read(1):
                            raise OSError('Oversized HTTP byte range')
                        f.flush()
                        os.fsync(f.fileno())
                    with mutex:
                        done[str(i)] = digest.hexdigest()
                        checkpoint()
            except BaseException:
                stopped.set()
                raise
        failure = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as pool:
            pending = {pool.submit(chunk, i) for i in range((size+CHUNK_BYTES-1)//CHUNK_BYTES) if str(i) not in done}
            while pending:
                completed, pending = concurrent.futures.wait(pending, timeout=.25, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    try:
                        future.result()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                with mutex:
                    received = initial + sum(transferred.values())
                if progress:
                    speed = (received-initial)/max(time.monotonic()-started, .001)
                    progress(f'Downloading: {received//1048576}/{size//1048576} MB ({min(100,received*100//size)}%) · {speed/1e6:.1f} MB/s')
        cancelled()
        if isinstance(failure, RangeUnsupported):
            sequential()
            return
        if failure:
            raise failure
        if len(done) != (size+CHUNK_BYTES-1)//CHUNK_BYTES:
            raise OSError('Download has incomplete chunks')
        part.replace(dest)
        state.unlink(missing_ok=True)
    finally:
        os.close(lockfd)
