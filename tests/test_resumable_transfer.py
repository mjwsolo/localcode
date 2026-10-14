import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localcode.bootstrap import DownloadCancelled
from localcode.transfer import download


@pytest.fixture
def source(monkeypatch, loopback_listener):
    monkeypatch.setattr('localcode.transfer.CHUNK_BYTES', 65536)
    data = bytes(range(256)) * 8192
    state = {'etag': '"version1"', 'ranges': [], 'ignore': False, 'bad': False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            raw = self.headers.get('Range')
            if raw and not state['ignore'] and not (state.get('ignore_chunks') and raw != 'bytes=0-0'):
                start, end = map(int, raw[6:].split('-'))
                state['ranges'].append((start, end))
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
                content = data[start:end+1]
                if state['bad'] and start > 65536:
                    content = content[:-1]
            else:
                self.send_response(200)
                content = data
            self.send_header('ETag', state['etag'])
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            try:
                time.sleep(.015)
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/model', data, state
    server.shutdown();server.server_close();thread.join()


def pause(url, dest):
    stop = threading.Event()
    state = dest.with_name(dest.name + '.resume.json')
    def progress(_):
        if state.exists() and len(json.loads(state.read_text())['done']) >= 2:
            stop.set()
    with pytest.raises(DownloadCancelled):
        download(url, dest, workers=1, progress=progress, cancel=stop)
    return json.loads(state.read_text())


def test_resume_reuses_verified_chunks(source, tmp_path):
    url, data, state = source;dest=tmp_path/'m.gguf'
    saved = pause(url, dest)
    assert not dest.exists()
    assert 0 < len(saved['done']) < len(data)//65536
    state['ranges'].clear()
    download(url, dest, workers=4)
    assert dest.read_bytes() == data
    fetched = {a//65536 for a,b in state['ranges'] if b > 0}
    assert not fetched.intersection(map(int, saved['done']))
    assert not dest.with_name(dest.name+'.resume.json').exists()


@pytest.mark.parametrize('change', ['etag', 'corrupt', 'malformed'])
def test_resume_revalidates_identity_and_partial_bytes(source, tmp_path, change):
    url,data,state=source;dest=tmp_path/'m.gguf'
    saved=pause(url,dest)
    if change=='etag': state['etag']='"version2"'
    elif change=='corrupt':
        with dest.with_name(dest.name+'.part').open('r+b') as f:f.write(b'corrupt')
    else:dest.with_name(dest.name+'.resume.json').write_text('broken json')
    state['ranges'].clear();download(url,dest,workers=4)
    assert dest.read_bytes()==data
    assert (0,65535) in state['ranges']


def test_servers_without_ranges_are_supported(source,tmp_path):
    url,data,state=source;state['ignore']=True;dest=tmp_path/'m'
    download(url,dest);assert dest.read_bytes()==data


def test_truncated_range_never_becomes_final_file(source,tmp_path):
    url,data,state=source;state['bad']=True;dest=tmp_path/'m'
    with pytest.raises(OSError):download(url,dest,workers=2)
    assert not dest.exists()
    state['bad']=False;download(url,dest)
    assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(data).digest()


def test_sidecar_symlink_is_rejected(source,tmp_path):
    url,_,_=source;dest=tmp_path/'m';victim=tmp_path/'victim';victim.write_text('keep')
    dest.with_name('m.part').symlink_to(victim)
    with pytest.raises(OSError,match='symlink'):download(url,dest)
    assert victim.read_text()=='keep'


def test_range_ignored_mid_download_falls_back_safely(source,tmp_path):
    url,data,state=source;state['ignore_chunks']=True;dest=tmp_path/'m'
    download(url,dest,workers=2)
    assert dest.read_bytes()==data


def test_concurrent_writer_is_rejected(source,tmp_path):
    import fcntl
    url,_,_=source;dest=tmp_path/'m'
    with dest.with_name('m.download.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(OSError,match='already in progress'):download(url,dest)


def test_resume_after_process_is_killed(source,tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    url,data,state=source;dest=tmp_path/'m'
    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    code='from pathlib import Path; import sys; import localcode.transfer as t; t.CHUNK_BYTES=65536; t.download(sys.argv[1],Path(sys.argv[2]),workers=1)'
    p=subprocess.Popen([sys.executable,'-c',code,url,str(dest)],env=env)
    checkpoint=dest.with_name('m.resume.json')
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and p.poll() is None:
            if checkpoint.exists() and len(json.loads(checkpoint.read_text())['done'])>=2:
                break
            time.sleep(.005)
        assert p.poll() is None
        p.kill();p.wait(timeout=5)
        saved=json.loads(checkpoint.read_text())
        assert saved['done'] and not dest.exists()
        state['ranges'].clear();download(url,dest,workers=2)
        assert dest.read_bytes()==data
        fetched={a//65536 for a,b in state['ranges'] if b>0}
        assert not fetched.intersection(map(int,saved['done']))
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=5)
