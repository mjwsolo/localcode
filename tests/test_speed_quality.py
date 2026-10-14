import json
import pytest
from localcode.ui.context_budget import Sample
from localcode.ui.diagnostics import timing_summary
from localcode.ui.launch import write_config
from localcode import recommendations


def test_measurements_remain_unknown_and_short_decodes_excluded():
    assert timing_summary([])['decode_tps'] is None
    sample = Sample(1000, 100, 100, 10, 20)
    data = timing_summary([sample])
    assert data['decode_tps'] == 10
    assert data['prefill_tps'] is None
    assert 89 < data['prompt_reuse_estimate_pct'] < 90
    assert timing_summary([Sample(100, 100, 100, 90, 1)])['decode_tps'] is None


def test_context_policy_defaults_and_rollback(tmp_path, monkeypatch):
    file = tmp_path / 'config.json'
    monkeypatch.delenv('LOCALCODE_PRUNE', raising=False)
    write_config(file, port=9999, ctx=8192, alias='test')
    config = json.loads(file.read_text())
    assert config['compaction']['prune'] is True
    assert config['tool_output'] == {'max_bytes':8192, 'max_lines':400}
    monkeypatch.setenv('LOCALCODE_PRUNE', '0')
    write_config(file, port=9999, ctx=131072, alias='test')
    config = json.loads(file.read_text())
    assert config['compaction']['prune'] is False
    assert config['tool_output']['max_bytes'] == 16384


def test_recommendations_are_machine_bound_and_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / 'profile.json'
    monkeypatch.setattr(recommendations, 'hardware_identity', lambda: {'ram':'16', 'chip':'M1'})
    recommendations.save_recommendation('a.gguf', {'n_ctx':8192}, path=path)
    assert recommendations.load_overrides('a.gguf', path) == {'LOCALCODE_OVERRIDE_NCTX':'8192'}
    monkeypatch.setattr(recommendations, 'hardware_identity', lambda: {'ram':'32', 'chip':'M2'})
    assert recommendations.load_overrides('a.gguf', path) == {}
    monkeypatch.setattr(recommendations, 'hardware_identity', lambda: {'ram':'16', 'chip':'M1'})
    recommendations.save_recommendation('a.gguf', {'n_ctx':-1, 'n_batch':512}, path=path)
    assert recommendations.load_overrides('a.gguf', path) == {}
    path.write_text('[]')
    assert recommendations.load_overrides('a.gguf', path) == {}


@pytest.mark.usefixtures("loopback_listener")
def test_probe_counts_first_delta_and_requires_complete_stream():
    import importlib.util
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path
    from unittest.mock import patch
    spec = importlib.util.spec_from_file_location('probe_server', Path(__file__).resolve().parents[1] / 'evals/probe_server.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.end_headers()
            for chunk in [{'choices':[{'delta':{'role':'assistant'}}]},
                          {'choices':[{'delta':{'content':'hi'}}]},
                          {'choices':[{'delta':{},'finish_reason':'stop'}], 'usage':{'completion_tokens':1}}]:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
    with patch('socket.getfqdn', return_value='localhost'):
        server = HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = probe.probe(server.server_port)
        assert 0 <= result['time_to_first_token_s'] <= result['wall_s']
        assert result['finish_reason'] == 'stop'
        assert 'hi' not in json.dumps(result)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_speed_gate_cannot_hide_slow_model_behind_fast_model(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'evals/gate.py'
    rows = [{'task':str(i),'model':model,'trial':1,'pass':True,'purpose':'regression','decode_tps':speed}
            for model, speed, count in [('slow',10,1),('fast',100,5)] for i in range(count)]
    baseline = tmp_path / 'b.jsonl'
    candidate = tmp_path / 'c.jsonl'
    baseline.write_text('\n'.join(map(json.dumps,rows)))
    candidate.write_text('\n'.join(json.dumps(dict(row,decode_tps=1 if row['model']=='slow' else 100)) for row in rows))
    result = subprocess.run([sys.executable,str(script),str(baseline),str(candidate),'--require-performance'], capture_output=True,text=True)
    assert result.returncode == 1
    assert 'slow median decode' in result.stdout
    candidate.write_text('\n'.join(json.dumps(dict(row,decode_tps=None)) for row in rows))
    result = subprocess.run([sys.executable,str(script),str(baseline),str(candidate),'--require-performance'], capture_output=True,text=True)
    assert result.returncode == 1
