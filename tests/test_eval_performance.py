"""Protect eval correctness independently of a loaded model."""
import importlib.util
from pathlib import Path
import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'evals'/'localcode_evals'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_interleaved_request_timings_and_unknown_usage():
    metrics = module('performance').summarize([], '''slot print_timing: id 0 | task 42 | prompt eval time = 100 ms / 50 tokens
slot print_timing: id 1 | task 43 | prompt eval time = 200 ms / 70 tokens
slot print_timing: id 0 | task 42 | eval time = 800 ms / 20 tokens
''')
    assert metrics['server_requests'] == [{'prefill_ms':100, 'prefill_tokens':50, 'decode_ms':800, 'decode_tokens':20}, {'prefill_ms':200, 'prefill_tokens':70}]
    assert metrics['usage_tokens'] is None
    assert metrics['time_to_first_token_s'] is None


def test_tool_lifecycle_deduplication_and_privacy():
    import json
    event = {'type':'tool_use','part':{'callID':'one','tool':'bash','state':{'status':'completed','input':{'command':'private-secret'},'time':{'start':1000,'end':1500}}}}
    repeat = json.loads(json.dumps(event)); repeat['part']['callID']='two'; repeat['part']['state']['status']='error'
    result = module('performance').summarize([event,event,repeat], '')
    assert result['tool_calls_by_name']=={'bash':2}
    assert result['repeated_tool_calls']==1
    assert result['tool_errors']==1
    assert result['tool_duration_s']==1
    assert 'private-secret' not in json.dumps(result)


def test_usage_preserves_separate_cache_counters():
    result=module('performance').summarize([{'type':'step_finish','part':{'tokens':{'input':20,'output':5,'cache':{'read':90,'write':10}}}}], '')
    assert result['usage_tokens']==dict(input=20, output=5, cache_read=90, cache_write=10)


def test_resource_sampling_excludes_unrelated_processes():
    rss=module('performance').tree_rss
    assert rss('10 1 20\n11 10 30\n12 11 40\n99 1 1000000',10)==90*1024
    assert rss('99 1 1000000',10) is None


@pytest.mark.usefixtures("process_inspection")
def test_timeout_retains_output_and_stops_child_group():
    import os
    import signal
    import subprocess
    import sys
    import time
    run = module('process').run_bounded
    script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run([sys.executable,'-c',script], timeout=0.5, text=True)
    pid = int(failure.value.stdout.strip())
    # A dead child may briefly remain as a zombie until reparented/reaped.
    state = subprocess.run(['ps','-o','stat=','-p',str(pid)], capture_output=True,text=True).stdout.strip()
    assert not state or state.startswith('Z')


@pytest.mark.usefixtures("loopback_listener")
def test_loopback_fixture_never_uses_reverse_dns(monkeypatch):
    from http.server import BaseHTTPRequestHandler
    spec = importlib.util.spec_from_file_location('http_fixture', Path(__file__).resolve().parents[1]/'evals/checks/http_fixture.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    def forbidden(*args):
        raise AssertionError('fixture attempted reverse DNS')
    monkeypatch.setattr('socket.getfqdn', forbidden)
    with fixture.LoopbackHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler) as server:
        assert server.server_name == '127.0.0.1'
        assert server.server_port > 0


def test_prefix_cache_hit_rate_per_step():
    steps = [{'type': 'step_finish', 'part': {'tokens': {'input': processed, 'output': 5, 'cache': {'read': cached, 'write': 0}}}}
             for processed, cached in ((1000, 0), (100, 1900), (0, 0))]
    result = module('performance').summarize(steps, '')
    assert result['prompt_cache_by_step'] == [{'cached': 0, 'processed': 1000}, {'cached': 1900, 'processed': 100}]
    assert result['prefix_cache_hit_rate'] == 0.6333
    assert result['reprocessed_prompt_tokens'] == 1100
    assert module('performance').summarize([], '')['prefix_cache_hit_rate'] is None


def test_malformed_tool_calls_are_counted_from_the_transcript():
    def call(cid, status, error=''):
        return {'type': 'tool_use', 'part': {'callID': cid, 'tool': 'edit', 'state': {'status': status, 'input': {}, 'error': error}}}
    events = [
        call('a', 'completed'),
        call('b', 'error', 'The tool input could not be parsed as JSON, so the call was not executed.'),
        call('c', 'error', 'Invalid arguments for tool "edit":\n- path: missing'),
        call('d', 'error', 'oldString not found in file'),            # ran and failed: not malformed
        {'type': 'text', 'part': {'text': 'I will call <tool_call>{"name": "read"}</tool_call>'}},
    ]
    result = module('performance').summarize(events, '')
    assert result['malformed_tool_calls'] == 3
    assert result['malformed_tool_call_rate'] == 0.6
    assert module('performance').summarize([], '')['malformed_tool_call_rate'] is None
