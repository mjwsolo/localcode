"""Protect eval correctness independently of a loaded model."""
import importlib.util
from pathlib import Path
import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'evals'/f'{name}.py')
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


def test_comparison_rejects_missing_and_duplicate_trials():
    compare=module('sweep').compare
    row={'task':'a','model':'m','trial':1,'pass':True,'wall_s':10}
    with pytest.raises(ValueError): compare([row],[])
    with pytest.raises(ValueError): compare([row,row],[row])
    with pytest.raises(ValueError): compare([],[])
    result=compare([row],[dict(row,**{'pass':False,'wall_s':1})])
    assert result['decision']=='investigate-regressions'
    assert result['median_paired_wall_delta_s'] is None


def test_resource_sampling_excludes_unrelated_processes():
    rss=module('performance').tree_rss
    assert rss('10 1 20\n11 10 30\n12 11 40\n99 1 1000000',10)==90*1024
    assert rss('99 1 1000000',10) is None


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


@pytest.mark.parametrize('name', ['run_tasks.py', 'sweep.py'])
def test_eval_cli_help_starts_without_loading_models(name):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1]/'evals'/name
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--model' in result.stdout
