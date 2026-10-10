#!/usr/bin/env python3
"""Sequential local A/B runs through the shipping task harness; never tune defaults.

Use --execute explicitly to load models. The default prints the experiment plan.
Each repetition reverses profile order to reduce ordering bias. All settings,
commits and hardware metadata are recorded; secrets and full environments aren't.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, supported by LocalCode
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
PROFILES = {
    'baseline': {},
    'compaction-checklist': {'LOCALCODE_COMPACTION_CHECKLIST': '1'},
    'no-vendor-draft': {'LOCALCODE_LLAMA_CPP_MTP_DRAFT': '0'},
    'context-16k': {'LOCALCODE_OVERRIDE_NCTX': '16384'},
    'batch-512': {'LOCALCODE_OVERRIDE_BATCH': '512'},
    'kv-q8': {'LOCALCODE_KV_CACHE_TYPE_K': 'q8_0', 'LOCALCODE_KV_CACHE_TYPE_V': 'q8_0'},
    'reasoning-on': {'LOCALCODE_INTERNAL_THINKING_MODE': 'on'},
}


def compare(baseline, candidate):
    """Fail closed on missing/duplicate trials. No averaging away task regressions."""
    def index(rows):
        result = {}
        for row in rows:
            key = (row['task'], row['model'], row['trial'])
            if key in result:
                raise ValueError(f'duplicate trial: {key}')
            result[key] = row
        return result
    b, c = index(baseline), index(candidate)
    if not b or b.keys() != c.keys():
        raise ValueError('comparison requires identical, nonempty task/model/trial sets')
    regressions = [list(k) for k in b if b[k]['pass'] and not c[k]['pass']]
    paired_times = [(b[k]['wall_s'], c[k]['wall_s']) for k in b if b[k]['pass'] and c[k]['pass']]
    return {'baseline_passes': sum(r['pass'] for r in b.values()),
            'candidate_passes': sum(r['pass'] for r in c.values()), 'trials': len(b),
            'regressed_trials': regressions,
            'median_paired_wall_delta_s': statistics.median(y-x for x,y in paired_times) if paired_times else None,
            'decision': 'investigate-regressions' if regressions else 'review-measurements',
            'note': 'A screening result, not statistical proof or approval to change defaults.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tasks', default='fix-one-bug,fix-two-bugs,ten-turn-session')
    ap.add_argument('--profiles', default=','.join(PROFILES))
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    names = a.profiles.split(',')
    if a.repetitions < 1 or len(names) != len(set(names)) or 'baseline' not in names or any(n not in PROFILES for n in names):
        ap.error('use unique known profiles including baseline and positive repetitions')
    # All named tasks must exist before we start any model process.
    tasks = a.tasks.split(',')
    if not tasks or any(not (HERE/'tasks'/t/'task.toml').is_file() or '/' in t or '..' in t for t in tasks):
        ap.error('unknown task')
    expected = {(task, a.model, trial) for task in tasks
                for trial in range(1, max(1, int(tomllib.loads((HERE/'tasks'/task/'task.toml').read_text()).get('trials', 1))) + 1)}
    schedule = [(r, n) for r in range(1, a.repetitions+1) for n in (names if r % 2 else list(reversed(names)))]
    print(json.dumps([{'repetition': r, 'profile': n, 'overrides': PROFILES[n]} for r,n in schedule], indent=2))
    if not a.execute:
        return 0
    out = (a.out or HERE/'runs'/f"sweep-{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    repo = Path(a.repo).resolve()
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    hardware = {}
    for key in ('hw.memsize', 'machdep.cpu.brand_string'):
        try:
            hardware[key] = subprocess.check_output(['sysctl','-n',key], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            hardware[key] = None
    (out/'manifest.json').write_text(json.dumps({'repo_sha':sha, 'dirty_checkout':bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()), 'model':a.model, 'hardware':hardware, 'platform':platform.platform(), 'machine':platform.machine(), 'profiles':{n:PROFILES[n] for n in names}, 'tasks':tasks, 'repetitions':a.repetitions}, indent=2))
    all_rows = {n:[] for n in names}
    for repetition, name in schedule:
        target = out/f'{repetition}-{name}'
        # Explicitly neutralize other experiment overrides; retain ordinary launcher config.
        env = {k:v for k,v in os.environ.items() if k not in {key for p in PROFILES.values() for key in p}}
        env.update(PROFILES[name])
        command = [sys.executable,str(HERE/'run_tasks.py'),'--repo',str(repo),'--model',a.model,'--tasks',a.tasks,'--trials','1','--out',str(target),'--label',name]
        with (out/f'{repetition}-{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (target/'results.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != len(expected) or {(r['task'], r['model'], r['trial']) for r in rows} != expected:
            raise ValueError('incomplete task run')
        for row in rows:
            row['trial'] = f"{repetition}:{row['trial']}"
        all_rows[name].extend(rows)
    reports = {n:compare(all_rows['baseline'],rows) for n,rows in all_rows.items() if n != 'baseline'}
    (out/'comparison.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
