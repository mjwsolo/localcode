#!/usr/bin/env python3
"""Measure a local server's streamed TTFT and cache reuse, without the harness.

Example: LOCALCODE_SERVER_KEY=... python evals/probe_server.py --port 8080
Run sequentially on an idle machine. Compare with evals/run_tasks.py for harness
latency and correctness. This probe sends a fixed synthetic prompt, no user data.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def probe(port, key='', *, tokens=96, timeout=120):
    body = {'messages': [{'role': 'user', 'content': 'Write a numbered list of twenty practical tips for readable Python functions.'}],
            'stream': True, 'stream_options': {'include_usage': True},
            'temperature': 0, 'max_tokens': tokens, 'cache_prompt': True}
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    start = time.monotonic()
    first = None
    usage = timings = finish = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if time.monotonic() - start > timeout:
                raise TimeoutError('Probe exceeded its total time budget')
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                done = True
                break
            chunk = json.loads(payload)
            usage = chunk.get('usage') or usage
            timings = chunk.get('timings') or timings
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls')):
                    first = time.monotonic() - start
                finish = choice.get('finish_reason') or finish
    if not done or not finish:
        raise ValueError('Incomplete stream: no terminal finish and DONE received')
    return {'time_to_first_token_s': first, 'wall_s': time.monotonic() - start,
            'usage': usage, 'timings': timings, 'finish_reason': finish}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=96)
    parser.add_argument('--auth-file', help='LocalCode run directory auth.json (key is never printed)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.repeat <= 20 or not 16 <= args.tokens <= 4096:
        parser.error('invalid port, repeat (1–20), or tokens (16–4096)')
    key = os.environ.get('LOCALCODE_SERVER_KEY', '')
    if args.auth_file:
        with open(args.auth_file) as stream:
            key = json.load(stream)['server_key']
    for trial in range(args.repeat):
        print(json.dumps({'trial': trial + 1, **probe(args.port, key, tokens=args.tokens)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'Probe failed: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
