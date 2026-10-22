#!/usr/bin/env python3
"""Exercise automatic hybrid/DFlash persistence against a running llama-server."""

import argparse
import json
from pathlib import Path
import time

from probe import Probe


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def stats(probe):
    return probe.api('/slots')[0]['persistent_cache']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--stage', choices=['produce', 'restore', 'miss'], required=True)
    args = parser.parse_args()
    probe = Probe(args.base, args.output)
    body = {'n_predict': 128, 'temperature': 0, 'seed': 7319,
            'return_tokens': True, 'cache_prompt': True}
    before = stats(probe)
    if args.stage == 'produce':
        prompt = probe.long_prompt(4096)
        initial = probe.record('initial', '/completion', body | {'prompt': prompt})
        saved = stats(probe)
        check(saved['saves'] > before['saves'], 'request was not saved automatically')
        continuation = body | {'prompt': prompt + initial['tokens'] + probe.tokenize('\nContinue the explanation with a concrete example.\n')}
        hot = probe.record('hot-continuation', '/completion', continuation)
        args.state.write_text(json.dumps({'request': continuation, 'hot': hot, 'saved': saved}) + '\n')
        check(hot['timings']['cache_n'] > 3000, 'GPU prefix reuse failed')
    else:
        state = json.loads(args.state.read_text())
        start = time.monotonic()
        restored = probe.record('automatic-continuation', '/completion', state['request'])
        after = stats(probe)
        reused = restored['timings']['cache_n']
        if args.stage == 'restore':
            check(reused > 3000, f'automatic restart reuse failed: {reused}')
            check(after['hits'] > before['hits'], 'no persistent hit recorded')
            check(after['disk_read_bytes'] > before['disk_read_bytes'], 'no disk data restored')
            check(restored['tokens'] == state['hot']['tokens'], 'restored output differs from GPU-hot reference')
            check(restored['timings'].get('draft_n_accepted', 0) > 0, 'DFlash did not accept restored proposals')
        else:
            check(reused == 0, f'expected safe cold miss, got {reused} cached tokens')
        (args.output / 'result.json').write_text(json.dumps({'stage': args.stage, 'passed': True,
            'cache_n': reused, 'timings': restored['timings'], 'elapsed_seconds': time.monotonic()-start,
            'before': before, 'after': after}, indent=2) + '\n')
    print(json.dumps(stats(probe), indent=2), flush=True)


if __name__ == '__main__':
    main()
