"""Probe composite concepts and compute recovery scores against primitives."""

import json
import sys
sys.path.insert(0, '.')

from src.client import NeuronpediaClient
from src.probes import load_config, probe_composite
from src.analysis import composition_recovery, null_distribution, RESULTS_DIR


def main():
    config = load_config()
    composites = list(config.get('composites', {}).keys())
    client = NeuronpediaClient()

    # probe composites
    print(f'probing {len(composites)} composites')
    for name in composites:
        probe_composite(name, client=client)

    # compute recovery scores
    recoveries = {}
    for name in composites:
        rec = composition_recovery(name)
        recoveries[name] = rec
        print(f'  {name}: {rec["aggregate_recovery"]:.3f} '
              f'[{rec["ci_low"]:.3f}, {rec["ci_high"]:.3f}]')

    # null baseline
    null = null_distribution()
    print(f'  null mean_jaccard={null["mean_jaccard"]:.3f}')

    # save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'composition_recovery.json'
    out.write_text(json.dumps(
        {'recoveries': recoveries, 'null': null}, indent=2
    ))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
