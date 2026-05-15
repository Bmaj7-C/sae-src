"""Probe opposition pairs and compute shared vs unique features."""

import json
import sys
sys.path.insert(0, '.')

from src.client import NeuronpediaClient
from src.probes import load_config, probe_pair
from src.analysis import RESULTS_DIR


def _pair_analysis(pair_name: str, pair_def: dict) -> dict:
    """Compute shared and unique features."""
    from src.analysis import _load_result

    result = _load_result(f'opposition_{pair_name}.json')
    set_a, set_b = set(), set()

    for entry in result['sides']['a']['prompts']:
        for hit in entry.get('hits', []):
            for f in hit['features']:
                set_a.add(f['feature_id'])

    for entry in result['sides']['b']['prompts']:
        for hit in entry.get('hits', []):
            for f in hit['features']:
                set_b.add(f['feature_id'])

    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        'pair': pair_name,
        'tokens': pair_def['pair'],
        'jaccard': jaccard,
        'n_shared': len(intersection),
        'n_unique_a': len(set_a - set_b),
        'n_unique_b': len(set_b - set_a),
        'shared_features': sorted(intersection),
        'unique_a': sorted(set_a - set_b),
        'unique_b': sorted(set_b - set_a),
    }


def main():
    config = load_config()
    opposition = config.get('opposition', {})
    client = NeuronpediaClient()

    # probe
    print(f'probing {len(opposition)} opposition pairs')
    for name in opposition:
        probe_pair(name, 'opposition', client=client)

    # analyze
    analyses = []
    for name, pair_def in opposition.items():
        a = _pair_analysis(name, pair_def)
        analyses.append(a)
        print(f'  {name}: J={a["jaccard"]:.3f} '
              f'(shared={a["n_shared"]}, '
              f'unique_a={a["n_unique_a"]}, unique_b={a["n_unique_b"]})')

    # save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'opposition_analysis.json'
    out.write_text(json.dumps(analyses, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
