"""Probe synonymy pairs and compute feature overlap."""

import json
import sys
sys.path.insert(0, '.')

from src.client import NeuronpediaClient
from src.probes import load_config, probe_pair
from src.analysis import feature_overlap, RESULTS_DIR


def _pair_overlap(pair_name: str, pair_def: dict) -> dict:
    """Compute overlap between two sides of a probed pair.

    Reuses `analysis.feature_overlap` by extracting
    feature sets from the pair result file directly.
    """
    from src.analysis import _load_result, _collect_feature_ids

    result = _load_result(f'synonymy_{pair_name}.json')
    set_a = set()
    set_b = set()

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
        'n_union': len(union),
        'shared_features': sorted(intersection),
    }


def main():
    config = load_config()
    synonymy = config.get('synonymy', {})
    client = NeuronpediaClient()

    # probe all pairs
    print(f'probing {len(synonymy)} synonymy pairs')
    for name in synonymy:
        probe_pair(name, 'synonymy', client=client)

    # compute overlaps
    overlaps = []
    for name, pair_def in synonymy.items():
        ov = _pair_overlap(name, pair_def)
        overlaps.append(ov)
        print(f'  {name}: J={ov["jaccard"]:.3f} ({ov["n_shared"]} shared)')

    # save overlap summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'synonymy_overlaps.json'
    out.write_text(json.dumps(overlaps, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
