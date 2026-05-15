"""Local analysis of SAE probe results.
Reads structured JSON from `results/`.

Core computations:
  - feature_overlap: Jaccard similarity on top-k feature sets
  - composition_recovery: fraction of primitive features recovered
    in a composite, with bootstrap confidence intervals
  - null_distribution: baseline recovery from unrelated concept pairs
"""

import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np

RESULTS_DIR = Path('results')


def results_dir_for_layer(layer: int | None = None) -> Path:
    if layer is not None:
        return RESULTS_DIR / f'layer_{layer}'
    return RESULTS_DIR


# -- helpers --

def _load_result(filename: str, layer: int | None = None) -> dict:
    path = results_dir_for_layer(layer) / filename
    if not path.exists():
        raise FileNotFoundError(f'result not found: {path}')
    return json.loads(path.read_text())


def _collect_feature_ids(
    result: dict, top_k: int | None = None, min_prompts: int = 1
) -> set[int]:
    """Extract feature ids from a primitive or pair-side result.

    If min_prompts > 1, only keeps features appearing in at least that
    many prompts (majority-vote filtering).
    """
    from collections import Counter
    counts: Counter[int] = Counter()
    prompt_entries = result.get('prompts', [])

    for entry in prompt_entries:
        seen_this_prompt: set[int] = set()
        for hit in entry.get('hits', []):
            feats = hit.get('features', [])
            if top_k is not None:
                feats = feats[:top_k]
            for f in feats:
                seen_this_prompt.add(f['feature_id'])
        for fid in seen_this_prompt:
            counts[fid] += 1

    return {fid for fid, c in counts.items() if c >= min_prompts}


def _collect_composite_feature_ids(
    result: dict, top_k: int | None = None, min_prompts: int = 1
) -> set[int]:
    """Extract feature ids from a composite result (all tokens, all prompts)."""
    from collections import Counter
    counts: Counter[int] = Counter()
    for entry in result.get('prompts', []):
        seen_this_prompt: set[int] = set()
        for token, hits in entry.get('token_hits', {}).items():
            for hit in hits:
                feats = hit.get('features', [])
                if top_k is not None:
                    feats = feats[:top_k]
                for f in feats:
                    seen_this_prompt.add(f['feature_id'])
        for fid in seen_this_prompt:
            counts[fid] += 1
    return {fid for fid, c in counts.items() if c >= min_prompts}


# -- public api --

def feature_overlap(
    concept_a: str,
    concept_b: str,
    top_k: int | None = None,
    min_prompts: int = 1,
    layer: int | None = None,
) -> dict:
    """Jaccard similarity between two primitive concepts' feature sets."""
    result_a = _load_result(f'primitive_{concept_a}.json', layer)
    result_b = _load_result(f'primitive_{concept_b}.json', layer)

    set_a = _collect_feature_ids(result_a, top_k, min_prompts)
    set_b = _collect_feature_ids(result_b, top_k, min_prompts)

    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        'concept_a': concept_a,
        'concept_b': concept_b,
        'top_k': top_k,
        'jaccard': jaccard,
        'n_shared': len(intersection),
        'n_union': len(union),
        'shared_features': sorted(intersection),
        'unique_a': sorted(set_a - set_b),
        'unique_b': sorted(set_b - set_a),
    }


def composition_recovery(
    composite_name: str,
    top_k: int | None = None,
    min_prompts: int = 1,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    layer: int | None = None,
) -> dict:
    """Measure how well a composite recovers its expected primitive features.

    For each expected primitive, computes the fraction of its features
    found in the composite's feature set. Reports per-primitive and
    aggregate recovery with bootstrap confidence intervals.

    Recovery = |primitive_features ∩ composite_features| / |primitive_features|
    """
    comp_result = _load_result(f'composite_{composite_name}.json', layer)
    comp_features = _collect_composite_feature_ids(comp_result, top_k, min_prompts)
    expected = comp_result.get('expected_primitives', [])

    per_primitive = {}
    all_scores = []

    for prim_name in expected:
        try:
            prim_result = _load_result(f'primitive_{prim_name}.json', layer)
        except FileNotFoundError:
            per_primitive[prim_name] = {'ERROR': 'result not found'}
            continue

        prim_features = _collect_feature_ids(prim_result, top_k, min_prompts)
        if not prim_features:
            per_primitive[prim_name] = {'recovery': 0.0, 'n_features': 0}
            continue

        recovered = prim_features & comp_features
        score = len(recovered) / len(prim_features)
        all_scores.append(score)

        per_primitive[prim_name] = {
            'recovery': score,
            'n_features': len(prim_features),
            'n_recovered': len(recovered),
            'recovered_ids': sorted(recovered),
        }

    # bootstrap ci on aggregate recovery
    rng = np.random.default_rng(seed)
    aggregate = float(np.mean(all_scores)) if all_scores else 0.0
    boot_means = []
    if len(all_scores) > 1:
        scores_arr = np.array(all_scores)
        for _ in range(n_bootstrap):
            sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
            boot_means.append(float(np.mean(sample)))
        alpha = (1 - ci) / 2
        ci_low = float(np.quantile(boot_means, alpha))
        ci_high = float(np.quantile(boot_means, 1 - alpha))
    else:
        ci_low, ci_high = aggregate, aggregate

    return {
        'composite': composite_name,
        'top_k': top_k,
        'aggregate_recovery': aggregate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_level': ci,
        'n_bootstrap': n_bootstrap,
        'per_primitive': per_primitive,
        'composite_n_features': len(comp_features),
    }


def null_distribution(
    top_k: int | None = None,
    min_prompts: int = 1,
    n_bootstrap: int = 1000,
    seed: int = 42,
    layer: int | None = None,
) -> dict:
    """Baseline feature overlap from all unrelated primitive pairs.

    Computes Jaccard for every pair of probed primitives. e
    Establishes null expectation i.e., if composition recovery is no
    better than random primitive overlap.
    """
    # find primitive results
    rdir = results_dir_for_layer(layer)
    primitives = sorted([
        p.stem.replace('primitive_', '')
        for p in rdir.glob('primitive_*.json')
    ])

    if len(primitives) < 2:
        return {
            'ERROR': 'min. 2 probed primitives for null distr.',
            'available': primitives,
        }

    pair_scores = []
    pair_details = []

    for a, b in combinations(primitives, 2):
        overlap = feature_overlap(a, b, top_k=top_k, min_prompts=min_prompts, layer=layer)
        pair_scores.append(overlap['jaccard'])
        pair_details.append({
            'pair': [a, b],
            'jaccard': overlap['jaccard'],
        })

    scores_arr = np.array(pair_scores)
    rng = np.random.default_rng(seed)

    # bootstrap ci on mean pairwise overlap
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        boot_means.append(float(np.mean(sample)))

    return {
        'n_primitives': len(primitives),
        'n_pairs': len(pair_scores),
        'top_k': top_k,
        'mean_jaccard': float(np.mean(scores_arr)),
        'std_jaccard': float(np.std(scores_arr)),
        'ci_low': float(np.quantile(boot_means, 0.025)),
        'ci_high': float(np.quantile(boot_means, 0.975)),
        'pairs': pair_details,
    }
