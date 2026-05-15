"""Probing via Neuronpedia SAE activations.

Reads concept definitions from `config/concepts.yaml`
Runs prompt variants through client.py. 
Aggregates per-token feature activations across prompts
Writes structured JSON to `results/`.

Three probe types for concept hierarchy:
  - primitive: single target token across prompt variants
  - synonymy/opposition: paired tokens across matched prompts
  - composite: each token of the composite term across prompts
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

from src.client import NeuronpediaClient

CONFIG_PATH = Path('config/concepts.yaml')
RESULTS_DIR = Path('results')


def _results_dir(layer: int | None = None) -> Path:
    if layer is not None:
        d = RESULTS_DIR / f'layer_{layer}'
    else:
        d = RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config() -> dict:
    """Load concepts.yaml as dict."""
    return yaml.safe_load(CONFIG_PATH.read_text())


def _extract_token_features(
    api_result: dict, target_token: str
) -> list[dict]:
    """Pull feature activations for target token from probe response.

    Handles subword splits: reconstructs running text from API tokens
    and finds all subword positions that span the target. e.g.,
    'digitalization' tokenized as ['Digital', 'ization'] returns
    features for both subwords.
    """
    results = api_result.get('results', [])
    target_lower = target_token.lower()

    # find which token positions belong to target word
    # by reconstructing running text and locating the target span
    match_positions = set()
    running = ''
    for tok_result in results:
        tok_str = tok_result.get('token', '')
        running += tok_str
        # check if any part of the target overlaps this token
        running_lower = running.lower()
        # look for target in the accumulated text
        idx = running_lower.rfind(target_lower)
        if idx != -1 and idx < len(running) and idx + len(target_lower) <= len(running):
            # if target found, mark all tokens whose characters overlap it
            char_pos = 0
            for tr in results:
                ts = tr.get('token', '')
                tok_start = char_pos
                tok_end = char_pos + len(ts)
                if tok_start < idx + len(target_lower) and tok_end > idx:
                    match_positions.add(tr['position'])
                char_pos += len(ts)

    # extract features from matched positions
    hits = []
    for tok_result in results:
        if tok_result['position'] not in match_positions:
            continue
        features = []
        for f in tok_result.get('topFeatures', []):
            feat_meta = f.get('feature', {})
            explanations = feat_meta.get('explanations', [])
            label = explanations[0].get('description', '') if explanations else ''
            features.append({
                'feature_id': f['featureIndex'],
                'activation': f['activationValue'],
                'label': label,
            })
        hits.append({
            'token': tok_result.get('token', ''),
            'position': tok_result['position'],
            'features': features,
        })
    return hits


def _metadata(config: dict) -> dict:
    """Standard metadata block for results."""
    return {
        'model': config['model'],
        'sae': config['sae_id'],
        'layer': config['layer'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def _write_result(name: str, data: dict, layer: int | None = None) -> Path:
    d = _results_dir(layer)
    path = d / f'{name}.json'
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


# -- probe functions --

def probe_primitive(
    concept_name: str,
    client: NeuronpediaClient | None = None,
    layer: int | None = None,
) -> dict:
    config = load_config()
    primitives = config.get('primitives', {})
    if concept_name not in primitives:
        raise ValueError(f'unknown primitive: {concept_name}')

    concept = primitives[concept_name]
    target_token = concept['target_token']
    prompts = concept['prompts']

    if client is None:
        client = NeuronpediaClient()

    per_prompt = []
    for prompt in prompts:
        raw = client.probe(prompt)
        hits = _extract_token_features(raw, target_token)
        per_prompt.append({
            'prompt': prompt,
            'hits': hits,
        })

    result = {
        'concept': concept_name,
        'type': 'primitive',
        'target_token': target_token,
        'prompts': per_prompt,
        'metadata': _metadata(config),
    }
    _write_result(f'primitive_{concept_name}', result, layer=layer)
    return result


def probe_pair(
    pair_name: str,
    pair_type: str,
    client: NeuronpediaClient | None = None,
) -> dict:
    """Probe a synonymy or opposition pair.

    Runs both sides (prompts_a, prompts_b)
    Extracts features for respective target tokens.
    """
    config = load_config()
    pairs = config.get(pair_type, {})
    if pair_name not in pairs:
        raise ValueError(f'unknown {pair_type} pair: {pair_name}')

    pair_def = pairs[pair_name]
    tokens = pair_def['pair']
    prompts_a = pair_def['prompts_a']
    prompts_b = pair_def['prompts_b']

    if client is None:
        client = NeuronpediaClient()

    sides = {}
    for label, target, prompts in [('a', tokens[0], prompts_a), ('b', tokens[1], prompts_b)]:
        per_prompt = []
        for prompt in prompts:
            raw = client.probe(prompt)
            hits = _extract_token_features(raw, target)
            per_prompt.append({'prompt': prompt, 'hits': hits})
        sides[label] = {
            'target_token': target,
            'prompts': per_prompt,
        }

    result = {
        'pair': pair_name,
        'type': pair_type,
        'sides': sides,
        'metadata': _metadata(config),
    }
    _write_result(f'{pair_type}_{pair_name}', result)
    return result


def probe_composite(
    concept_name: str,
    client: NeuronpediaClient | None = None,
    layer: int | None = None,
) -> dict:
    config = load_config()
    composites = config.get('composites', {})
    if concept_name not in composites:
        raise ValueError(f'unknown composite: {concept_name}')

    concept = composites[concept_name]
    label = concept['label']
    prompts = concept['prompts']
    composite_tokens = label.lower().split()

    if client is None:
        client = NeuronpediaClient()

    per_prompt = []
    for prompt in prompts:
        raw = client.probe(prompt)
        token_hits = {}
        for ct in composite_tokens:
            hits = _extract_token_features(raw, ct)
            if hits:
                token_hits[ct] = hits
        per_prompt.append({
            'prompt': prompt,
            'token_hits': token_hits,
        })

    result = {
        'concept': concept_name,
        'type': 'composite',
        'label': label,
        'composite_tokens': composite_tokens,
        'expected_primitives': concept.get('expected_primitives', []),
        'prompts': per_prompt,
        'metadata': _metadata(config),
    }
    _write_result(f'composite_{concept_name}', result, layer=layer)
    return result
