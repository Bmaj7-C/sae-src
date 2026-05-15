"""Run steered generation under each condition from `concepts.yaml`.

- Save incrementally after each condition 
- Resumes from partial results on restart
- Handles rate limits with inter-condition cooldown and per-generation retry.
"""

import json
import time
import sys
sys.path.insert(0, '.')

from src.client import NeuronpediaClient, DEFAULT_MODEL, DEFAULT_SOURCE
from src.probes import load_config
from src.analysis import RESULTS_DIR

OUT_PATH = RESULTS_DIR / 'steering_results.json'


def _load_partial() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


def _save(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))


def main():
    config = load_config()
    steer_cfg = config.get('steering', {})
    prompt = steer_cfg['prompt']
    conditions = steer_cfg.get('conditions', {})
    n_gen = steer_cfg.get('n_generations', 50)
    temperature = steer_cfg.get('temperature', 0.5)

    client = NeuronpediaClient()
    all_results = _load_partial()

    cond_names = list(conditions.keys())
    for ci, cond_name in enumerate(cond_names):
        if cond_name in all_results:
            print(f'skip {cond_name} (already complete)')
            continue

        cond = conditions[cond_name]
        feature_ids = cond.get('features', [])
        strength = cond.get('strength', 0)

        features = [
            {
                'modelId': DEFAULT_MODEL,
                'layer': DEFAULT_SOURCE,
                'index': fid,
                'strength': float(strength),
            }
            for fid in feature_ids
        ]

        if not features:
            # baseline: neutral feature at strength 0
            features = [{
                'modelId': DEFAULT_MODEL,
                'layer': DEFAULT_SOURCE,
                'index': 0,
                'strength': 0.0,
            }]

        print(f'condition: {cond_name} ({len(feature_ids)} features, '
              f'strength={strength}, n={n_gen})')

        generations = []
        for i in range(n_gen):
            try:
                result = client.steer(
                    prompt=prompt,
                    features=features,
                    temperature=temperature,
                    seed=i,
                )
            except Exception as e:
                print(f'  {cond_name} {i+1}/{n_gen} failed: {e}',
                      flush=True)
                time.sleep(120)
                try:
                    result = client.steer(
                        prompt=prompt,
                        features=features,
                        temperature=temperature,
                        seed=i,
                    )
                except Exception:
                    print(f'  {cond_name} {i+1}/{n_gen} skipped', flush=True)
                    continue

            generations.append({
                'seed': i,
                'steered': result.get('STEERED', ''),
                'default': result.get('DEFAULT', ''),
            })
            print(f'  {cond_name} {i+1}/{n_gen}', flush=True)

        all_results[cond_name] = {
            'features': feature_ids,
            'strength': strength,
            'n_generations': len(generations),
            'generations': generations,
        }
        _save(all_results)
        print(f'saved {cond_name} ({len(generations)}/{n_gen} generations)')

        # cooldown between conditions
        if ci < len(cond_names) - 1:
            print('cooldown 60s', flush=True)
            time.sleep(60)

    print('done')


if __name__ == '__main__':
    main()
