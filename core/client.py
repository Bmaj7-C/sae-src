"""Neuronpedia API wrapper with disk caching and rate-limit handling.

Wraps two core capabilities:
  - probe: top-k SAE feature activations per token in context
  - steer: generate text with specified SAE features amplified

All responses cached to disk by content hash. 
Retries with exponential backoff on 429 (rate limit) and 5xx errors.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = 'https://www.neuronpedia.org/api'

DEFAULT_MODEL = 'gemma-2-9b-it'
DEFAULT_SOURCE = '9-gemmascope-res-131k'


class NeuronpediaClient:
    """Thin wrapper around Neuronpedia REST API with caching."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        source_id: str = DEFAULT_SOURCE,
        cache_dir: str | Path = 'cache',
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        self.model_id = model_id
        self.source_id = source_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        self.session = requests.Session()
        key = api_key or os.environ.get('NEURONPEDIA_API_KEY')
        if key:
            self.session.headers['X-Api-Key'] = key

    # -- public api --

    def probe(
        self,
        text: str,
        num_results: int = 10,
        density_threshold: float = 0.01,
    ) -> dict[str, Any]:
        """Probe text for top-k SAE feature activations per token.

        Returns raw API response: a list of per-token results, each
        containing token string, position, and ranked feature activations
        with activation values and feature metadata.
        """
        payload = {
            'modelId': self.model_id,
            'source': self.source_id,
            'text': text,
            'numResults': num_results,
            'ignoreBos': True,
            'densityThreshold': density_threshold,
        }
        return self._request('POST', '/search-topk-by-token', payload)

    def steer(
        self,
        prompt: str,
        features: list[dict],
        temperature: float = 0.5,
        n_tokens: int = 300,
        freq_penalty: float = 2.0,
        seed: int = 16,
        strength_multiplier: float = 4.0,
    ) -> dict[str, Any]:
        """Generate text with specified SAE features amplified.

        Args:
            features: list of dicts with keys
                {modelId, layer, index, strength}.
                'layer' corresponds to the SAE source id.
        """
        payload = {
            'prompt': prompt,
            'modelId': self.model_id,
            'features': features,
            'temperature': temperature,
            'n_tokens': n_tokens,
            'freq_penalty': freq_penalty,
            'seed': seed,
            'strength_multiplier': strength_multiplier,
            'steer_method': 'SIMPLE_ADDITIVE',
        }
        return self._request('POST', '/steer', payload)

    def get_feature(self, feature_index: int) -> dict[str, Any]:
        """Fetch metadata for a single SAE feature."""
        path = f'/feature/{self.model_id}/{self.source_id}/{feature_index}'
        return self._request('GET', path)

    # -- internals --

    def _cache_key(self, method: str, path: str, payload: dict | None) -> str:
        """Deterministic hash from request parameters."""
        blob = json.dumps(
            {'method': method, 'path': path, 'payload': payload},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f'{key}.json'

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _write_cache(self, key: str, data: dict) -> None:
        self._cache_path(key).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def _request(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict[str, Any]:
        """Execute request with cache-first strategy and retry on failure."""
        cache_key = self._cache_key(method, path, payload)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        url = BASE_URL + path
        backoff = 1.0

        for _ in range(self.max_retries):
            try:
                if method == 'GET':
                    resp = self.session.get(url, timeout=60)
                else:
                    resp = self.session.post(url, json=payload, timeout=60)

                if resp.status_code == 429 or resp.status_code >= 500:
                    # retryable — back off
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                resp.raise_for_status()
                data = resp.json()
                self._write_cache(cache_key, data)
                return data

            except requests.exceptions.Timeout:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

        raise RuntimeError(
            f'np request failed after {self.max_retries} retries: '
            f'{method} {path}'
        )
