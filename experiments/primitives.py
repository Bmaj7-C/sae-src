"""Probe all primitive concepts defined in `concepts.yaml`."""

import sys
sys.path.insert(0, '.')

from src.client import NeuronpediaClient
from src.probes import load_config, probe_primitive


def main():
    config = load_config()
    primitives = list(config.get('primitives', {}).keys())
    client = NeuronpediaClient()

    print(f'probing {len(primitives)} primitives')
    for name in primitives:
        probe_primitive(name, client=client)

    print('done')


if __name__ == '__main__':
    main()
