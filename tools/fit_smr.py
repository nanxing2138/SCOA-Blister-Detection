"""Fit positive amplitude weights offline from D logs; never copy current-image choices."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def fit(events, steps=5000):
    target = np.bincount([e['selected_index'] for e in events], minlength=3).astype(float) / len(events)
    mask = np.zeros((len(events), 3), dtype=float)
    for index, event in enumerate(events):
        mask[index, event['feasible']] = 1
    logits = np.zeros(3)
    for _ in range(steps):
        weights = np.exp(logits - logits.max()).clip(1e-6)
        probabilities = mask * weights
        probabilities /= probabilities.sum(1, keepdims=True)
        difference = probabilities.mean(0) - target
        logits -= 2 * difference
        logits -= logits.mean()
        if abs(difference).max() < 1e-6:
            break
    weights = np.exp(logits - logits.max()).clip(1e-6)
    expected = mask * weights
    expected = (expected / expected.sum(1, keepdims=True)).mean(0)
    return {'weights': weights.tolist(), 'target_frequencies': target.tolist(),
            'fitted_frequencies': expected.tolist(), 'max_difference_pp': float(100 * abs(expected - target).max()),
            'events': len(events), 'origin': 'new_offline_fit_not_paper_weights'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    strata = defaultdict(list)
    for filename in args.logs:
        for line in Path(filename).read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event['policy'] != 'D':
                raise ValueError('Fit only against D policy logs')
            if event['selected_index'] is None or not event.get('feasible'):
                continue
            stage = min(4, (event['epoch'] - 1) // 60)
            stratum = 'defect' if event['has_defect'] else 'no_defect'
            strata[f'{event["seed"]}/{stage}/{stratum}'].append(event)
    output = {key: fit(events) for key, events in strata.items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'Fitted {len(output)} strata. Missing stage/seed strata remain unavailable.')


if __name__ == '__main__':
    main()
