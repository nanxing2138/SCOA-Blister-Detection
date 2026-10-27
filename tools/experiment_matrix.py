"""Build a reviewable experiment manifest. Optional execution uses argv, no shell interpolation."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from scoa.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['core', 'components', 'auxiliary'], default='core')
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='experiments/generated')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = load_config(args.config)
    variants = []
    if args.suite == 'core':
        variants = [(p, {'optical': {'policy': p}}) for p in ('basic', 'A', 'B', 'C', 'D')]
    elif args.suite == 'components':
        for scoa, cbam, wiou in itertools.product((False, True), repeat=3):
            name = f'{"D" if scoa else "basic"}_C{int(cbam)}W{int(wiou)}'
            variants.append((name, {'model': {'cbam': cbam, 'wiou': wiou}, 'optical': {'policy': 'D' if scoa else 'basic'}}))
    else:
        variants = [(p, {'optical': {'policy': p}}) for p in ('legacy', 'AM-photo', 'AugMax-P')]
        variants += [('S1', {'optical': {'policy': 'B', 'structure_mode': 'whole'}}),
                     ('S2', {'optical': {'policy': 'B', 'structure_mode': 'mean'}}),
                     ('D-instance', {'optical': {'policy': 'D', 'difficulty_average': 'instance'}})]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, updates in variants:
        for seed in (42, 3407, 2026):
            cfg = deepcopy(base)
            for section, value in updates.items():
                cfg[section].update(value)
            cfg.update(name=name, output=f'runs/{name}')
            cfg['train'].update(seed=seed, pretrained=args.pretrained)
            path = output / f'{name}_{seed}.yaml'
            path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            train = [sys.executable, '-m', 'scoa.train', '--config', str(path), '--device', args.device]
            evaluations = []
            for nms in (('standard', 'diou') if args.suite == 'components' else ('standard',)):
                evaluations.append([sys.executable, '-m', 'scoa.evaluate', '--config', str(path), '--checkpoint',
                                    f'runs/{name}/seed_{seed}/best.pth', '--device', args.device,
                                    '--set', f'evaluation.nms={nms}', '--output', f'runs/{name}/seed_{seed}/test_{nms}.json'])
            jobs.append({'name': name, 'seed': seed, 'train': train, 'evaluations': evaluations})
    (output / f'{args.suite}_manifest.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    print(f'{len(jobs)} training identities prepared; post-processing reuses best checkpoints.')
    if args.execute:
        for job in jobs:
            subprocess.run(job['train'], check=True)
            for evaluation in job['evaluations']:
                subprocess.run(evaluation, check=True)


if __name__ == '__main__':
    main()
