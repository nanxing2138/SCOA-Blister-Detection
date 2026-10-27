"""Actual uninterrupted-vs-resumed integrated training, using synthetic data only."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scoa.checkpoints import load_checkpoint
from scoa.config import load_config
from scoa.train import train
from tools.make_synthetic import generate


def same_state(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/resume_check')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    generate(output/'data')
    cfg = load_config('configs/smoke.yaml')
    cfg['data']['root'] = str(output/'data')
    cfg['train'].update(epochs=2, no_aug_epochs=1, workers=args.workers)
    cfg['model'].update(cbam=True, wiou=True)
    cfg['output'] = str(output/'full')
    with (output/'training.log').open('w', encoding='utf-8') as handle, redirect_stdout(handle):
        full = train(cfg, device='cpu')
        cfg['output'] = str(output/'resumed')
        interrupted = train(cfg, device='cpu', stop_after_epoch=1)
        train(cfg, device='cpu', resume=interrupted/'latest.pth')
    a, b = load_checkpoint(full/'latest.pth'), load_checkpoint(interrupted/'latest.pth')
    fields = ('model', 'ema', 'ema_updates', 'optimizer', 'scaler', 'epoch', 'global_step', 'best_ap50', 'torch_rng')
    checks = {key: bool(same_state(a[key], b[key])) for key in fields}
    if not all(checks.values()):
        raise AssertionError(f'Resume equivalence failed: {checks}')
    report = {'status': 'passed', 'synthetic': True, 'comparison': '2 epochs continuous vs 1 epoch + resume',
              'workers': args.workers, 'model': 'CBAM_WIoU_SCOA', 'checks_exact_equal': checks,
              'successful_updates_each_run': a['ema_updates'], 'scope': 'CPU epoch-boundary determinism only'}
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
