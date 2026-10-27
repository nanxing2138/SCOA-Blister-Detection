"""Verify the real bundled COCO checkpoint, shape-matched transfer and CPU forward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scoa.config import load_config
from scoa.checkpoints import load_checkpoint
from scoa.difficulty import raw_forward
from scoa.model import build_model, load_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolox_s.pth')
    parser.add_argument('--output', default='validation_records/20261027/pretrained_verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = Path(args.weights)
    source = json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Checkpoint does not match recorded SHA256')
    checkpoint = load_checkpoint(path)
    weights = checkpoint.get('model', checkpoint)
    if any(weights[f'head.cls_preds.{level}.bias'].shape != (80,) for level in range(3)):
        raise ValueError('Expected the official 80-class COCO classification heads')
    results = {}
    for name, cfg_path in [('core', 'configs/paper.yaml'), ('integrated', 'configs/integrated.yaml')]:
        torch.manual_seed(42)
        config = load_config(cfg_path)
        model = build_model(config).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()
                  if key not in weights or weights[key].shape != value.shape}
        transfer = load_pretrained(model, path)
        after = model.state_dict()
        for key, value in after.items():
            if key in weights and weights[key].shape == value.shape:
                if not torch.equal(value, weights[key]):
                    raise AssertionError(f'Matching tensor was not loaded: {key}')
            elif not torch.equal(value, before[key]):
                raise AssertionError(f'Unmatched tensor initialization was overwritten: {key}')
        expected_skips = {f'head.cls_preds.{level}.{kind}' for level in range(3) for kind in ('weight', 'bias')}
        if set(transfer['skipped_keys']) != expected_skips:
            raise AssertionError(f'Unexpected source incompatibility: {transfer["skipped_keys"]}')
        with torch.no_grad():
            raw, _, _ = raw_forward(model, torch.full((1, 3, 640, 640), 114.0))
        if raw.shape != (1, 8400, 9) or not torch.isfinite(raw).all():
            raise AssertionError('Transferred model forward is invalid')
        results[name] = dict(transfer, output_shape=list(raw.shape), finite=True,
                             unmatched_tensors_preserved=True, matched_tensors_verified=True)
    result = {'status': 'passed', 'scope': 'real_COCO_pretraining_transfer_not_blister_accuracy',
              'source': source, 'torch_version': torch.__version__, 'models': results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: {'loaded_tensors': r['loaded_tensors'], 'missing_tensors': len(r['missing_keys']),
                             'skipped_source_tensors': len(r['skipped_keys']), 'output_shape': r['output_shape']}
                      for name, r in results.items()}))


if __name__ == '__main__':
    main()
