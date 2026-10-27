import argparse
import hashlib
import json
from pathlib import Path
import torch
from .config import load_config
from .checkpoints import load_checkpoint
from .data import CocoDataset, letterbox
from .difficulty import eval_mode, image_tensor, raw_forward
from .metrics import evaluate_records
from .model import build_model
from .postprocess import postprocess


@torch.no_grad()
def evaluate_model(model, dataset, config, device, image_ids=None):
    records = []
    with eval_mode(model):
        for index in range(len(dataset)):
            sample = dataset[index]
            if image_ids is not None and sample['id'] not in image_ids:
                continue
            image, labels, _, _ = letterbox(sample['image'], sample['labels'], config['train']['input_size'])
            raw, _, _ = raw_forward(model, image_tensor(image.astype('float32') / 255, device))
            prediction = postprocess(raw, config['evaluation'])[0].cpu().tolist()
            records.append({'image_id': sample['id'], 'file_name': sample['file_name'],
                            'gt': labels.tolist(), 'predictions': prediction})
    if image_ids is not None and {r['image_id'] for r in records} != image_ids:
        raise ValueError('Requested stratum IDs include missing images')
    if not records:
        raise ValueError('Evaluation image set is empty')
    result = evaluate_records(records, config['evaluation']['score_threshold'], config['evaluation']['pr_threshold'])
    identity = json.dumps(sorted(r['image_id'] for r in records), separators=(',', ':')).encode('utf-8')
    result.update(synthetic=bool(dataset.annotation.get('synthetic', False)), coordinate_frame='letterboxed_input',
                  input_size=config['train']['input_size'], nms=config['evaluation']['nms'],
                  evaluated_image_ids=sorted(r['image_id'] for r in records),
                  evaluated_image_ids_sha256=hashlib.sha256(identity).hexdigest(),
                  nms_threshold=config['evaluation']['nms_threshold'],
                  max_detections=config['evaluation']['max_detections'],
                  annotation_sha256=hashlib.sha256(dataset.annotation_path.read_bytes()).hexdigest())
    return result, records


def restore_model(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or not {'config', 'ema', 'epoch'} <= checkpoint.keys():
        raise ValueError('Expected this project\'s trained checkpoint with config/ema/epoch; COCO yolox_s.pth is only for train.pretrained')
    config = checkpoint['config']
    model = build_model(config)
    model.load_state_dict(checkpoint['ema'], strict=True)
    return model.to(device).eval(), config, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--ids', help='JSON list of image IDs for a fixed optical/external stratum')
    parser.add_argument('--output', required=True)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config = load_config(args.config, args.set)
    model, saved, checkpoint = restore_model(args.checkpoint, args.device)
    if saved['model'] != config['model'] or saved['train']['input_size'] != config['train']['input_size']:
        raise ValueError('Evaluation model/input_size must match the selected checkpoint')
    dataset = CocoDataset(config, args.split)
    ids = set(json.loads(Path(args.ids).read_text(encoding='utf-8'))) if args.ids else None
    result, records = evaluate_model(model, dataset, config, args.device, ids)
    result.update(name=config['name'], seed=saved['train']['seed'], selected_epoch=checkpoint['epoch'],
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  training_configuration={'model': saved['model'], 'optical': saved['optical'], 'train': saved['train']},
                  split=args.split, status='executed')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    output.with_suffix('.predictions.json').write_text(json.dumps(records, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    main()
