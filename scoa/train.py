import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .basic import basic_augmentation
from .config import load_config, validate_config
from .checkpoints import load_checkpoint
from .data import CocoDataset, batch_tensors, collate_raw, letterbox
from .evaluate import evaluate_model
from .model import ModelEMA, build_model, load_pretrained, make_optimizer
from .randomness import stream
from .selection import Selector


def lr_at(step, steps_per_epoch, cfg):
    warmup = cfg['warmup_epochs'] * steps_per_epoch
    total = cfg['epochs'] * steps_per_epoch
    tail = cfg['no_aug_epochs'] * steps_per_epoch
    if warmup and step <= warmup:
        return (cfg['lr'] - cfg['warmup_lr']) * (step / warmup) ** 2 + cfg['warmup_lr']
    minimum = cfg['lr'] * cfg['min_lr_ratio']
    if step >= total - tail:
        return minimum
    return minimum + .5 * (cfg['lr'] - minimum) * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup - tail)))


def write_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def comparison_config(config):
    value = deepcopy(config)
    value.pop('output', None)
    value['data'].pop('root', None)
    value['train'].pop('pretrained', None)
    value['train'].pop('workers', None)
    return value


def train(config, device='cuda', resume=None, stop_after_epoch=None):
    validate_config(config)
    cfg = config['train']
    if stop_after_epoch is not None and (not isinstance(stop_after_epoch, int) or stop_after_epoch < 1):
        raise ValueError('stop_after_epoch must be a positive epoch number')
    output = Path(config['output']) / f'seed_{cfg["seed"]}'
    if any((output / name).exists() for name in ('latest.pth', 'best.pth', 'epochs.jsonl', 'candidates.jsonl')) and not resume:
        raise FileExistsError(f'{output}/latest.pth exists; use --resume or another output')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if str(device).startswith('cuda'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg['deterministic']
    torch.use_deterministic_algorithms(cfg['deterministic'])
    dataset, validation = CocoDataset(config, 'train'), CocoDataset(config, 'val')
    if not len(dataset) or not len(validation):
        raise ValueError('Training/validation splits must be nonempty')
    signatures = {split: hashlib.sha256(data.annotation_path.read_bytes()).hexdigest()
                  for split, data in [('train', dataset), ('val', validation)]}
    model = build_model(config).to(device)
    pretrained = None
    if not resume:
        if cfg['pretrained']:
            pretrained = load_pretrained(model, cfg['pretrained'])
        elif not cfg['allow_scratch']:
            raise ValueError('Paper protocol needs COCO pretraining: set train.pretrained=weights/yolox_s.pth; scratch requires explicit train.allow_scratch=true')
    optimizer = make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'] and str(device).startswith('cuda'))
    ema = ModelEMA(model)
    start, best_ap, global_step = 1, float('-inf'), 0
    if resume:
        checkpoint = load_checkpoint(resume)
        if comparison_config(checkpoint['config']) != comparison_config(config) or checkpoint['data_sha256'] != signatures:
            raise ValueError('Resume configuration or annotation hashes changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        ema.ema.load_state_dict(checkpoint['ema'], strict=True)
        ema.updates = checkpoint['ema_updates']
        start, best_ap, global_step = checkpoint['epoch'] + 1, checkpoint['best_ap50'], checkpoint['global_step']
        if stop_after_epoch is not None and stop_after_epoch < start:
            raise ValueError('stop_after_epoch does not leave an epoch after this checkpoint')
        existing_latest = output / 'latest.pth'
        if existing_latest.exists():
            existing = load_checkpoint(existing_latest)
            if existing['epoch'] > checkpoint['epoch']:
                raise ValueError('Resume would rewind an existing run; choose a fresh output directory')
            del existing
        destination_best = output / 'best.pth'
        if not destination_best.exists() and math.isfinite(best_ap):
            source_best = Path(resume).resolve().parent / 'best.pth'
            if not source_best.exists():
                raise ValueError('Keep best.pth beside latest.pth when resuming so selected weights are not lost')
            shutil.copy2(source_best, destination_best)
        pretrained = checkpoint.get('pretrained')
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if str(device).startswith('cuda') and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        del checkpoint
    selector = Selector(config)
    steps_per_epoch = math.ceil(len(dataset) / cfg['batch_size'])
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    metadata = {'synthetic': bool(dataset.annotation.get('synthetic', False)), 'config': config,
                'data_sha256': signatures, 'pretrained': pretrained, 'torch_version': torch.__version__,
                'method_status': 'paper_driven_reconstruction', 'hardware_probed': False}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'candidates.jsonl').open('a', encoding='utf-8') as candidate_log, (output / 'epochs.jsonl').open('a', encoding='utf-8') as epoch_log:
        for epoch in range(start, min(cfg['epochs'], stop_after_epoch or cfg['epochs']) + 1):
            generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
            loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, generator=generator,
                                collate_fn=collate_raw, num_workers=cfg['workers'], drop_last=False)
            model.train()
            model.head.use_l1 = epoch > cfg['epochs'] - cfg['no_aug_epochs']
            elapsed, loss_sum, updates, skipped = 0.0, 0.0, 0, 0
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            for batch_index, samples in enumerate(loader):
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
                begin = time.perf_counter()
                images, targets = [], []
                for sample in samples:
                    image_id = sample['id']
                    if selector.optical_trigger(epoch, image_id):
                        original, labels, valid, _ = letterbox(sample['image'], sample['labels'], cfg['input_size'])
                        selected, record = selector.select(original.astype(np.float32) / 255, labels, valid,
                                                            epoch, image_id, ema.ema, model, device)
                        image = selected * 255
                        write_jsonl(candidate_log, record)
                    else:
                        image, labels = basic_augmentation(sample, dataset, config, stream(cfg['seed'], epoch, image_id, 'basic'), epoch)
                    images.append(image)
                    targets.append(labels)
                inputs, labels = batch_tensors(images, targets, device)
                optimizer.zero_grad(set_to_none=True)
                learning_rate = lr_at(global_step, steps_per_epoch, cfg)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    losses = model(inputs, labels)
                    loss = losses['total_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, batch {batch_index}; no result claimed')
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
                    updates += 1
                else:
                    skipped += 1
                global_step += 1
                loss_sum += float(loss.detach())
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - begin
                if (batch_index + 1) % cfg['log_every'] == 0:
                    print(f'epoch={epoch}/{cfg["epochs"]} batch={batch_index+1}/{steps_per_epoch} loss={float(loss.detach()):.6f} lr={learning_rate:.8f}', flush=True)
            summary = {'epoch': epoch, 'seed': cfg['seed'], 'mean_loss': loss_sum / steps_per_epoch,
                       'training_step_seconds': elapsed, 'successful_updates': updates, 'amp_skipped_updates': skipped,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if str(device).startswith('cuda') else None}
            improved = False
            if epoch % cfg['val_every'] == 0 or epoch == cfg['epochs']:
                selection_config = deepcopy(config)
                selection_config['evaluation']['nms'] = cfg['checkpoint_nms']
                result, _ = evaluate_model(ema.ema, validation, selection_config, device)
                summary['validation'] = result['metrics']
                summary['checkpoint_nms'] = cfg['checkpoint_nms']
                ap = result['metrics']['AP50']
                if ap is None:
                    raise ValueError('Validation AP50 is undefined')
                improved = ap > best_ap
                if improved:
                    best_ap = ap
            write_jsonl(epoch_log, summary)
            checkpoint = {'model': model.state_dict(), 'ema': ema.ema.state_dict(), 'ema_updates': ema.updates,
                          'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                          'global_step': global_step, 'best_ap50': best_ap, 'config': config,
                          'data_sha256': signatures, 'pretrained': pretrained, 'torch_rng': torch.get_rng_state(),
                          'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                          'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
            atomic_save(checkpoint, output / 'latest.pth')
            if improved:
                atomic_save(checkpoint, output / 'best.pth')
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-epoch', type=int, help='Stop at an epoch boundary without changing the total-epoch schedule')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train(load_config(args.config, args.set), args.device, args.resume, args.stop_after_epoch)


if __name__ == '__main__':
    main()
