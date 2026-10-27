from copy import deepcopy
from pathlib import Path
import math
import yaml


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(cfg):
    required = {
        'data': {'root', 'train', 'val', 'test', 'image_dir'},
        'model': {'depth', 'width', 'cbam', 'wiou'},
        'train': {'epochs', 'batch_size', 'input_size', 'lr', 'momentum', 'weight_decay', 'nesterov',
                  'warmup_epochs', 'warmup_lr', 'min_lr_ratio', 'no_aug_epochs', 'amp', 'seed', 'workers',
                  'deterministic', 'val_every', 'checkpoint_nms', 'log_every', 'pretrained', 'allow_scratch'},
        'basic': {'mosaic_prob', 'degrees', 'translate', 'scale', 'shear', 'hsv_prob', 'hsv_gains', 'flip_prob'},
        'optical': {'policy', 'probability', 'amplitudes', 'grid', 'variance_threshold', 'epsilon_q', 'quantile',
                    'structure_threshold', 'structure_mode', 'hard_start_epoch', 'difficulty_average', 'smr_weights'},
        'evaluation': {'score_threshold', 'pr_threshold', 'nms_threshold', 'nms', 'max_detections'},
    }
    if not isinstance(cfg, dict) or set(cfg) != set(required) | {'name', 'output'}:
        raise ValueError('Configuration requires name/output and exactly data/model/train/basic/optical/evaluation sections')
    for section, fields in required.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) != fields:
            raise ValueError(f'Invalid/missing/unknown keys in {section}: expected {sorted(fields)}')
    def number(section, key, low=None, high=None, integer=False, positive=False):
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{section}.{key} must be finite numeric')
        if integer and not isinstance(value, int):
            raise ValueError(f'{section}.{key} must be an integer')
        if (low is not None and value < low) or (high is not None and value > high) or (positive and value <= 0):
            raise ValueError(f'Out of range: {section}.{key}')
    for key in ('epochs', 'batch_size', 'input_size', 'val_every', 'log_every'):
        number('train', key, integer=True, positive=True)
    for key in ('warmup_epochs', 'no_aug_epochs', 'workers', 'seed'):
        number('train', key, low=0, integer=True)
    if cfg['train']['input_size'] < 32 or cfg['train']['input_size'] % 32:
        raise ValueError('input_size must be a positive multiple of 32')
    if cfg['train']['warmup_epochs'] + cfg['train']['no_aug_epochs'] > cfg['train']['epochs']:
        raise ValueError('warmup_epochs + no_aug_epochs cannot exceed epochs')
    number('train', 'lr', positive=True)
    for key in ('momentum', 'min_lr_ratio'):
        number('train', key, low=0, high=1)
    number('train', 'weight_decay', low=0)
    number('train', 'warmup_lr', low=0)
    if cfg['train']['nesterov'] and cfg['train']['momentum'] <= 0:
        raise ValueError('Nesterov requires positive momentum')
    if cfg['train']['checkpoint_nms'] not in {'standard', 'diou'}:
        raise ValueError('Unknown checkpoint_nms')
    for section, keys in [('train', ('nesterov', 'amp', 'deterministic', 'allow_scratch')), ('model', ('cbam', 'wiou'))]:
        for key in keys:
            if not isinstance(cfg[section][key], bool):
                raise ValueError(f'{section}.{key} must be true/false')
    for key in ('depth', 'width'):
        number('model', key, positive=True)
    for key in ('mosaic_prob', 'hsv_prob', 'flip_prob'):
        number('basic', key, low=0, high=1)
    for key in ('degrees', 'translate', 'shear'):
        number('basic', key, low=0)
    scale, gains = cfg['basic']['scale'], cfg['basic']['hsv_gains']
    if not isinstance(scale, list) or len(scale) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in scale) or scale[0] > scale[1]:
        raise ValueError('basic.scale requires two ordered positive finite numbers')
    if not isinstance(gains, list) or len(gains) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in gains):
        raise ValueError('basic.hsv_gains requires three finite nonnegative numbers')
    optical = cfg['optical']
    if optical['policy'] not in {'basic', 'A', 'B', 'C', 'D', 'SMR-B', 'legacy', 'AM-photo', 'AugMax-P'}:
        raise ValueError('Unknown optical policy')
    if optical['structure_mode'] not in {'whole', 'mean', 'quantile'} or optical['difficulty_average'] not in {'class', 'instance'}:
        raise ValueError('Unknown structure_mode or difficulty_average')
    for key in ('probability', 'structure_threshold'):
        number('optical', key, low=0, high=1)
    number('optical', 'quantile', positive=True, high=1)
    number('optical', 'epsilon_q', positive=True)
    number('optical', 'variance_threshold', low=0)
    for key in ('grid', 'hard_start_epoch'):
        number('optical', key, integer=True, positive=True)
    amplitudes = optical['amplitudes']
    if not isinstance(amplitudes, list) or not amplitudes or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in amplitudes):
        raise ValueError('optical.amplitudes must contain finite positive numbers')
    if optical['policy'] == 'SMR-B' and (not optical['smr_weights'] or amplitudes != [.5, 1., 1.5]):
        raise ValueError('SMR-B requires fitted weights and the three paper amplitudes')
    if cfg['evaluation']['nms'] not in {'standard', 'diou'}:
        raise ValueError('Unknown evaluation.nms')
    for key in ('score_threshold', 'pr_threshold', 'nms_threshold'):
        number('evaluation', key, low=0, high=1)
    number('evaluation', 'max_detections', integer=True, positive=True)
    return cfg


def load_config(path, overrides=(), _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError('Cyclic config extends chain')
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('YAML configuration must be a mapping')
    parent = cfg.pop('extends', None)
    if parent:
        cfg = merge(load_config(path.parent / parent, _seen=(*_seen, path)), cfg)
    for override in overrides:
        key, value = override.split('=', 1)
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f'Unknown configuration key: {key}')
        target[parts[-1]] = yaml.safe_load(value)
    return validate_config(cfg)
