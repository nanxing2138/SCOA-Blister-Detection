import numpy as np
import pytest
import torch
from scoa.basic import basic_augmentation
from scoa.config import load_config
from scoa.data import batch_tensors, letterbox
from scoa.model import build_model, load_pretrained
from scoa.randomness import stream
from scoa.train import lr_at

torch.set_num_threads(2)


def test_training_formula_and_actual_auxiliary_l1_backward():
    model = build_model(load_config('configs/paper.yaml')).train()
    model.head.use_l1 = True
    image = np.full((64, 64, 3), 114., np.float32)
    labels = np.array([[8, 8, 26, 30, 1], [34, 8, 56, 30, 3]], np.float32)
    inputs, targets = batch_tensors([image, image], [labels, labels], 'cpu')
    result = model(inputs, targets)
    assert float(result['l1_loss'].detach()) > 0
    torch.testing.assert_close(result['total_loss'], result['cls_loss'] + result['conf_loss'] + result['iou_loss'] + result['l1_loss'])
    result['total_loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_scheduler_and_final15_epoch_boundary():
    cfg = load_config('configs/paper.yaml')['train']
    assert lr_at(0, 100, cfg) == 0
    assert lr_at(500, 100, cfg) == cfg['lr']
    assert lr_at(28500, 100, cfg) == cfg['lr'] * .05
    assert lr_at(29999, 100, cfg) == cfg['lr'] * .05


def test_basic_final15_disables_geometric_transforms():
    config = load_config('configs/paper.yaml')
    config['train']['input_size'] = 64
    config['basic'].update(hsv_prob=0., flip_prob=0.)
    yy, xx = np.mgrid[:36, :64]
    image = np.repeat(((xx + yy) % 200).astype(np.uint8)[..., None], 3, -1)
    labels = np.array([[10, 8, 30, 30, 2]], np.float32)
    sample = {'image': image, 'labels': labels}
    class ForbiddenMosaicSource:
        def __len__(self):
            raise AssertionError('Mosaic must be disabled in epoch286')
    output, target = basic_augmentation(sample, ForbiddenMosaicSource(), config, stream(42, 286, 1, 'basic'), 286)
    expected, expected_labels, _, _ = letterbox(image, labels, 64)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(target, expected_labels)


def test_shape_matched_pretraining_and_added_cbam_keys(tmp_path):
    config = load_config('configs/paper.yaml')
    model = build_model(config)
    state = model.state_dict()
    key = 'backbone.backbone.stem.conv.conv.weight'
    checkpoint = {'model': {key: torch.ones_like(state[key]), 'head.cls_preds.0.weight': torch.zeros(80, 128, 1, 1)}}
    path = tmp_path / 'shape_matching_fixture.pth'
    torch.save(checkpoint, path)
    config['model']['cbam'] = True
    target = build_model(config)
    info = load_pretrained(target, path)
    assert info['loaded_tensors'] == 1
    assert 'head.cls_preds.0.weight' in info['skipped_keys']
    torch.testing.assert_close(target.state_dict()[key], torch.ones_like(state[key]))
    assert any('cbam' in k for k in info['missing_keys'])
