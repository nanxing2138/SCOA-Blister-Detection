from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from scoa.augment import augmax
from scoa.config import load_config
from scoa.data import CocoDataset, batch_tensors
from scoa.evaluate import restore_model
from scoa.geometry import bboxes_iou
from scoa.model import StableSquaredIoULoss, build_model
from tools.compare_results import validate_evaluation
from tools.make_synthetic import generate

torch.set_num_threads(2)


def test_fp16_large_boxes_do_not_overflow_area_or_loss():
    xyxy = torch.tensor([[0., 0., 640., 640.]], dtype=torch.float16)
    assert float(bboxes_iou(xyxy, xyxy)[0, 0]) == 1.
    box = torch.tensor([[320., 320., 640., 640.]], dtype=torch.float16, requires_grad=True)
    loss = StableSquaredIoULoss()(box, box.detach())
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all()
    assert float(loss.sum()) == 0.
    loss.sum().backward()
    assert torch.isfinite(box.grad).all()


def test_training_decode_is_float32_even_with_fp16_head_output():
    model = build_model(load_config('configs/paper.yaml'))
    raw = torch.zeros(1, 9, 1, 1, dtype=torch.float16)
    raw[:, 2:4] = math.log(640 / 32)
    raw.requires_grad_(True)
    decoded, grid = model.head.get_output_and_grid(raw, 0, 32, raw.type())
    assert decoded.dtype == torch.float32 and grid.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    decoded.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_empty_annotations_have_finite_real_training_backward():
    model = build_model(load_config('configs/paper.yaml')).train()
    images = [np.full((64, 64, 3), 114., np.float32)] * 2
    labels = [np.empty((0, 5), np.float32)] * 2
    x, target = batch_tensors(images, labels, 'cpu')
    result = model(x, target)
    assert torch.isfinite(result['total_loss'])
    result['total_loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_augmax_nonfinite_original_returns_original_not_random_mixture():
    model = build_model(load_config('configs/paper.yaml')).eval()
    with torch.no_grad():
        model.head.reg_preds[0].bias.fill_(float('inf'))
    image = np.full((64, 64, 3), .3, np.float32)
    labels = np.array([[8, 8, 26, 28, 1]], np.float32)
    chains = [np.full_like(image, value) for value in (.4, .5, .6)]
    result, reason = augmax(image, chains, labels, [.2, .3, .5], .5, model, 'cpu')
    assert reason == 'nonfinite_search'
    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize('override', ['train.epochs=0', 'train.lr=.nan', 'train.batch_size=-1',
    'train.no_aug_epochs=301', 'optical.probability=1.1', 'optical.epsilon_q=0',
    'optical.structure_mode=typo', 'optical.difficulty_average=typo', 'evaluation.nms=typo'])
def test_invalid_configs_fail_before_training(override):
    with pytest.raises(ValueError):
        load_config('configs/paper.yaml', [override])


def test_unknown_yaml_keys_and_extends_cycle_are_rejected(tmp_path):
    cfg = load_config('configs/paper.yaml')
    cfg['train']['batch_szie'] = 4
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)
    a, b = tmp_path/'a.yaml', tmp_path/'b.yaml'
    a.write_text('extends: b.yaml', encoding='utf-8')
    b.write_text('extends: a.yaml', encoding='utf-8')
    with pytest.raises(ValueError, match='Cyclic'):
        load_config(a)


def test_dataset_annotation_identity_and_path_validation(tmp_path):
    generate(tmp_path)
    cfg = load_config('configs/smoke.yaml')
    cfg['data']['root'] = str(tmp_path)
    data = CocoDataset(cfg)
    item = deepcopy(data.images[0])
    item['file_name'] = '../escape.png'
    with pytest.raises(ValueError, match='inside'):
        data.image_path(item)
    item['file_name'] = 'train_1.png'
    assert data.image_path(item).is_file()
    annotation = data.annotation
    annotation['annotations'].append(deepcopy(annotation['annotations'][0]))
    (tmp_path/'annotations/train.json').write_text(json.dumps(annotation), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate annotation'):
        CocoDataset(cfg)


def diagnostic_record(name='D', nms='standard', ids=(1, 2)):
    cfg = load_config('configs/integrated.yaml' if name.startswith('integrated') else 'configs/paper.yaml')
    return {'name': name, 'synthetic': False, 'split': 'test', 'evaluated_image_ids': list(ids),
            'evaluated_image_ids_sha256': hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest(),
            'nms': nms, 'nms_threshold': .45, 'max_detections': 100, 'input_size': 640,
            'coordinate_frame': 'letterboxed_input', 'protocol': {'ap_score_threshold': .001,
            'pr_score_threshold': .25, 'recall_points': 101}, 'metrics': {'AP50_95': .5},
            'annotation_sha256': 'DIAGNOSTIC_FIXTURE_ONLY',
            'training_configuration': {'model': cfg['model'], 'optical': cfg['optical']}}


def test_paper_comparison_checks_subset_and_nms_and_model_identity():
    first, second = diagnostic_record(ids=(1, 2)), diagnostic_record(ids=(1,))
    assert validate_evaluation(first) != validate_evaluation(second)
    with pytest.raises(ValueError, match='protocol'):
        validate_evaluation(diagnostic_record(name='integrated', nms='standard'))
    assert validate_evaluation(diagnostic_record(name='integrated-standard', nms='standard'))
    first['training_configuration']['model']['cbam'] = True
    with pytest.raises(ValueError, match='actual trained'):
        validate_evaluation(first)


def test_wrong_checkpoint_format_has_actionable_error(tmp_path):
    path = tmp_path/'coco_fixture.pth'
    torch.save({'model': {}}, path)
    with pytest.raises(ValueError, match='train.pretrained'):
        restore_model(path, 'cpu')
