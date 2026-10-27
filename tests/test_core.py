from copy import deepcopy
import numpy as np
import pytest
import torch
from scoa.config import load_config
from scoa.data import batch_tensors, letterbox
from scoa.difficulty import FixedAssignmentScorer, difficulty, raw_forward
from scoa.metrics import evaluate_records
from scoa.model import CBAM_SITES, ModelEMA, WIoU3, build_model, make_optimizer
from scoa.optics import draw_pool, glare_at
from scoa.postprocess import suppress, postprocess
from scoa.randomness import stream
from scoa.selection import Selector
from scoa.structure import (InvalidDefectBox, box_pixels, contrast_structure,
                            luminance, order_statistic, photometric, reference_regions)

torch.set_num_threads(2)


def test_letterbox_bgr_scale_and_padding():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:, :, 0] = 255
    labels = np.array([[300, 300, 600, 600, 3]], np.float32)
    canvas, target, valid, ratio = letterbox(image, labels, 640)
    assert valid == (0, 0, 640, 360)
    assert ratio == 1 / 3
    assert np.array_equal(target[0], [100, 100, 200, 200, 3])
    assert np.all(canvas[360:] == 114)
    assert np.all(canvas[:360, :, 0] == 255)


def test_pixel_centres_and_half_open_bounds():
    yy, xx = box_pixels([.5, .5, 2.5, 2.5], (0, 0, 4, 4))
    assert set(zip(xx, yy)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    with pytest.raises(InvalidDefectBox):
        reference_regions(np.zeros((4, 4, 3)), [[0, 0, .9, .9]], (0, 0, 4, 4))


def test_physical_rgb_luminance():
    assert luminance(np.array([[[1., 0., 0.]]]))[0, 0] == pytest.approx(.114)
    assert luminance(np.array([[[0., 0., 1.]]]))[0, 0] == pytest.approx(.299)


def test_quantile_is_fourth_lowest_without_interpolation():
    assert order_statistic(list(range(16)), .25) == 3
    assert order_statistic([.1, .9, .7], .25) == .1


def test_population_covariance_and_brightness_shift():
    a = np.array([[0., .2], [.4, .6]])
    region = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert contrast_structure(a, a + .1, region) == pytest.approx(1.)
    assert contrast_structure(a, .6 - a, region) == 0
    assert contrast_structure(np.ones((2, 2)), np.zeros((2, 2)), region) == 1


def test_reference_only_tile_selection_and_flat_fallback():
    flat = np.full((8, 8, 3), .4)
    regions = reference_regions(flat, [[0, 0, 8, 8]], (0, 0, 8, 8))
    assert len(regions[0]['tiles']) == 1
    assert len(regions[0]['tiles'][0][0]) == 64
    yy, xx = np.mgrid[:8, :8]
    textured = np.repeat(((xx + yy) % 2)[..., None], 3, -1)
    assert len(reference_regions(textured, [[0, 0, 8, 8]], (0, 0, 8, 8))[0]['tiles']) == 16


def test_overexposure_denominator_keeps_already_saturated():
    original = np.full((2, 2, 3), .4)
    original[0, 0] = 1
    candidate = original.copy()
    candidate[0, 1] = 1
    passed, diagnostics = photometric(original, candidate, [[0, 0, 2, 2]], (0, 0, 2, 2), [], 1)
    assert passed
    assert diagnostics[0]['new_overexposure'] == .25


def test_analytic_gaussian_and_shared_draw():
    spot = dict(x=10.5, y=9.5, amplitude=.3, sigma_x=8, sigma_y=8, theta=0.)
    assert glare_at([[10.5, 9.5]], [spot])[0] == .3
    assert glare_at([[100, 100]], [spot])[0] == 0
    image = np.full((64, 64, 3), .2, np.float32)
    pool = draw_pool(image, (0, 0, 64, 36), stream(42, 1, 100, 'optical_pool'))
    assert len(pool.candidates) == 3
    for candidate in pool.candidates:
        np.testing.assert_array_equal(candidate[36:], image[36:])
        assert np.isfinite(candidate).all()
        assert 0 <= candidate.min() <= candidate.max() <= 1
    np.testing.assert_allclose(pool.candidates[1][:36] - pool.candidates[0][:36],
                               np.repeat((.5 * pool.glare)[..., None], 3, -1), atol=1e-7)


def test_balancing_and_negative_scores():
    classes = torch.tensor([0] * 9 + [3])
    delta = torch.tensor([.2] * 9 + [1.])
    assert float(difficulty(delta, classes)) == pytest.approx(.6)
    assert float(difficulty(delta, classes, 'instance')) == pytest.approx(.28)
    assert float(difficulty(-delta, classes)) == pytest.approx(-.6)


def test_actual_yolox_logits_parity_shapes_and_640_features():
    cfg = load_config('configs/paper.yaml')
    model = build_model(cfg).eval()
    with torch.no_grad():
        image = torch.full((1, 3, 64, 64), 114.)
        raw, grid, strides = raw_forward(model, image)
        ordinary = model(image)
        assert raw.shape == (1, 84, 9)
        torch.testing.assert_close(raw[..., :4], ordinary[..., :4])
        torch.testing.assert_close(raw[..., 4:].sigmoid(), ordinary[..., 4:])
        features = model.backbone(torch.zeros(1, 3, 640, 640))
        assert [list(f.shape) for f in features] == [[1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]]
        full, _, _ = raw_forward(model, torch.zeros(1, 3, 640, 640))
        assert full.shape == (1, 8400, 9)


@pytest.mark.parametrize('integrated', [False, True])
def test_actual_backward_optimizer_ema_and_scoring_state(integrated):
    cfg = load_config('configs/paper.yaml')
    cfg['model'].update(cbam=integrated, wiou=integrated)
    model = build_model(cfg).train()
    if integrated:
        actual = [model.backbone.get_submodule(p).m.cbam.mlp[0].in_features for p in CBAM_SITES]
        assert actual == [32, 64, 128, 256, 128, 64, 128, 256]
    optimizer = make_optimizer(model, cfg['train'])
    labels = np.array([[8, 8, 24, 28, 0], [34, 8, 52, 28, 3]], np.float32)
    image = np.full((64, 64, 3), 114., np.float32)
    inputs, targets = batch_tensors([image, image], [labels, labels], 'cpu')
    ema = ModelEMA(model)
    before = model.head.obj_preds[0].bias.detach().clone()
    loss = model(inputs, targets)['total_loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.head.obj_preds[0].bias)
    ema.update(model)
    assert ema.updates == 1
    snapshot = {k: v.clone() for k, v in ema.ema.state_dict().items()}
    scorer = FixedAssignmentScorer(ema.ema, image / 255, labels, 'cpu')
    assert scorer.reason is None
    scores, reason, records = scorer.scores([image / 255, image / 255])
    assert reason is None
    assert scores == pytest.approx([0, 0], abs=1e-5)
    assert len(records['original']) == len(labels)
    for k, v in ema.ema.state_dict().items():
        torch.testing.assert_close(v, snapshot[k], rtol=0, atol=0)
    assert ema.updates == 1


def test_ema_first_update_formula_and_integer_buffers():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEMA(model)
    with torch.no_grad():
        model.weight.fill_(3)
        model.num_batches_tracked.fill_(5)
    ema.update(model)
    decay = .9998 * (1 - np.exp(-1 / 2000))
    torch.testing.assert_close(ema.ema.weight, torch.full((2,), 3 - 2 * decay))
    assert int(ema.ema.num_batches_tracked) == 0


def test_wiou_finite_and_empty_and_eval_state():
    module = WIoU3()
    prediction = torch.tensor([[0., 0., 0., 0.], [3., 4., 2., 2.]], requires_grad=True)
    target = torch.tensor([[0., 0., 0., 0.], [0., 0., 2., 2.]])
    values = module(prediction, target)
    assert torch.isfinite(values).all()
    values.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert module(prediction[:0], target[:0]).numel() == 0
    module.eval()
    mean = module.loss_mean.clone()
    module(prediction, target)
    assert torch.equal(mean, module.loss_mean)


def test_selector_shared_pool_and_empty_annotations_fallback():
    cfg = load_config('configs/paper.yaml')
    image = np.full((64, 64, 3), .4, np.float32)
    labels = np.empty((0, 5), np.float32)
    records = []
    for policy in ('A', 'B', 'C', 'D'):
        cfg['optical']['policy'] = policy
        selected, record = Selector(cfg).select(image, labels, (0, 0, 64, 36), 20, 123, None, None, 'cpu')
        records.append(record)
        assert record['feasible'] == [0, 1, 2]
        if policy in ('C', 'D'):
            assert record['fallback'] == 'empty_annotations'
    assert all(r['draw'] == records[0]['draw'] for r in records)


def test_invalid_box_and_empty_feasible_return_original():
    cfg = load_config('configs/paper.yaml')
    image = np.full((64, 64, 3), .4, np.float32)
    labels = np.array([[0, 0, .8, .8, 3]], np.float32)
    selected, record = Selector(cfg).select(image, labels, (0, 0, 64, 36), 1, 123, None, None, 'cpu')
    assert record['fallback'] == 'invalid_defect_box'
    np.testing.assert_array_equal(selected, image)
    cfg['optical']['structure_threshold'] = 1.01
    labels = np.array([[0, 0, 32, 32, 3]], np.float32)
    selected, record = Selector(cfg).select(image, labels, (0, 0, 64, 36), 1, 123, None, None, 'cpu')
    assert record['fallback'] == 'empty_feasible_set'
    np.testing.assert_array_equal(selected, image)


def test_nms_diou_and_degenerate_predictions():
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]])
    scores = torch.tensor([.9, .8, .7])
    for mode in ('standard', 'diou'):
        assert suppress(boxes, scores, .45, mode).tolist() == [0, 2]
    cfg = load_config('configs/paper.yaml')['evaluation']
    raw = torch.tensor([[[5., 5., 0., 10., 10., 10., 0., 0., 0.]]])
    assert len(postprocess(raw, cfg)[0]) == 0


def test_ap_perfect_duplicate_and_category_errors():
    perfect = [{'gt': [[0, 0, 10, 10, 1]], 'predictions': [[0, 0, 10, 10, .9, 1]]}]
    result = evaluate_records(perfect)
    assert result['metrics']['AP50_95'] == 1
    assert result['metrics']['defect_recall'] == 1
    duplicate = deepcopy(perfect)
    duplicate[0]['predictions'].append([0, 0, 10, 10, .8, 1])
    result = evaluate_records(duplicate)
    assert result['metrics']['AP50_95'] == 1  # duplicate occurs after full recall
    assert result['metrics']['defect_precision'] == .5
    wrong = deepcopy(perfect)
    wrong[0]['predictions'][0][-1] = 2
    result = evaluate_records(wrong)
    assert result['defect_counts'] == {'tp': 0, 'fp': 1, 'fn': 1}


def test_101_point_ap_has_low_confidence_threshold_separate_from_pr():
    records = [{'gt': [[0, 0, 10, 10, 1], [20, 20, 30, 30, 1]],
                'predictions': [[0, 0, 10, 10, .01, 1]]}]
    result = evaluate_records(records)
    assert result['metrics']['AP50'] == pytest.approx(51 / 101)
    assert result['metrics']['defect_recall'] == 0
    assert result['metrics']['defect_precision'] is None
