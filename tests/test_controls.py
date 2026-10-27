import json
import subprocess
import sys
from copy import deepcopy
import numpy as np
import pytest
import torch
from scoa.augment import augmax, photo_chains, project_simplex
from scoa.config import load_config
from scoa.model import build_model
from scoa.randomness import stream

torch.set_num_threads(2)


def test_simplex_projection():
    for value in ([-1., 2., .5], [0., 0., 0.], [.2, .3, .5]):
        result = project_simplex(torch.tensor(value))
        assert torch.all(result >= 0)
        assert float(result.sum()) == pytest.approx(1.)
    torch.testing.assert_close(project_simplex(torch.tensor([.2, .3, .5])), torch.tensor([.2, .3, .5]))


def test_actual_augmax_search_preserves_student_and_padding():
    model = build_model(load_config('configs/paper.yaml')).train()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    rng = stream(42, 20, 1, 'photo')
    image = rng.uniform(.2, .7, (64, 64, 3)).astype(np.float32)
    image[36:] = 114 / 255
    valid = (0, 0, 64, 36)
    chains, descriptions = photo_chains(image, valid, rng)
    labels = np.array([[4, 4, 22, 28, 1], [38, 5, 58, 29, 3]], np.float32)
    output, reason = augmax(image, chains, labels, np.array([.2, .3, .5]), .5, model, 'cpu')
    assert reason is None
    assert output.shape == image.shape and np.isfinite(output).all()
    np.testing.assert_allclose(output[36:], image[36:], atol=1e-6)
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


def test_synthetic_comparison_is_rejected(tmp_path):
    path = tmp_path / 'fake_paper_result.json'
    path.write_text(json.dumps({'synthetic': True}), encoding='utf-8')
    result = subprocess.run([sys.executable, 'tools/compare_results.py', '--results', str(path),
                             '--output', str(tmp_path / 'comparison.json')], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / 'comparison.json').exists()


def test_smr_positive_fit_from_diagnostic_records():
    from tools.fit_smr import fit
    events = [{'selected_index': i % 3, 'feasible': [0, 1, 2]} for i in range(300)]
    result = fit(events)
    assert all(w > 0 for w in result['weights'])
    assert result['max_difference_pp'] < .01
