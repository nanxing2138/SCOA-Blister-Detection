"""Auxiliary AM-photo and paper-specific AugMax-P controls, not AugMax-DuBIN."""
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from .difficulty import FixedAssignmentScorer, eval_mode, image_tensor


def photo_chains(image, valid, rng):
    x0, y0, x1, y1 = valid
    source = np.round(image[y0:y1, x0:x1, ::-1] * 255).clip(0, 255).astype(np.uint8)
    chains, descriptions = [], []
    operations = ('brightness', 'contrast', 'colour', 'sharpness', 'autocontrast', 'equalise')
    enhancers = {'brightness': ImageEnhance.Brightness, 'contrast': ImageEnhance.Contrast,
                 'colour': ImageEnhance.Color, 'sharpness': ImageEnhance.Sharpness}
    for _ in range(3):
        output, desc = Image.fromarray(source), []
        for _ in range(int(rng.integers(1, 4))):
            name = operations[int(rng.integers(len(operations)))]
            factor = None
            if name in enhancers:
                factor = float(rng.uniform(0.7, 1.3))
                output = enhancers[name](output).enhance(factor)
            elif name == 'autocontrast':
                output = ImageOps.autocontrast(output)
            else:
                output = ImageOps.equalize(output)
            desc.append({'operation': name, 'factor': factor})
        array = image.copy()
        array[y0:y1, x0:x1] = np.asarray(output).astype(np.float32)[..., ::-1] / 255
        chains.append(array)
        descriptions.append(desc)
    return chains, descriptions


def project_simplex(value):
    ordered = value.sort(descending=True).values
    cumulative = ordered.cumsum(0) - 1
    index = torch.arange(1, len(value) + 1, device=value.device, dtype=value.dtype)
    active = ordered - cumulative / index > 0
    rho = active.nonzero(as_tuple=False)[-1, 0]
    threshold = cumulative[rho] / index[rho]
    return (value - threshold).clamp_min(0)


def augmax(image, chains, labels, weights, blend, student, device):
    """Five projected ascent steps on mixture weights/blend; fixed current-student assignments."""
    scorer = FixedAssignmentScorer(student, image, labels, device)
    original = image_tensor(image, device)
    chain_tensors = torch.cat([image_tensor(i, device) for i in chains], 0)
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    m = torch.tensor(blend, device=device, dtype=torch.float32)
    initial = (1 - m) * original + m * (chain_tensors * w[:, None, None, None]).sum(0, keepdim=True)
    if scorer.reason:
        if scorer.reason.startswith('nonfinite'):
            return image.copy(), 'nonfinite_search'
        return initial.detach()[0].permute(1, 2, 0).cpu().numpy() / 255, scorer.reason
    best, best_value = initial.detach(), float('-inf')
    with eval_mode(student):
        for step in range(6):  # initial draw + five projected candidates
            w = w.detach().requires_grad_(True)
            m = m.detach().requires_grad_(True)
            mixed = (1 - m) * original + m * (chain_tensors * w[:, None, None, None]).sum(0, keepdim=True)
            loss = scorer.losses(mixed).mean()
            if not torch.isfinite(loss):
                return image.copy(), 'nonfinite_search'
            if float(loss.detach()) > best_value:
                best, best_value = mixed.detach(), float(loss.detach())
            if step < 5:
                grad_w, grad_m = torch.autograd.grad(loss, (w, m), only_inputs=True)
                if not torch.isfinite(grad_w).all() or not torch.isfinite(grad_m):
                    return image.copy(), 'nonfinite_search'
                w = project_simplex(w + 0.1 * grad_w)
                m = (m + 0.1 * grad_m).clamp(0, 1)
    return best[0].permute(1, 2, 0).cpu().numpy() / 255, None


def mixture(image, valid, rng, labels, policy, epoch, student, device):
    chains, descriptions = photo_chains(image, valid, rng)
    weights, blend = rng.dirichlet(np.ones(3)), float(rng.beta(1, 1))
    record = {'chains': descriptions, 'initial_weights': weights.tolist(), 'initial_blend': blend}
    if policy == 'AugMax-P' and epoch >= 15:
        result, reason = augmax(image, chains, labels, weights, blend, student, device)
        record['fallback'] = reason
    else:
        result = (1 - blend) * image + blend * sum(w * c for w, c in zip(weights, chains))
    # Exactly preserve original padding despite floating-point mixture roundoff.
    x0, y0, x1, y1 = valid
    output = image.copy()
    output[y0:y1, x0:x1] = result[y0:y1, x0:x1]
    return output.astype(np.float32), record
