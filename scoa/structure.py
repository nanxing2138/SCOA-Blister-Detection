import math
import numpy as np
from .optics import glare_at


class InvalidDefectBox(ValueError):
    """Preprocessing violates the paper's minimum valid-pixel requirement."""


def luminance(bgr):
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def box_pixels(box, valid):
    """Pixel centres in half-open continuous box intersected with valid content."""
    x0, y0, x1, y1 = map(float, box[:4])
    vx0, vy0, vx1, vy1 = valid
    if not np.isfinite([x0, y0, x1, y1]).all() or x0 >= x1 or y0 >= y1:
        raise InvalidDefectBox('Non-finite or degenerate defect box')
    # ceil(bound-0.5) gives exact inclusive lower/exclusive upper pixel-index bounds.
    xs = np.arange(max(vx0, math.ceil(x0 - 0.5)), min(vx1, math.ceil(x1 - 0.5)), dtype=int)
    ys = np.arange(max(vy0, math.ceil(y0 - 0.5)), min(vy1, math.ceil(y1 - 0.5)), dtype=int)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return yy.ravel(), xx.ravel()


def reference_regions(image, defect_boxes, valid, grid=4, variance_threshold=1e-4):
    light = luminance(image).astype(np.float64)
    regions = []
    for box in defect_boxes:
        yy, xx = box_pixels(box, valid)
        if len(xx) < 2:
            raise InvalidDefectBox('Fewer than two valid pixels in defect box')
        gx = np.floor((xx + 0.5 - box[0]) * grid / (box[2] - box[0])).astype(int)
        gy = np.floor((yy + 0.5 - box[1]) * grid / (box[3] - box[1])).astype(int)
        tiles = []
        for tile in range(grid * grid):
            mask = gy * grid + gx == tile
            if mask.sum() >= 2 and np.var(light[yy[mask], xx[mask]], ddof=0) >= variance_threshold:
                tiles.append((yy[mask], xx[mask]))
        regions.append({'whole': (yy, xx), 'tiles': tiles or [(yy, xx)]})
    return regions


def contrast_structure(reference, candidate, region, epsilon=1e-4):
    yy, xx = region
    a, b = reference[yy, xx].astype(np.float64), candidate[yy, xx].astype(np.float64)
    da, db = a - a.mean(), b - b.mean()
    value = (2 * np.mean(da * db) + epsilon) / (np.mean(da * da) + np.mean(db * db) + epsilon)
    return float(np.clip(value, 0, 1))


def order_statistic(scores, quantile):
    if not scores:
        raise ValueError('Empty tile scores')
    return float(np.sort(scores)[max(0, math.ceil(quantile * len(scores)) - 1)])


def structure_scores(image, candidate, regions, config):
    a, b = luminance(image), luminance(candidate)
    results = []
    for region in regions:
        if config['structure_mode'] == 'whole':
            value = contrast_structure(a, b, region['whole'], config['epsilon_q'])
        else:
            scores = [contrast_structure(a, b, tile, config['epsilon_q']) for tile in region['tiles']]
            value = float(np.mean(scores)) if config['structure_mode'] == 'mean' else order_statistic(scores, config['quantile'])
        results.append(value)
    return results


def photometric(image, candidate, boxes, valid, spots, amplitude):
    diagnostics = []
    accepted = True
    for box in boxes:
        yy, xx = box_pixels(box, valid)
        if len(xx) < 2:
            raise InvalidDefectBox('Fewer than two valid pixels')
        ref_over = image[yy, xx].max(-1) >= 0.98
        new_over = candidate[yy, xx].max(-1) >= 0.98
        proportion = float(np.mean(new_over & ~ref_over))
        center = (np.asarray(box[:2]) + np.asarray(box[2:4])) / 2
        center_glare = float(amplitude * glare_at(center, spots)[0])
        passed = center_glare <= 0.15 and proportion <= 0.30
        accepted = accepted and passed
        diagnostics.append({'new_overexposure': proportion, 'center_increment': center_glare, 'pass': bool(passed)})
    return bool(accepted), diagnostics
