from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class Pool:
    candidates: list
    amplitudes: list
    spots: list
    glare: np.ndarray
    draw: dict


def glare_at(points, spots):
    """Analytic Eq(3)-(4) at continuous coordinates, including unrounded box centres."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    total = np.zeros(len(points), dtype=np.float64)
    for spot in spots:
        dx, dy = points[:, 0] - spot['x'], points[:, 1] - spot['y']
        cosine, sine = math.cos(spot['theta']), math.sin(spot['theta'])
        u, v = dx * cosine + dy * sine, -dx * sine + dy * cosine
        q = (u / spot['sigma_x']) ** 2 + (v / spot['sigma_y']) ** 2
        total += spot['amplitude'] * np.exp(-q / 2) * (q <= 9)
    return total


def draw_pool(image, valid, rng, amplitudes=(0.5, 1.0, 1.5)):
    """Eq(1)-(7). HWC float [0,1]; integer valid=(x0,y0,x1,y1), padding unchanged."""
    x0, y0, x1, y1 = map(int, valid)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('Empty valid content')
    content = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    points = np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5), -1)
    spots = []
    for _ in range(int(rng.integers(1, 4))):
        spots.append(dict(amplitude=float(rng.uniform(0.15, 0.40)),
                          sigma_x=float(rng.uniform(8, 32)), sigma_y=float(rng.uniform(8, 32)),
                          theta=float(rng.uniform(0, math.pi)),
                          x=float(rng.integers(x0, x1)) + 0.5,
                          y=float(rng.integers(y0, y1)) + 0.5))
    glare = glare_at(points, spots).reshape(content.shape[:2]).astype(np.float32)
    cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
    radius, strength = float(rng.uniform(32, 96)), float(rng.uniform(0.2, 0.5))
    blur = bool(rng.integers(0, 2))
    sigma = float(rng.uniform(0.3, 1.2) if blur else rng.uniform(0.5, 1.0))
    kernel = 2 * math.ceil(3 * sigma) + 1
    filtered = cv2.GaussianBlur(content, (kernel, kernel), sigma,
                                borderType=cv2.BORDER_REFLECT_101)
    sharpen_strength = None
    if not blur:
        sharpen_strength = float(rng.uniform(0.2, 0.5))
        filtered = np.clip(content + sharpen_strength * (content - filtered), 0, 1)
    weight = strength * np.maximum(1 - np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / radius, 0)
    transformed = (1 - weight[..., None]) * content + weight[..., None] * filtered
    candidates = []
    for amplitude in amplitudes:
        candidate = image.copy()
        candidate[y0:y1, x0:x1] = np.clip(transformed + amplitude * glare[..., None], 0, 1)
        candidates.append(candidate.astype(np.float32))
    return Pool(candidates, list(amplitudes), spots, glare,
                dict(spots=spots, local_center=[cx, cy], radius=radius, strength=strength,
                     mode='blur' if blur else 'sharpen', sigma=sigma, kernel=kernel,
                     sharpen_strength=sharpen_strength))
