"""YOLOX-style Mosaic/affine/HSV/flip with an explicit independent RNG."""
import math
import cv2
import numpy as np
from .data import letterbox


def mosaic_coordinates(index, xc, yc, w, h, size):
    if index == 0:
        large = max(xc - w, 0), max(yc - h, 0), xc, yc
        small = w - (large[2] - large[0]), h - (large[3] - large[1]), w, h
    elif index == 1:
        large = xc, max(yc - h, 0), min(xc + w, size * 2), yc
        small = 0, h - (large[3] - large[1]), min(w, large[2] - large[0]), h
    elif index == 2:
        large = max(xc - w, 0), yc, xc, min(2 * size, yc + h)
        small = w - (large[2] - large[0]), 0, w, min(large[3] - large[1], h)
    else:
        large = xc, yc, min(xc + w, size * 2), min(2 * size, yc + h)
        small = 0, 0, min(w, large[2] - large[0]), min(h, large[3] - large[1])
    return large, small


def basic_augmentation(sample, dataset, config, rng, epoch):
    size, cfg = config['train']['input_size'], config['basic']
    image, labels = sample['image'].copy(), sample['labels'].copy()
    original, original_labels, _, _ = letterbox(image, labels, size)
    if epoch <= config['train']['epochs'] - config['train']['no_aug_epochs'] and rng.random() < cfg['mosaic_prob']:
        canvas = np.full((2 * size, 2 * size, 3), 114, dtype=np.uint8)
        yc, xc = int(rng.uniform(0.5 * size, 1.5 * size)), int(rng.uniform(0.5 * size, 1.5 * size))
        parts = [sample] + [dataset[int(rng.integers(len(dataset)))] for _ in range(3)]
        label_parts = []
        for index, part in enumerate(parts):
            source, target = part['image'], part['labels'].copy()
            h, w = source.shape[:2]
            ratio = min(size / h, size / w)
            resized = cv2.resize(source, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
            h, w = resized.shape[:2]
            large, small = mosaic_coordinates(index, xc, yc, w, h, size)
            lx0, ly0, lx1, ly1 = large
            sx0, sy0, sx1, sy1 = small
            canvas[ly0:ly1, lx0:lx1] = resized[sy0:sy1, sx0:sx1]
            target[:, :4] *= ratio
            target[:, [0, 2]] += lx0 - sx0
            target[:, [1, 3]] += ly0 - sy0
            label_parts.append(target)
        labels = np.concatenate(label_parts, 0)
        labels[:, :4] = np.clip(labels[:, :4], 0, 2 * size)
        angle, scale = rng.uniform(-cfg['degrees'], cfg['degrees']), rng.uniform(*cfg['scale'])
        rotation = cv2.getRotationMatrix2D((0, 0), angle, scale)
        shear_x, shear_y = np.tan(rng.uniform(-cfg['shear'], cfg['shear'], 2) * math.pi / 180)
        matrix = np.ones((2, 3))
        matrix[0] = rotation[0] + shear_y * rotation[1]
        matrix[1] = rotation[1] + shear_x * rotation[0]
        # YOLOX 0.3.0 applies translation around zero after four-image placement.
        matrix[:, 2] = rng.uniform(-cfg['translate'], cfg['translate'], 2) * size
        image = cv2.warpAffine(canvas, matrix, (size, size), borderValue=(114, 114, 114))
        if len(labels):
            corners = np.ones((4 * len(labels), 3))
            corners[:, :2] = labels[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
            warped = (corners @ matrix.T).reshape(-1, 4, 2)
            labels[:, :4] = np.concatenate((warped.min(1), warped.max(1)), -1).clip(0, size)
    else:
        image, labels = original.copy(), original_labels.copy()
    if rng.random() < cfg['hsv_prob']:
        delta = (rng.uniform(-1, 1, 3) * cfg['hsv_gains'] * rng.integers(0, 2, 3)).astype(np.int16)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + delta[0]) % 180
        hsv[..., 1:] = (hsv[..., 1:] + delta[1:]).clip(0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < cfg['flip_prob']:
        image = image[:, ::-1].copy()
        labels[:, [0, 2]] = size - labels[:, [2, 0]]
    valid = np.minimum(labels[:, 2] - labels[:, 0], labels[:, 3] - labels[:, 1]) > 1
    labels = labels[valid]
    if not len(labels) and len(original_labels):
        image, labels = original, original_labels
    return image.astype(np.float32), labels
