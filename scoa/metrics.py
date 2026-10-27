"""Paper §4.4: global top100, 101-point interpolated AP, no ignored/crowd GT."""
import numpy as np
from .data import CLASSES


def iou_numpy(a, b):
    if not len(b):
        return np.empty(0, dtype=np.float64)
    intersection = np.maximum(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0).prod(-1)
    return intersection / np.maximum(np.maximum(a[2:] - a[:2], 0).prod() +
                                     np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1) - intersection, 1e-16)


def match_class(records, category, iou_threshold, score_threshold):
    detections, total_gt = [], 0
    for image in records:
        gt = np.asarray(image['gt'], dtype=float).reshape(-1, 5)
        pred = np.asarray(image['predictions'], dtype=float).reshape(-1, 6)
        boxes = gt[gt[:, 4] == category, :4]
        total_gt += len(boxes)
        pred = pred[(pred[:, 5] == category) & (pred[:, 4] >= score_threshold)]
        pred = pred[np.argsort(-pred[:, 4], kind='stable')]
        used = np.zeros(len(boxes), dtype=bool)
        for prediction in pred:
            overlaps = iou_numpy(prediction[:4], boxes)
            overlaps[used] = -1
            index = int(np.argmax(overlaps)) if len(overlaps) else None
            true_positive = index is not None and overlaps[index] >= iou_threshold
            if true_positive:
                used[index] = True
            detections.append((float(prediction[4]), int(true_positive)))
    detections.sort(key=lambda pair: -pair[0])
    tp = np.asarray([pair[1] for pair in detections], dtype=np.float64)
    return tp, total_gt


def interpolated_ap(tp, total_gt):
    if total_gt == 0:
        return None, None
    tp_sum = tp.cumsum()
    recall = tp_sum / total_gt
    precision = tp_sum / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
    curve = np.zeros(101, dtype=float)
    inside = positions < len(precision)
    curve[inside] = precision[positions[inside]]
    return float(curve.mean()), curve.tolist()


def evaluate_records(records, score_threshold=0.001, pr_threshold=0.25):
    per_class, defect = {}, {'tp': 0, 'fp': 0, 'fn': 0}
    aps = []
    for category, name in enumerate(CLASSES):
        category_aps, curve = [], None
        for threshold in np.linspace(0.5, 0.95, 10):
            tp, total_gt = match_class(records, category, float(threshold), score_threshold)
            ap, recall_curve = interpolated_ap(tp, total_gt)
            category_aps.append(ap)
            if threshold == 0.5:
                curve = recall_curve
        tp, total_gt = match_class(records, category, 0.5, pr_threshold)
        true_positive = int(tp.sum())
        counts = {'tp': true_positive, 'fp': len(tp) - true_positive, 'fn': total_gt - true_positive}
        per_class[name] = dict(AP50=category_aps[0], AP75=category_aps[5],
                               AP50_95=float(np.mean(category_aps)) if category_aps[0] is not None else None,
                               precision=true_positive / len(tp) if len(tp) else None,
                               recall=true_positive / total_gt if total_gt else None,
                               counts=counts, precision_at_101_recall=curve)
        aps.append(category_aps)
        if category > 0:
            for key in counts:
                defect[key] += counts[key]
    existing = [a for a in aps if a[0] is not None]
    overall = {'AP50': float(np.mean([a[0] for a in existing])) if existing else None,
               'AP75': float(np.mean([a[5] for a in existing])) if existing else None,
               'AP50_95': float(np.mean(existing)) if existing else None,
               'defect_precision': defect['tp'] / (defect['tp'] + defect['fp']) if defect['tp'] + defect['fp'] else None,
               'defect_recall': defect['tp'] / (defect['tp'] + defect['fn']) if defect['tp'] + defect['fn'] else None}
    return {'metrics': overall, 'per_class': per_class, 'defect_counts': defect,
            'image_count': len(records), 'units': 'fraction_0_to_1',
            'protocol': {'ap_score_threshold': score_threshold, 'pr_score_threshold': pr_threshold,
                         'recall_points': 101, 'iou_thresholds': np.linspace(.5, .95, 10).tolist()}}
