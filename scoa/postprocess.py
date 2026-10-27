import torch
from .geometry import bboxes_iou, cxcywh_to_xyxy


def suppress(boxes, scores, threshold=0.45, mode='standard'):
    if mode not in {'standard', 'diou'}:
        raise ValueError('NMS must be standard or diou')
    order = torch.argsort(scores, descending=True, stable=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        similarity = bboxes_iou(boxes[current].reshape(1, 4), boxes[rest])[0]
        if mode == 'diou':
            center_a = (boxes[current, :2] + boxes[current, 2:]) / 2
            center_b = (boxes[rest, :2] + boxes[rest, 2:]) / 2
            diagonal = (torch.maximum(boxes[current, 2:], boxes[rest, 2:]) -
                        torch.minimum(boxes[current, :2], boxes[rest, :2])).square().sum(-1)
            similarity = similarity - (center_a - center_b).square().sum(-1) / diagonal.clamp_min(1e-7)
        order = rest[similarity <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def postprocess(raw, config):
    """raw [B,A,9]: decoded cxcywh + objectness/class logits. Return xyxy,score,class."""
    results = []
    for predictions in raw.float():
        class_prob, classes = predictions[:, 5:].sigmoid().max(-1)
        scores = predictions[:, 4].sigmoid() * class_prob
        boxes = cxcywh_to_xyxy(predictions[:, :4])
        valid = torch.isfinite(boxes).all(-1) & torch.isfinite(scores)
        valid &= (boxes[:, 2:] > boxes[:, :2]).all(-1) & (scores >= config['score_threshold'])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        selected = []
        for category in range(4):
            indices = (classes == category).nonzero(as_tuple=False).flatten()
            selected.append(indices[suppress(boxes[indices], scores[indices], config['nms_threshold'], config['nms'])])
        keep = torch.cat(selected)
        keep = keep[torch.argsort(scores[keep], descending=True, stable=True)[:config['max_detections']]]
        results.append(torch.cat((boxes[keep], scores[keep, None], classes[keep, None].float()), 1))
    return results
