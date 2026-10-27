import torch


def meshgrid(tensors):
    return torch.meshgrid(*tensors, indexing='ij')


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def bboxes_iou(a, b, xyxy=True):
    # FP16 box areas can overflow at normal 640-pixel detection scales.
    a, b = a.float(), b.float()
    if not xyxy:
        a, b = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    lo = torch.maximum(a[:, None, :2], b[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-16)


def aligned_iou(a, b):
    a, b = cxcywh_to_xyxy(a.float()), cxcywh_to_xyxy(b.float())
    intersection = (torch.minimum(a[..., 2:], b[..., 2:]) -
                    torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-7)
