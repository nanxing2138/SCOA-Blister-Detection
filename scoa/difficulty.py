from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import aligned_iou, meshgrid


@contextmanager
def eval_mode(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, state in states:
            module.training = state


def image_tensor(image, device):
    array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0).to(device) * 255.0


def raw_forward(model, images):
    """Eval head without sigmoid/BN changes or mutable grid cache; gradients to input allowed."""
    head = model.head
    outputs, grids, strides = [], [], []
    for level, (feature, stride) in enumerate(zip(model.backbone(images), head.strides)):
        feature = head.stems[level](feature)
        cls = head.cls_preds[level](head.cls_convs[level](feature))
        reg_feature = head.reg_convs[level](feature)
        reg = head.reg_preds[level](reg_feature)
        obj = head.obj_preds[level](reg_feature)
        raw = torch.cat((reg, obj, cls), 1).flatten(2).permute(0, 2, 1).float()
        h, w = feature.shape[-2:]
        yy, xx = meshgrid([torch.arange(h, device=images.device), torch.arange(w, device=images.device)])
        grid = torch.stack((xx, yy), -1).reshape(1, -1, 2).float()
        decoded = torch.cat(((raw[..., :2] + grid) * stride,
                             raw[..., 2:4].exp() * stride, raw[..., 4:]), -1)
        outputs.append(decoded)
        grids.append(grid)
        strides.append(torch.full((1, h * w), float(stride), device=images.device))
    return torch.cat(outputs, 1), torch.cat(grids, 1), torch.cat(strides, 1)


def instance_losses(raw, positive_indices, gt_indices, boxes_cxcywh, classes):
    """Eq(13), including all four unweighted BCE classes and ordinary 1-IoU."""
    predictions = raw[0, positive_indices].float()
    targets = F.one_hot(classes[gt_indices].long(), 4).float()
    loss = F.binary_cross_entropy_with_logits(predictions[:, 5:], targets, reduction='none').sum(-1)
    loss = loss + F.binary_cross_entropy_with_logits(predictions[:, 4], torch.ones_like(predictions[:, 4]), reduction='none')
    loss = loss + 5 * (1 - aligned_iou(predictions[:, :4], boxes_cxcywh[gt_indices]))
    sums = loss.new_zeros(len(classes)).scatter_add(0, gt_indices, loss)
    counts = loss.new_zeros(len(classes)).scatter_add(0, gt_indices, torch.ones_like(loss))
    if torch.any(counts == 0):
        raise ValueError('unmatched_instance')
    return sums / counts


def difficulty(delta, classes, average='class'):
    if average == 'instance':
        return delta.mean()
    return torch.stack([delta[classes == c].mean() for c in classes.unique()]).mean()


class FixedAssignmentScorer:
    def __init__(self, model, original, labels, device):
        self.model, self.device, self.reason = model, device, None
        if not len(labels):
            self.reason = 'empty_annotations'
            return
        label_tensor = torch.as_tensor(labels, device=device, dtype=torch.float32)
        self.classes = label_tensor[:, 4].long()
        self.boxes = torch.cat(((label_tensor[:, :2] + label_tensor[:, 2:4]) / 2,
                                label_tensor[:, 2:4] - label_tensor[:, :2]), 1)
        images = image_tensor(original, device)
        with torch.no_grad(), eval_mode(model):
            raw, grid, strides = raw_forward(model, images)
            if not torch.isfinite(raw).all():
                self.reason = 'nonfinite_original'
                return
            # Upstream SimOTA expects logits and decoded boxes, not probabilities.
            eligible, _ = model.head.get_in_boxes_info(self.boxes, strides, grid[..., 0], grid[..., 1], raw.shape[1], len(labels))
            if not eligible.any():
                self.reason = 'unmatched_instance'
                return
            _, foreground, _, matched, _ = model.head.get_assignments(
                0, len(labels), raw.shape[1], self.boxes, self.classes.float(), raw[0, :, :4],
                strides, grid[..., 0], grid[..., 1], raw[..., 5:], raw[..., :4],
                raw[..., 4:5], None, images)
            self.positive = foreground.nonzero(as_tuple=False).flatten()
            self.matched = matched.long()
            if len(self.matched.unique()) != len(labels):
                self.reason = 'unmatched_instance'
                return
            self.original_loss = instance_losses(raw, self.positive, self.matched, self.boxes, self.classes).detach()
            if not torch.isfinite(self.original_loss).all():
                self.reason = 'nonfinite_original_loss'

    def losses(self, images):
        raw, _, _ = raw_forward(self.model, images)
        return instance_losses(raw, self.positive, self.matched, self.boxes, self.classes)

    def scores(self, candidates, average='class'):
        if self.reason:
            return None, self.reason, None
        values, records = [], []
        with torch.no_grad(), eval_mode(self.model):
            for candidate in candidates:
                losses = self.losses(image_tensor(candidate, self.device))
                score = difficulty(losses - self.original_loss, self.classes, average)
                if not torch.isfinite(score) or not torch.isfinite(losses).all():
                    return None, 'nonfinite_candidate_score', None
                values.append(float(score))
                records.append(losses.cpu().tolist())
        return values, None, {'original': self.original_loss.cpu().tolist(), 'candidates': records,
                              'classes': self.classes.cpu().tolist(), 'positive_locations': self.positive.cpu().tolist(),
                              'matched_gt_indices': self.matched.cpu().tolist()}
