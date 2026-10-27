from copy import deepcopy
import hashlib
import math
import torch
from torch import nn
from ._vendor.yolo_pafpn import YOLOPAFPN
from ._vendor.yolo_head import YOLOXHead
from ._vendor.losses import IOUloss
from ._vendor.yolox import YOLOX
from .checkpoints import load_checkpoint
from .geometry import aligned_iou, cxcywh_to_xyxy


CBAM_SITES = ('backbone.dark2.1', 'backbone.dark3.1', 'backbone.dark4.1',
              'backbone.dark5.2', 'C3_p4', 'C3_p3', 'C3_n3', 'C3_n4')


class StableYOLOXHead(YOLOXHead):
    """Keep the upstream model/assigner, but decode training boxes in FP32 under AMP."""
    def get_output_and_grid(self, output, k, stride, dtype):
        output = output.float()
        return super().get_output_and_grid(output, k, stride, output.type())


class StableSquaredIoULoss(IOUloss):
    """The identical 1-IoU² objective with FP32 areas and intersections."""
    def forward(self, pred, target):
        return super().forward(pred.float(), target.float())


class CBAM(nn.Module):
    """Shared average/max channel MLP, then max/mean spatial attention."""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, max(1, channels // 16)),
                                 nn.ReLU(), nn.Linear(max(1, channels // 16), channels))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                     nn.BatchNorm2d(1, eps=1e-5, momentum=0.01))

    def forward(self, x):
        gate = (self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))).sigmoid()
        x = x * gate[:, :, None, None]
        spatial = torch.cat((x.amax(1, keepdim=True), x.mean(1, keepdim=True)), 1)
        return x * self.spatial(spatial).sigmoid()


class WIoU3(nn.Module):
    """Stateful WIoU-v3; all geometric arithmetic and running mean are FP32."""
    def __init__(self):
        super().__init__()
        self.register_buffer('loss_mean', torch.tensor(1.0, dtype=torch.float32))

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        loss = 1 - aligned_iou(pred, target)
        if loss.numel() == 0:
            return loss
        if self.training:
            with torch.no_grad():
                self.loss_mean.mul_(1 - 1e-4).add_(1e-4 * loss.detach().mean())
        mean = self.loss_mean.float().clamp_min(1e-7)
        beta = (loss.detach() / mean).detach()
        focus = beta / (3.0 * torch.pow(1.9, beta - 3.0))
        a, b = cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target)
        enclosing = torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])
        diagonal = enclosing.square().sum(-1).detach().clamp_min(1e-7)
        distance = (pred[:, :2] - target[:, :2]).square().sum(-1)
        return focus * torch.exp(distance / diagonal) * loss


def build_model(config):
    m = config['model']
    backbone = YOLOPAFPN(depth=m['depth'], width=m['width'])
    head = StableYOLOXHead(num_classes=4, width=m['width'])
    head.iou_loss = StableSquaredIoULoss(reduction='none')
    model = YOLOX(backbone, head)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps, module.momentum = 1e-3, 0.03
    head.initialize_biases(0.01)
    if m['cbam']:
        for site in CBAM_SITES:
            csp = backbone.get_submodule(site)
            # Retain all pretrained m.0/... keys. Add after Bottlenecks, before CSP concat.
            csp.m.add_module('cbam', CBAM(csp.conv1.conv.out_channels))
    if m['wiou']:
        head.iou_loss = WIoU3()
    return model


def load_pretrained(model, path):
    """Only load matching tensors. Call only on a trusted checkpoint."""
    checkpoint = load_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise ValueError('Pretraining checkpoint must be a state_dict or a dict containing model')
    weights = checkpoint.get('model', checkpoint)
    if not isinstance(weights, dict):
        raise ValueError('Pretraining model entry must contain tensor weights')
    own = model.state_dict()
    selected = {k: v for k, v in weights.items() if k in own and torch.is_tensor(v) and own[k].shape == v.shape}
    if not selected:
        raise ValueError('Checkpoint contains no matching model tensors')
    result = model.load_state_dict(selected, strict=False)
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {'path': str(path), 'sha256': digest, 'loaded_tensors': len(selected),
            'missing_keys': result.missing_keys, 'skipped_keys': sorted(set(weights) - set(selected))}


class ModelEMA:
    def __init__(self, model):
        self.ema = deepcopy(model).float().eval()
        self.ema.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = 0.9998 * (1 - math.exp(-self.updates / 2000))
        source = model.state_dict()
        for name, value in self.ema.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(source[name].detach().float(), alpha=1 - decay)


def make_optimizer(model, cfg):
    bn, decay, bias = [], [], []
    for module in model.modules():
        if getattr(module, 'bias', None) is not None and isinstance(module.bias, nn.Parameter):
            bias.append(module.bias)
        weight = getattr(module, 'weight', None)
        if isinstance(weight, nn.Parameter):
            (bn if isinstance(module, nn.modules.batchnorm._BatchNorm) else decay).append(weight)
    optimizer = torch.optim.SGD(bn, lr=cfg['lr'], momentum=cfg['momentum'], nesterov=cfg['nesterov'])
    optimizer.add_param_group({'params': decay, 'weight_decay': cfg['weight_decay']})
    optimizer.add_param_group({'params': bias})
    return optimizer
