from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ('medicine', 'empty', 'broke', 'lack')


def letterbox(image, labels, size):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    if new_w < 1 or new_h < 1:
        raise ValueError('Image aspect ratio produces empty content')
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    labels = labels.copy()
    labels[:, :4] *= ratio
    return canvas, labels, (0, 0, new_w, new_h), ratio


def labels_for_head(labels):
    result = np.empty((len(labels), 5), dtype=np.float32)
    result[:, 0] = labels[:, 4]
    result[:, 1:3] = (labels[:, :2] + labels[:, 2:4]) / 2
    result[:, 3:5] = labels[:, 2:4] - labels[:, :2]
    return result


def batch_tensors(images, labels, device):
    arrays = [np.ascontiguousarray(i.transpose(2, 0, 1), dtype=np.float32) for i in images]
    inputs = torch.from_numpy(np.stack(arrays)).to(device)
    targets = np.zeros((len(images), max(1, max(map(len, labels))), 5), dtype=np.float32)
    for index, value in enumerate(labels):
        targets[index, :len(value)] = labels_for_head(value)
    return inputs, torch.from_numpy(targets).to(device)


def collate_raw(batch):
    return batch


class CocoDataset(Dataset):
    def __init__(self, config, split='train'):
        self.root = Path(config['data']['root'])
        self.image_dir = self.root / config['data']['image_dir']
        self.annotation_path = self.root / config['data'][split]
        self.annotation = json.loads(self.annotation_path.read_text(encoding='utf-8'))
        cats = sorted(self.annotation['categories'], key=lambda c: c['id'])
        if [(c['id'], c['name']) for c in cats] != list(zip(range(1, 5), CLASSES)):
            raise ValueError('Require categories 1:medicine, 2:empty, 3:broke, 4:lack')
        self.images = self.annotation['images']
        if len({i['id'] for i in self.images}) != len(self.images):
            raise ValueError('Duplicate image IDs')
        ids = {i['id'] for i in self.images}
        ann_ids = [ann['id'] for ann in self.annotation['annotations']]
        if len(set(ann_ids)) != len(ann_ids):
            raise ValueError('Duplicate annotation IDs')
        self.by_image = defaultdict(list)
        for ann in self.annotation['annotations']:
            if ann['image_id'] not in ids or ann['category_id'] not in range(1, 5):
                raise ValueError('Annotation has unknown image/category')
            if ann.get('iscrowd', 0) or ann.get('ignore', 0):
                raise ValueError('Paper protocol requires ignore=iscrowd=0')
            self.by_image[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def image_path(self, item):
        relative = Path(str(item['file_name']).replace('\\', '/'))
        path = (self.image_dir / relative).resolve()
        try:
            path.relative_to(self.image_dir.resolve())
        except ValueError as error:
            raise ValueError('COCO file_name must stay inside image_dir') from error
        return path

    def __getitem__(self, index):
        item = self.images[index]
        path = self.image_path(item)
        # imdecode supports Unicode Windows paths without silently returning None.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (item['height'], item['width']):
            raise ValueError(f'Image missing or size disagrees with annotation: {path}')
        labels = []
        for ann in self.by_image[item['id']]:
            x, y, w, h = ann['bbox']
            if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
                raise ValueError(f'Invalid annotation box {ann["id"]}')
            if x < 0 or y < 0 or x + w > item['width'] + 1e-3 or y + h > item['height'] + 1e-3:
                raise ValueError(f'Annotation outside image: {ann["id"]}')
            labels.append([x, y, x + w, y + h, ann['category_id'] - 1])
        return {'image': image, 'labels': np.asarray(labels, dtype=np.float32).reshape(-1, 5),
                'id': item['id'], 'index': index, 'file_name': item['file_name']}
