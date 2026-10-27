"""Generate explicitly synthetic COCO fixtures. Never evidence for paper accuracy."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def generate(root, train=4, val=2, test=2):
    root = Path(root)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(('medicine', 'empty', 'broke', 'lack'))]
    for split_index, (split, count) in enumerate([('train', train), ('val', val), ('test', test)]):
        images, annotations = [], []
        for index in range(count):
            image_id = split_index * 10000 + index + 1
            rng = np.random.default_rng(2026 + image_id)
            yy, xx = np.mgrid[:180, :320]
            background = 125 + 12 * np.sin(xx / 7) + 8 * np.cos(yy / 9) + rng.normal(0, 3, xx.shape)
            image = np.repeat(np.clip(background, 0, 255).astype(np.uint8)[..., None], 3, -1)
            for category in range(4):
                x, y, w, h = 12 + category * 77, 32 + int(rng.integers(-4, 5)), 64, 110
                cv2.ellipse(image, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (215, 210, 205), 3)
                if category in (0, 2):
                    cv2.ellipse(image, (x + w // 2, y + h // 2), (18, 34), 0, 0, 360, (155, 185, 230), -1)
                else:
                    cv2.ellipse(image, (x + w // 2, y + h // 2), (21, 38), 0, 0, 360, (65, 80, 90), -1)
                if category in (2, 3):
                    points = np.array([[x + 2, y + 25], [x + 18, y + 34], [x + 6, y + 45], [x + 23, y + 51]])
                    cv2.polylines(image, [points], False, (15, 15, 15), 3)
                annotations.append({'id': index * 4 + category + 1, 'image_id': image_id,
                                    'category_id': category + 1, 'bbox': [x, y, w, h], 'area': w * h,
                                    'iscrowd': 0, 'ignore': 0})
            filename = f'{split}_{image_id}.png'
            cv2.imencode('.png', image)[1].tofile(root / 'images' / filename)
            images.append({'id': image_id, 'file_name': filename, 'height': 180, 'width': 320,
                           'entity_id': f'synthetic_entity_{image_id}'})
        content = {'synthetic': True, 'description': 'CODE FLOW FIXTURE ONLY; NOT REAL PHARMACEUTICAL DATA',
                   'images': images, 'annotations': annotations, 'categories': categories}
        (root / 'annotations' / f'{split}.json').write_text(json.dumps(content, indent=2), encoding='utf-8')
    print(f'Synthetic fixtures saved in {root}; these cannot reproduce paper results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/synthetic')
    args = parser.parse_args()
    generate(args.output)
