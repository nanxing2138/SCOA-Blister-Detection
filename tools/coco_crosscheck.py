"""Optional numerical cross-check against official pycocotools; no private data uploaded."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scoa.metrics import evaluate_records


def fixture():
    gt = [[0, 0, 10, 10, 0], [20, 0, 30, 10, 1], [40, 0, 50, 10, 2], [60, 0, 70, 10, 3]]
    return [{'image_id': 1, 'gt': gt, 'predictions': [[0, 0, 10, 10, .99, 0], [0, 0, 10, 10, .98, 0],
             [40, 0, 50, 10, .90, 1], [20, 0, 30, 10, .80, 1], [40, 0, 49, 10, .75, 2], [60, 0, 65, 10, .70, 3]]},
            {'image_id': 2, 'gt': gt, 'predictions': [[0, 0, 10, 10, .95, 0], [40, 0, 50, 10, .85, 2], [60, 0, 70, 10, .65, 3]]}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', help='scoa.evaluate predictions JSON; omit for fixed diagnostic fixture')
    parser.add_argument('--output', default='runs/coco_crosscheck.json')
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text(encoding='utf-8')) if args.records else fixture()
    annotations, predictions, images = [], [], []
    for index, record in enumerate(records):
        image_id = index + 1
        images.append({'id': image_id, 'width': 640, 'height': 640})
        for x0, y0, x1, y1, category in record['gt']:
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': int(category) + 1,
                                'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': (x1 - x0) * (y1 - y0), 'iscrowd': 0, 'ignore': 0})
        for x0, y0, x1, y1, score, category in record['predictions']:
            if score >= .001:
                predictions.append({'image_id': image_id, 'category_id': int(category) + 1,
                                    'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': score})
    if not predictions:
        raise ValueError('Use a nonempty prediction fixture to make this cross-check meaningful')
    coco = COCO()
    coco.dataset = {'info': {}, 'images': images, 'annotations': annotations,
                    'categories': [{'id': i, 'name': str(i)} for i in range(1, 5)]}
    coco.createIndex()
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, 'bbox')
    evaluator.params.imgIds = [i['id'] for i in images]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ours = evaluate_records(records)['metrics']
    reference = {'AP50_95': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]), 'AP75': float(evaluator.stats[2])}
    differences = {key: abs(ours[key] - reference[key]) for key in reference}
    if max(differences.values()) > 1e-10:
        raise AssertionError(f'Custom evaluator differs from COCOapi: {differences}')
    report = {'status': 'passed', 'fixture': 'user_records' if args.records else 'fixed_diagnostic_not_paper_data',
              'custom': {key: ours[key] for key in reference}, 'cocoapi': reference,
              'absolute_differences': differences, 'pycocotools_maxDets': [1, 10, 100],
              'limit': 'COCO maxDets is per category; paper input records must already have global top100 applied.'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
