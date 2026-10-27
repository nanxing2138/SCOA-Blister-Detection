import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from .data import CLASSES, letterbox
from .difficulty import image_tensor, raw_forward
from .evaluate import restore_model
from .postprocess import postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='runs/inference')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--nms', choices=['standard', 'diou'])
    parser.add_argument('--score', type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.score) or not 0 <= args.score <= 1:
        raise ValueError('--score must be finite and lie in [0,1]')
    if args.threads:
        torch.set_num_threads(args.threads)
    model, config, checkpoint = restore_model(args.checkpoint, args.device)
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode input image')
    canvas, _, _, ratio = letterbox(image, np.empty((0, 5), dtype=np.float32), config['train']['input_size'])
    if args.nms:
        config['evaluation']['nms'] = args.nms
    config['evaluation']['score_threshold'] = args.score
    with torch.no_grad():
        raw, _, _ = raw_forward(model, image_tensor(canvas.astype(np.float32) / 255, args.device))
        predictions = postprocess(raw, config['evaluation'])[0].cpu().numpy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in predictions:
        box = row[:4] / ratio
        box[[0, 2]] = box[[0, 2]].clip(0, image.shape[1])
        box[[1, 3]] = box[[1, 3]].clip(0, image.shape[0])
        category, score = int(row[5]), float(row[4])
        records.append({'xyxy_original': box.tolist(), 'xyxy_input': row[:4].tolist(),
                        'score': score, 'class': CLASSES[category], 'category_id': category + 1})
        x0, y0, x1, y1 = np.round(box).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (20, 210, 40), 2)
        cv2.putText(image, f'{CLASSES[category]} {score:.3f}', (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 210, 40), 1)
    (output / 'predictions.json').write_text(json.dumps({'selected_epoch': checkpoint['epoch'], 'detections': records}, indent=2), encoding='utf-8')
    cv2.imencode('.png', image)[1].tofile(output / 'prediction.png')
    print(f'Wrote {len(records)} detections to {output}')


if __name__ == '__main__':
    main()
