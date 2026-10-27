import argparse
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
import torch
from .data import letterbox
from .difficulty import image_tensor, raw_forward
from .evaluate import restore_model
from .postprocess import postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output', default='runs/benchmark.json')
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError('rounds/iterations must be positive; warmup must be nonnegative')
    if args.threads:
        torch.set_num_threads(args.threads)
    model, config, _ = restore_model(args.checkpoint, args.device)
    source = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError('Cannot decode input image')
    cuda = args.device.startswith('cuda')
    def synchronize():
        if cuda:
            torch.cuda.synchronize(args.device)
    def process():
        canvas, _, _, _ = letterbox(source, np.empty((0, 5), np.float32), config['train']['input_size'])
        inputs = image_tensor(canvas.astype(np.float32) / 255, args.device)
        with torch.cuda.amp.autocast(enabled=cuda):
            raw, _, _ = raw_forward(model, inputs)
        return postprocess(raw, config['evaluation'])[0].cpu()
    timings, means = [], []
    with torch.no_grad():
        for _ in range(args.rounds):
            for _ in range(args.warmup):
                process()
            synchronize()
            current = []
            for _ in range(args.iterations):
                synchronize()
                begin = time.perf_counter()
                process()
                synchronize()
                current.append(1000 * (time.perf_counter() - begin))
            timings.extend(current)
            means.append(statistics.mean(current))
        forward_ms = None
        if cuda:
            # Separate resident-input CUDA-event network timing; do not add to end-to-end.
            canvas, _, _, _ = letterbox(source, np.empty((0, 5), np.float32), config['train']['input_size'])
            inputs = image_tensor(canvas.astype(np.float32) / 255, args.device)
            values = []
            for _ in range(args.iterations):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(args.device))
                with torch.cuda.amp.autocast():
                    raw_forward(model, inputs)
                end.record(torch.cuda.current_stream(args.device))
                end.synchronize()
                values.append(begin.elapsed_time(end))
            forward_ms = statistics.mean(values)
    report = {'round_means_ms': means, 'end_to_end_mean_ms': statistics.mean(means),
              'round_sample_sd_ms': statistics.stdev(means) if len(means) > 1 else None,
              'end_to_end_p95_ms': float(np.percentile(timings, 95)), 'forward_cuda_event_ms': forward_ms,
              'warmup_each_round': args.warmup, 'iterations_each_round': args.iterations,
              'rounds': args.rounds, 'input_size': config['train']['input_size'],
              'protocol_matches_paper_counts': (args.warmup, args.iterations, args.rounds) == (200, 1000, 3),
              'excludes_image_file_decode': True}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
