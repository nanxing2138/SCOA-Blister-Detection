"""Summarize real results only; synthetic or missing seed results never become paper evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

PAPER_AP = {'basic': (66.53, .43), 'A': (68.35, .34), 'B': (69.45, .39),
            'C': (69.05, .28), 'D': (70.05, .40), 'integrated': (74.03, .42), 'integrated-standard': (73.78, .40),
            'SMR-B': (69.88, .41), 'AM-photo': (68.95, .36), 'AugMax-P': (69.95, .24)}


def summary(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values) >= 2 else None, 'n': len(values)}


def validate_evaluation(record):
    if record.get('synthetic') is not False or record.get('split') != 'test':
        raise ValueError('Require an explicitly nonsynthetic frozen test evaluation')
    name = record['name']
    if name not in PAPER_AP:
        raise ValueError(f'Unknown comparison name {name}')
    ids = record.get('evaluated_image_ids')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Missing/invalid evaluated image IDs; rerun scoa.evaluate')
    digest = hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode('utf-8')).hexdigest()
    if digest != record.get('evaluated_image_ids_sha256'):
        raise ValueError('Evaluated image ID checksum is missing or inconsistent')
    expected_nms = 'diou' if name == 'integrated' else 'standard'
    protocol = record.get('protocol', {})
    if (record.get('nms') != expected_nms or record.get('nms_threshold') != .45 or
        record.get('max_detections') != 100 or record.get('input_size') != 640 or
        record.get('coordinate_frame') != 'letterboxed_input' or
        protocol.get('ap_score_threshold') != .001 or protocol.get('pr_score_threshold') != .25 or
        protocol.get('recall_points') != 101):
        raise ValueError('Evaluation protocol differs from its paper reference; integrated-standard requires standard NMS')
    training = record.get('training_configuration', {})
    model, optical = training.get('model', {}), training.get('optical', {})
    integrated = name in {'integrated', 'integrated-standard'}
    expected_policy = 'D' if integrated else name
    if (model.get('cbam') is not integrated or model.get('wiou') is not integrated or
        optical.get('policy') != expected_policy or optical.get('difficulty_average') != 'class' or
        optical.get('structure_mode') != 'quantile'):
        raise ValueError('Comparison name does not match the actual trained model/policy')
    value = record.get('metrics', {}).get('AP50_95')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Require a finite AP50_95 fraction')
    return record['annotation_sha256'], digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='*', default=[])
    parser.add_argument('--output', default='runs/paper_comparison.json')
    args = parser.parse_args()
    groups = defaultdict(dict)
    dataset_hashes = set()
    for filename in args.results:
        record = json.loads(Path(filename).read_text(encoding='utf-8'))
        if record.get('synthetic'):
            raise ValueError('Synthetic metrics cannot be compared as paper reproductions')
        if record.get('split') != 'test':
            raise ValueError('Paper comparison requires frozen test evaluations')
        if record['name'] not in PAPER_AP:
            raise ValueError(f'Unknown comparison name {record["name"]}; use basic/A/B/C/D/integrated or a listed auxiliary policy')
        if record.get('status') != 'executed' or not record.get('annotation_sha256'):
            raise ValueError('Require an executed scoa.evaluate result with annotation identity')
        dataset_hashes.add(validate_evaluation(record))
        if record['seed'] in groups[record['name']]:
            raise ValueError('Repeated evaluation of one seed is not another training replicate')
        groups[record['name']][record['seed']] = record
    if len(dataset_hashes) > 1:
        raise ValueError('Comparison inputs use different annotations or image subsets; evaluate strata separately')
    report = {'units': 'percentage_points', 'results': {}, 'paired_effects': {}}
    for name, reference in PAPER_AP.items():
        actual = groups.get(name, {})
        values = [100 * r['metrics']['AP50_95'] for r in actual.values()]
        report['results'][name] = {'paper_mean': reference[0], 'paper_sample_sd': reference[1],
                                   'actual': summary(values) if values else None,
                                   'status': 'executed' if values else 'not_verified_missing_real_data',
                                   'required_seeds_complete': set(actual) == {42, 3407, 2026},
                                   'difference_from_paper': statistics.mean(values) - reference[0] if values else None}
    for left, right in [('B', 'A'), ('D', 'C'), ('C', 'A'), ('D', 'B'), ('D', 'SMR-B')]:
        common = sorted(set(groups.get(left, {})) & set(groups.get(right, {})))
        if common:
            diffs = [100 * (groups[left][s]['metrics']['AP50_95'] - groups[right][s]['metrics']['AP50_95']) for s in common]
            report['paired_effects'][f'{left}-{right}'] = dict(summary(diffs), seeds=common)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {output}; missing results remain null.')


if __name__ == '__main__':
    main()
