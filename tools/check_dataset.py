import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoa.config import load_config
from scoa.data import CocoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--paper-counts', action='store_true')
    parser.add_argument('--output', default='runs/dataset_check.json')
    args = parser.parse_args()
    config = load_config(args.config)
    report, content_hashes, entity_ids = {}, {}, {}
    expected = {'train': (6826, [59368, 2452, 1992, 1328]),
                'val': (853, [7421, 306, 249, 166]), 'test': (853, [7421, 307, 249, 166])}
    for split in ('train', 'val', 'test'):
        dataset = CocoDataset(config, split)
        counts = Counter(a['category_id'] for a in dataset.annotation['annotations'])
        entities_missing = 0
        for index, item in enumerate(dataset.images):
            dataset[index]  # Verify image decode and every box's dimensions.
            digest = hashlib.sha256(dataset.image_path(item).read_bytes()).hexdigest()
            if digest in content_hashes and content_hashes[digest] != split:
                raise ValueError(f'Exact duplicate across splits: {item["file_name"]}')
            content_hashes[digest] = split
            entity = item.get('entity_id')
            if entity is None:
                entities_missing += 1
            elif entity in entity_ids and entity_ids[entity] != split:
                raise ValueError(f'Physical entity crosses splits: {entity}')
            else:
                entity_ids[entity] = split
        actual = (len(dataset), [counts[i] for i in range(1, 5)])
        if args.paper_counts and actual != expected[split]:
            raise ValueError(f'{split} counts {actual} disagree with paper {expected[split]}')
        report[split] = {'image_count': actual[0], 'class_counts': actual[1],
                         'missing_entity_ids': entities_missing,
                         'annotation_sha256': hashlib.sha256(dataset.annotation_path.read_bytes()).hexdigest()}
    report['limits'] = 'Equal counts do not prove identical paper split IDs. Near-duplicate and adjacent-frame review requires source metadata/manual review.'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
