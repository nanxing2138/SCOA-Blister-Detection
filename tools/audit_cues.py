"""Aggregate genuine blinded review labels; no proxy can generate semantic judgments."""
import argparse
import csv
import json
from pathlib import Path


def summarize(rows):
    accepted_lost = sum(r['accepted'] == '1' and r['label'] == 'lost' for r in rows)
    accepted_definite = sum(r['accepted'] == '1' and r['label'] in {'lost', 'discernible'} for r in rows)
    discernible = sum(r['label'] == 'discernible' for r in rows)
    rejected_discernible = sum(r['accepted'] == '0' and r['label'] == 'discernible' for r in rows)
    return {'rated': len(rows), 'accepted_lost': accepted_lost, 'accepted_definite': accepted_definite,
            'accepted_cue_loss': accepted_lost / accepted_definite if accepted_definite else None,
            'discernible_rejected': rejected_discernible / discernible if discernible else None,
            'uncertain': sum(r['label'] == 'uncertain' for r in rows),
            'ambiguous_reference': sum(r['label'] == 'ambiguous_reference' for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.csv, encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    allowed = {'discernible', 'lost', 'uncertain', 'ambiguous_reference'}
    if any(r['label'] not in allowed or r['accepted'] not in {'0', '1'} for r in rows):
        raise ValueError('CSV requires label and binary accepted; ambiguous originals are separate')
    if any(r['class'] not in {'empty', 'broke', 'lack'} for r in rows):
        raise ValueError('Cue audit classes must be empty/broke/lack')
    if not rows or any(not r.get('source_entity_id', '').strip() for r in rows):
        raise ValueError('Nonempty review rows and source entity IDs are required')
    identities = [(r['pair_id'], r['rule']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate pair/rule; repeated candidates are not independent samples')
    report = {}
    for rule in sorted({r['rule'] for r in rows}):
        subset = [r for r in rows if r['rule'] == rule]
        report[rule] = {category: summarize([r for r in subset if r['class'] == category]) for category in ('empty', 'broke', 'lack')}
        report[rule]['pooled'] = summarize(subset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
