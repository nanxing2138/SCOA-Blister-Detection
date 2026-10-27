"""Record actual process outcomes for tests and synthetic end-to-end runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/verification')
    parser.add_argument('--skip-tests', action='store_true', help='Only when the same unit suite has already been run for this revision')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'execution.json').exists():
        raise FileExistsError('Use a new --output directory to preserve earlier validation evidence')
    commands = [
        ['tools/verify_vendor.py'],
        ['-m', 'compileall', '-q', 'scoa', 'tools', 'tests'],
        ['-m', 'pytest', '-q', '--junitxml', str(output / 'tests.xml')],
        ['tools/make_synthetic.py'],
        ['tools/check_dataset.py', '--config', 'configs/smoke.yaml', '--output', str(output / 'dataset_check.json')],
        ['-m', 'scoa.train', '--config', 'configs/smoke.yaml', '--device', 'cpu', '--threads', '2', '--set', f'output={output}/core'],
        ['-m', 'scoa.train', '--config', 'configs/smoke.yaml', '--device', 'cpu', '--threads', '2',
         '--set', f'output={output}/integrated', '--set', 'model.cbam=true', '--set', 'model.wiou=true', '--set', 'evaluation.nms=diou'],
        ['-m', 'scoa.evaluate', '--config', 'configs/smoke.yaml', '--checkpoint', str(output / 'core/seed_42/best.pth'),
         '--device', 'cpu', '--threads', '2', '--output', str(output / 'synthetic_test.json')],
        ['-m', 'scoa.evaluate', '--config', 'configs/smoke.yaml', '--checkpoint', str(output / 'integrated/seed_42/best.pth'),
         '--device', 'cpu', '--threads', '2', '--set', 'model.cbam=true', '--set', 'model.wiou=true',
         '--set', 'evaluation.nms=diou', '--output', str(output / 'synthetic_integrated_test.json')],
        ['-m', 'scoa.infer', '--checkpoint', str(output / 'core/seed_42/best.pth'),
         '--image', 'data/synthetic/images/test_20001.png', '--device', 'cpu', '--threads', '2', '--output', str(output / 'inference')],
        ['tools/compare_results.py', '--output', 'experiments/paper_comparison.json'],
    ]
    if args.skip_tests:
        commands = [command for command in commands if command[:2] != ['-m', 'pytest']]
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'synthetic_code_flow_only', 'commands': []}
    for index, command in enumerate(commands):
        result = subprocess.run([sys.executable, '-X', 'utf8', *command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        (output / f'{index:02d}.log').write_text(result.stdout, encoding='utf-8')
        report['commands'].append({'argv': command, 'exit_code': result.returncode, 'log': f'{index:02d}.log'})
        report['status'] = 'failed' if result.returncode else 'running'
        (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'[{index+1}/{len(commands)}] exit={result.returncode}: {" ".join(command)}', flush=True)
        if result.returncode:
            print(result.stdout, flush=True)
            raise SystemExit(result.returncode)
    report['status'] = 'passed'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
