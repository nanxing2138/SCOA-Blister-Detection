"""Package portable source and validation evidence, excluding runtimes/data/checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='../SCOA_reproduction_20261027.zip')
    parser.add_argument('--include-weights', action='store_true', help='Bundle downloaded weights after checking their recorded SHA256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    excluded = {'.validation_env', '.venv', '.git', '.pytest_cache', '__pycache__', 'runs', 'data', 'build', 'dist', 'delivery'}
    if not args.include_weights:
        excluded.add('weights')
    entries, hashes = [], {}
    for directory, directories, filenames in os.walk(root):
        # Skip project outputs at root; retain upstream source subpackages named data.
        skipped = excluded if Path(directory) == root else {'.git', '.pytest_cache', '__pycache__'}
        directories[:] = sorted(d for d in directories if d not in skipped and not d.endswith('.egg-info'))
        for filename in sorted(filenames):
            path = Path(directory) / filename
            relative_path = path.relative_to(root)
            bundled_weight = args.include_weights and relative_path.parts[0] == 'weights' and path.suffix == '.pth'
            if (path.suffix in {'.pyc', '.pth', '.zip'} and not bundled_weight) or path == output:
                continue
            if path.suffix == '.part':
                continue
            if bundled_weight:
                metadata = json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))
                if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                    raise ValueError(f'Weight hash disagrees with recorded source metadata: {path.name}')
            relative = relative_path.as_posix()
            entries.append((path, relative))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in entries:
            archive.write(path, root.name + '/' + relative)
        archive.writestr(root.name + '/PACKAGE_MANIFEST.json', json.dumps(hashes, indent=2))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Corrupt archive entry: {bad}')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.sha256.txt').write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    print(json.dumps({'archive': str(output), 'files': len(entries), 'bytes': output.stat().st_size, 'sha256': digest}))


if __name__ == '__main__':
    main()
