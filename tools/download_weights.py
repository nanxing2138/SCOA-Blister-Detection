"""Download official YOLOX assets atomically and retain provenance and SHA256."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['s', 'l'], default='s')
    parser.add_argument('--output', default='weights')
    args = parser.parse_args()
    filename = f'yolox_{args.model}.pth'
    release_url = 'https://api.github.com/repos/Megvii-BaseDetection/YOLOX/releases/tags/0.1.1rc0'
    request = urllib.request.Request(release_url, headers={'User-Agent': 'SCOA-reproduction'})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(a for a in release['assets'] if a['name'] == filename)
    url = asset['browser_download_url']
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    if path.exists():
        raise FileExistsError(f'{path} already exists; refusing to replace it')
    temporary = path.with_suffix('.pth.part')
    if temporary.exists():
        raise FileExistsError(f'{temporary} exists; choose a new --output directory to preserve the partial download')
    digest, downloaded = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={'User-Agent': 'SCOA-reproduction'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('xb') as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
    if downloaded != asset['size']:
        raise ValueError(f'Incomplete download: {downloaded} bytes vs official {asset["size"]}; retained .part for inspection')
    upstream_digest = asset.get('digest')
    if upstream_digest and upstream_digest != f'sha256:{digest.hexdigest()}':
        raise ValueError('Official asset digest does not match download')
    temporary.rename(path)
    metadata = {'url': url, 'release_url': release['html_url'], 'release_tag': release['tag_name'],
                'release_published_at': release['published_at'], 'asset_id': asset['id'],
                'filename': filename, 'bytes': downloaded, 'sha256': digest.hexdigest(),
                'upstream_digest': upstream_digest, 'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                'training_data': 'COCO', 'source_num_classes': 80, 'purpose': 'detector_pretraining_not_blister_finetuning',
                'same_binary_as_paper': 'unknown_no_paper_hash'}
    path.with_suffix('.sha256.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata))
