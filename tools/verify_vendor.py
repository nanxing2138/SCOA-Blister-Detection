"""Verify that the actual detector/SimOTA implementation has not been silently rewritten."""
import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'third_party/model_file_sha256.json').read_text(encoding='utf-8'))
    for filename, expected in manifest.items():
        original = root / 'third_party/YOLOX/yolox/models' / filename
        assert hashlib.sha256(original.read_bytes()).hexdigest() == expected, filename
        modified = original.read_text(encoding='utf-8').replace(
            'from yolox.utils import bboxes_iou, meshgrid', 'from scoa.geometry import bboxes_iou, meshgrid')
        assert (root / 'scoa/_vendor' / filename).read_text(encoding='utf-8') == modified, filename
    return {'verified_files': list(manifest), 'allowed_change': 'head utils import only'}


if __name__ == '__main__':
    print(json.dumps(verify()))
