"""Extract source evidence without modifying the supplied DOCX (stdlib only)."""
from pathlib import Path
import argparse
import json
import zipfile
import xml.etree.ElementTree as ET

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'v': 'urn:schemas-microsoft-com:vml',
      'o': 'urn:schemas-microsoft-com:office:office'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, default=Path('source_evidence'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        rels = {r.attrib['Id']: r.attrib for r in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
        paragraphs = []
        lines = []
        for i, p in enumerate(root.findall('.//w:body//w:p', NS)):
            text = ''.join(n.text or '' for n in p.iter() if n.tag in {f'{{{NS["w"]}}}t', f'{{{NS["m"]}}}t'})
            media = []
            for node in p.findall('.//v:imagedata', NS):
                rid = node.get(f'{{{NS["r"]}}}id')
                if rid in rels:
                    target = rels[rid]['Target']
                    media.append(target)
                    dest = args.out / target
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read('word/' + target))
            paragraphs.append({'id': i, 'text': text, 'media': media})
            if text or media:
                lines.append(f'[P{i}] {text}' + (f' [MEDIA: {", ".join(media)}]' if media else ''))
        tables = []
        for table in root.findall('.//w:tbl', NS):
            rows = []
            for row in table.findall('w:tr', NS):
                rows.append([''.join(t.text or '' for t in cell.findall('.//w:t', NS))
                             for cell in row.findall('w:tc', NS)])
            tables.append(rows)
        links = [r['Target'] for r in rels.values() if r.get('TargetMode') == 'External']
        (args.out/'paragraphs.json').write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out/'paper.txt').write_text('\n'.join(lines), encoding='utf-8')
        (args.out/'tables.json').write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out/'links.json').write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Extracted {len(paragraphs)} paragraphs, {len(tables)} tables, {len(links)} links to {args.out}')

if __name__ == '__main__':
    main()
