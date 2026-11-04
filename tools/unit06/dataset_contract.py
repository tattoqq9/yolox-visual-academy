"""Local-only COCO bbox dataset validator for later formal dataset replacement.

No image copying, upload, server access, or publication. It writes a PRIVATE manifest
with per-image hashes into out/unit06 (gitignored), or a location outside the repo.
Works with JPEG/PNG without third-party dependencies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'yva.unit06.dataset-manifest.v1'


def _int(x: Any, label: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f'{label}: integer ID required')
    return x


def image_dimensions(path: Path) -> tuple[int, int]:
    """Decode only image headers; do not read image pixels or emit previews."""
    with path.open('rb') as fp:
        head = fp.read(24)
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            if head[12:16] != b'IHDR':
                raise ValueError(f'{path.name}: invalid PNG header')
            return struct.unpack('>II', head[16:24])
        if head[:2] != b'\xff\xd8':
            raise ValueError(f'{path.name}: only PNG/JPEG supported')
        fp.seek(2)
        while True:
            marker = fp.read(1)
            if not marker:
                raise ValueError(f'{path.name}: JPEG dimensions not found')
            if marker != b'\xff':
                raise ValueError(f'{path.name}: invalid JPEG segment')
            while (code := fp.read(1)) == b'\xff':
                pass
            if not code or code == b'\xd9':
                raise ValueError(f'{path.name}: missing JPEG SOF')
            if code in (b'\x01', *[bytes([i]) for i in range(0xd0, 0xd8)]):
                continue
            size_b = fp.read(2)
            if len(size_b) != 2:
                raise ValueError(f'{path.name}: truncated JPEG segment')
            seglen = int.from_bytes(size_b, 'big')
            if seglen < 2:
                raise ValueError(f'{path.name}: invalid JPEG segment length')
            if code[0] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                if seglen < 7:
                    raise ValueError(f'{path.name}: invalid JPEG SOF')
                data = fp.read(5)
                if len(data) != 5:
                    raise ValueError(f'{path.name}: truncated JPEG SOF')
                return (int.from_bytes(data[3:5], 'big'), int.from_bytes(data[1:3], 'big'))
            fp.seek(seglen - 2, 1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _within(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def validate(source: dict[str, Any], image_root: Path, split: str,
             previous: list[dict[str, Any]] | None = None,
             rights_reviewed: bool = False) -> dict[str, Any]:
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train / val / test')
    if not isinstance(source, dict):
        raise ValueError('COCO annotations must be a JSON object')
    root = image_root.resolve(strict=True)
    categories = source.get('categories')
    images = source.get('images')
    anns = source.get('annotations')
    if not isinstance(categories, list) or not categories:
        raise ValueError('COCO categories must be a nonempty list')
    if not isinstance(images, list) or not images or not isinstance(anns, list):
        raise ValueError('COCO images must be nonempty and annotations must be a list')
    catmap: dict[int, str] = {}
    for c in categories:
        key = _int(c['id'], 'category.id')
        val = c['name']
        if key in catmap or not isinstance(val, str) or not val.strip():
            raise ValueError('duplicate or invalid category')
        catmap[key] = val.strip()
    if len(catmap) != len(set(catmap.values())):
        raise ValueError('duplicate category names')
    image_info: dict[int, dict] = {}
    digests = set()
    files = set()
    for im in images:
        iid = _int(im['id'], 'image.id')
        fn = im['file_name']
        width = _int(im['width'], 'image.width')
        height = _int(im['height'], 'image.height')
        if iid in image_info or width <= 0 or height <= 0:
            raise ValueError('duplicate image ID / invalid dimensions')
        if not isinstance(fn, str) or not fn or '\\' in fn or Path(fn).is_absolute():
            raise ValueError('unsafe filename')
        path = (root / fn).resolve(strict=True)
        if not _within(path, root) or not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError(f'invalid image path {fn}')
        if fn in files:
            raise ValueError('duplicate image path')
        files.add(fn)
        actual = image_dimensions(path)
        if actual != (width, height):
            raise ValueError(f'{fn}: declared dimensions {(width,height)} != actual {actual}')
        digest = sha256_file(path)
        if digest in digests:
            raise ValueError('duplicate image content in one split')
        digests.add(digest)
        image_info[iid] = {'id': iid, 'file_name': fn, 'width': width, 'height': height,
                           'sha256': digest, 'annotations': []}
    annotation_ids = set()
    for a in anns:
        aid = _int(a['id'], 'annotation.id')
        iid = _int(a['image_id'], 'annotation.image_id')
        cid = _int(a['category_id'], 'annotation.category_id')
        if aid in annotation_ids or iid not in image_info or cid not in catmap:
            raise ValueError('duplicate annotation ID / unknown image or category')
        annotation_ids.add(aid)
        bbox = a.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) for q in bbox
        ):
            raise ValueError(f'annotation {aid}: invalid bbox')
        x, y, w, h = bbox
        im = image_info[iid]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > im['width']+1e-8 or y+h > im['height']+1e-8:
            raise ValueError(f'annotation {aid}: empty or out-of-bounds bbox')
        for flag in ('ignore', 'iscrowd'):
            if a.get(flag, 0) not in (0, 1, False, True):
                raise ValueError(f'annotation {aid}: invalid {flag}')
        im['annotations'].append({
            'id': aid, 'category_id': cid, 'bbox_xywh_px': [float(z) for z in bbox],
            'ignore': bool(a.get('ignore', 0)), 'iscrowd': bool(a.get('iscrowd', 0))
        })
    if previous:
        for m in previous:
            if m.get('schema') != SCHEMA or m.get('split') == split:
                raise ValueError('compare manifest must be from another split, same schema')
            other = {i['sha256'] for i in m['images']}
            leaked = digests & other
            if leaked:
                raise ValueError(f'train/val/test leakage: {len(leaked)} duplicate image hash(es)')
    # Publication must be a separate, explicitly reviewed process.
    result = {'schema': SCHEMA, 'split': split, 'rights_reviewed': bool(rights_reviewed),
              'publishable': False, 'image_count': len(images), 'annotation_count': len(anns),
              'categories': [{'id': k, 'name': v} for k, v in sorted(catmap.items())],
              'images': sorted(image_info.values(), key=lambda v: v['id'])}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Local-only COCO validation. Output is PRIVATE and is never published.')
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--output', default=None, type=Path)
    parser.add_argument('--compare-manifest', action='append', type=Path, default=[])
    parser.add_argument('--rights-reviewed', action='store_true', help='Records only the user declaration; does not permit publication')
    args = parser.parse_args(argv)
    output = (args.output or ROOT/'out'/'unit06'/f'dataset_manifest_{args.split}.json').resolve()
    if _within(output, ROOT) and not _within(output, ROOT/'out'):
        parser.error('output must NOT be stored in the public repository; use out/ (gitignored) or an external private folder')
    if args.annotations.resolve() == output:
        parser.error('refusing to overwrite annotation input')
    prev = [json.loads(p.read_text('utf-8')) for p in args.compare_manifest]
    source = json.loads(args.annotations.read_text('utf-8'))
    result = validate(source, args.images, args.split, prev, args.rights_reviewed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'PASS {args.split}: {result["image_count"]} images / {result["annotation_count"]} annotations, private {output}')
    print('PUBLICATION: NOT ALLOWED by this tool; external human rights/license review required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
