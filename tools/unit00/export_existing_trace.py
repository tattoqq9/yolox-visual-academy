#!/usr/bin/env python3
"""Derive a small, explicitly untrained teaching sample from existing public traces.

Pure source transformation, no model loads, no training. Run from repository root.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_BACKBONE = ROOT / 'site/lab/03-nano/trace.json'
SRC_NECK = ROOT / 'site/lab/03-fpn-head/trace.json'
OUT = ROOT / 'site/lab/00-overview/actual-untrained-trace.json'


def grid_sample(values, width, height, target_width=26):
    """Nearest index subsampling (not pooled/interpolated, so each saved float is source exact)."""
    target_height = max(1, round(target_width * height / width))
    if len(values) != width * height:
        raise ValueError(f'shape mismatch: {len(values)} != {width} * {height}')
    subset = [values[min(height - 1, int((y + .5) * height / target_height)) * width
                     + min(width - 1, int((x + .5) * width / target_width))]
              for y in range(target_height) for x in range(target_width)]
    return {'width': target_width, 'height': target_height, 'values': subset,
            'method': 'nearest-input-position, no interpolation or pooling'}


def main():
    bb = json.loads(SRC_BACKBONE.read_text(encoding='utf8'))
    neck = json.loads(SRC_NECK.read_text(encoding='utf8'))
    assert bb['provenance']['trained'] is False and neck['provenance']['learned'] is False
    assert bb['preprocess']['originalWH'] == neck['input']['originalWH']
    selected = []
    for stage, title in [('dark2', 'Backbone dark2'), ('dark3', 'Backbone dark3')]:
        node = bb['stages'][stage]
        _, channels, height, width = node['shape']
        chan = node['channels'][0]
        selected.append({'id': stage, 'name': title, 'source': '03-nano/trace.json',
                         'shape': node['shape'], 'channel': chan['index'],
                         'sample': grid_sample(chan['values'], width, height)})
    node = neck['nodes']['p3']
    _, channels, height, width = node['shape']
    chan = node['samples'][0]
    selected.append({'id': 'p3', 'name': 'PAFPN P3', 'source': '03-fpn-head/trace.json',
                     'shape': node['shape'], 'channel': chan['index'],
                     'sample': grid_sample(chan['values'], width, height)})
    payload = {
        'schema': 'yva.unit00.untrained-preview.v1',
        'type': 'real-pytorch-forward-subsamples',
        'trained': False,
        'important': 'Untrained random seed weights: numerical activations, NOT learned head features or detections.',
        'input': {'image': '../../assets/003Z-selfie.png', 'originalWH': bb['preprocess']['originalWH'],
                  'modelInput': bb['preprocess']['inputWH'], 'preprocess': bb['preprocess']['channel_order'],
                  'seed': bb['provenance']['seed']},
        'sha256': {'backbone': hashlib.sha256(SRC_BACKBONE.read_bytes()).hexdigest(),
                   'neck': hashlib.sha256(SRC_NECK.read_bytes()).hexdigest()},
        'maps': selected
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf8')
    print(f'Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
