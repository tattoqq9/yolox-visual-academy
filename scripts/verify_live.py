#!/usr/bin/env python3
"""HTTP GET verification for all published YOLOX Visual Academy units.

This is a network/asset check, not a replacement for actual GUI interaction.
Usage: python3 scripts/verify_live.py https://USER.github.io/yolox-visual-academy/ --attempts 6
"""
import argparse
import sys
import time
from urllib.request import Request, urlopen
from urllib.parse import urljoin

PATHS = {
    'index.html': b'YOLOX Visual Academy',
    'assets/site.css': None,
    'assets/site.js': None,
    'assets/curriculum.json': b'03-fpn-head',
    'assets/003Z-selfie.png': b'\x89PNG\r\n\x1a\n',
    'lab/00-overview/index.html': b'UNIT 00',
    'lab/00-overview/styles.css': None,
    'lab/00-overview/app.js': b'YVAUnit00',
    'lab/00-overview/actual-untrained-trace.json': b'yva.unit00.untrained-preview.v1',
    'learn/00-overview.html': b'YOLOX',
    'lab/01-cnn/index.html': b'YOLOX',
    'lab/01-cnn/app.js': None,
    'learn/01-cnn.html': b'YOLOX',
    'downloads/unit01_numpy_starter.zip': b'PK\x03\x04',
    'lab/02-grid/index.html': b'YOLOX',
    'lab/02-grid/app.js': None,
    'learn/02-stride-grid.html': b'YOLOX',
    'downloads/unit02_numpy_starter.zip': b'PK\x03\x04',
    'lab/03-block/index.html': b'YOLOX',
    'lab/03-block/app.js': None,
    'learn/03-block.html': b'YOLOX',
    'downloads/unit03_numpy_starter.zip': b'PK\x03\x04',
    'lab/03-nano/index.html': b'YOLOX',
    'lab/03-nano/app.js': None,
    'lab/03-nano/trace.json': b'yolox-academy.nano-trace.v1',
    'learn/03-nano.html': b'YOLOX',
    'downloads/unit03b_pytorch_starter.zip': b'PK\x03\x04',
    'lab/03-fpn-head/index.html': b'UNIT 03-C',
    'lab/03-fpn-head/app.js': b'yolox-academy.neck-head-trace.v1',
    'lab/03-fpn-head/styles.css': None,
    'lab/03-fpn-head/trace.json': b'yolox-academy.neck-head-trace.v1',
    'learn/03-fpn-head.html': b'YOLOX',
    'downloads/unit03c_pytorch_starter.zip': b'PK\x03\x04',
    'lab/04-detection/index.html': b'BBox',
    'lab/04-detection/math.js': b'classificationCost' if False else b'classifyCost',
    'lab/04-detection/app.js': b'trace04.json',
    'lab/04-detection/styles.css': None,
    'lab/04-detection/trace04.json': b'yva.unit04.untrained-head-raw.v1',
    'learn/04-detection.html': b'BBox',
    'downloads/unit04_python_starter.zip': b'PK\x03\x04',
    'lab/05-training/index.html': b'YOLOX',
    'lab/05-training/app.js': b'yva.unit05.public.v1',
    'lab/05-training/styles.css': None,
    'lab/05-training/trace05.json': b'yva.unit05.public.v1',
    'lab/05-training/val-0.png': b'\x89PNG\r\n\x1a\n',
    'lab/05-training/val-1.png': b'\x89PNG\r\n\x1a\n',
    'lab/05-training/val-2.png': b'\x89PNG\r\n\x1a\n',
    'lab/05-training/val-3.png': b'\x89PNG\r\n\x1a\n',
    'lab/05-training/val-4.png': b'\x89PNG\r\n\x1a\n',
    'lab/05-training/val-5.png': b'\x89PNG\r\n\x1a\n',
    'learn/05-training.html': b'YOLOX',
    'downloads/unit05_pytorch_starter.zip': b'PK\x03\x04',
    'lab/06-tiny/index.html': b'Unit 06',
    'lab/06-tiny/math.js': b'decodeSignedLTRB',
    'lab/06-tiny/app.js': b'YvaMath06',
    'lab/06-tiny/styles.css': None,
    'lab/06-tiny/synthetic-tiny.png': b'\x89PNG\r\n\x1a\n',
    'learn/06-tiny.html': b'YOLOX',
    'downloads/unit06_python_starter.zip': b'PK\x03\x04',

}


def check(base):
    failures=[]
    for path, marker in PATHS.items():
        url=urljoin(base,path)
        try:
            with urlopen(Request(url,headers={'User-Agent':'YOLOX-Academy-Pages-Smoke/2.0',
                                              'Cache-Control':'no-cache'}),timeout=18) as response:
                data=response.read()
                status=response.status
            ok=(status==200 and len(data)>0 and (marker is None or marker in data))
            if ok: print(f'PASS {status}: {url} ({len(data):,} bytes)',flush=True)
            else: failures.append(f'{status}: {url} ({len(data)} bytes, marker={marker!r})')
        except Exception as e:
            failures.append(f'{url}: {e}')
    for f in failures:print('FAIL',f,file=sys.stderr)
    return not failures


def main():
    p=argparse.ArgumentParser()
    p.add_argument('base_url',help='e.g. https://USER.github.io/yolox-visual-academy/')
    p.add_argument('--attempts',type=int,default=1)
    p.add_argument('--interval',type=int,default=10)
    a=p.parse_args()
    base=a.base_url.rstrip('/')+'/'
    if not base.startswith('https://'):
        print('公開URL（https://）を指定してください',file=sys.stderr)
        return 2
    if a.attempts<1 or a.interval<0:
        print('attempts>=1, interval>=0 にしてください',file=sys.stderr)
        return 2
    for i in range(a.attempts):
        print(f'Check {i+1}/{a.attempts}: {base}',flush=True)
        if check(base):
            print(f'PASS: 公開サイトの主要{len(PATHS)}エンドポイントを確認しました')
            print('注意: ブラウザのボタン・キャンバス操作の検証とは別です')
            return 0
        if i<a.attempts-1:time.sleep(a.interval)
    print('FAIL: 公開サイトのHTTP検証に失敗しました',file=sys.stderr)
    return 1

if __name__=='__main__':sys.exit(main())
