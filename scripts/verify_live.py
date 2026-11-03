#!/usr/bin/env python3
"""Verify a deployed GitHub Pages URL using real HTTP GET requests."""
import sys, time, argparse
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from urllib.error import URLError, HTTPError
PATHS={
    'index.html':b'YOLOX Visual Academy',
    'lab/01-cnn/index.html':b'YOLOX',
    'lab/02-grid/index.html':b'YOLOX',
    'learn/01-cnn.html':b'YOLOX',
    'learn/02-stride-grid.html':b'YOLOX',
    'assets/site.css':None,
    'lab/01-cnn/app.js':None,
    'lab/02-grid/app.js':None,
    'downloads/unit01_numpy_starter.zip':None,
    'downloads/unit02_numpy_starter.zip':None,
}
def check(base):
    failures=[]
    for path,marker in PATHS.items():
        url=urljoin(base,path)
        try:
            with urlopen(Request(url,headers={'User-Agent':'YOLOX-Academy-Pages-Smoke/1.0','Cache-Control':'no-cache'}),timeout=18) as resp:
                data=resp.read()
                status=resp.status
            ok=(status==200 and len(data)>0 and (marker is None or marker in data))
            if ok: print(f'PASS {status}: {url} ({len(data):,} bytes)')
            else: failures.append(f'{status}: {url} ({len(data)} bytes, marker={marker!r})')
        except Exception as e: failures.append(f'{url}: {e}')
    for f in failures: print('FAIL',f,file=sys.stderr)
    return not failures
def main():
    p=argparse.ArgumentParser()
    p.add_argument('base_url',help='e.g. https://USER.github.io/REPO/')
    p.add_argument('--attempts',type=int,default=1)
    p.add_argument('--interval',type=int,default=10)
    a=p.parse_args()
    base=a.base_url.rstrip('/')+'/'
    if not base.startswith('https://'):
        print('公開URL（https://）を指定してください',file=sys.stderr);return 2
    for i in range(a.attempts):
        print(); print(f'Check {i+1}/{a.attempts}: {base}',flush=True)
        if check(base):
            print('PASS: 公開サイトの主要10エンドポイントを確認しました')
            return 0
        if i<a.attempts-1: time.sleep(a.interval)
    print('FAIL: 公開サイトの検証に失敗しました',file=sys.stderr)
    return 1
if __name__=='__main__':sys.exit(main())
