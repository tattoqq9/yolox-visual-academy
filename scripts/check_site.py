#!/usr/bin/env python3
"""Standard-library-only checks for an offline-first static educational website."""
import sys
from pathlib import Path
from urllib.parse import urlsplit, unquote
import re
from html.parser import HTMLParser

class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.add(a['id'])
        for k in ('href','src','poster'):
            if a.get(k): self.refs.append((k, a[k]))

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'site').resolve()
    failures = []
    if not (root/'index.html').is_file():
        print('ERROR: site/index.html がありません',file=sys.stderr); return 2
    htmlfiles=list(root.rglob('*.html'))
    parser_map={}
    for file in htmlfiles:
        p=References()
        try: p.feed(file.read_text(encoding='utf-8'))
        except Exception as e: failures.append(f'{file.relative_to(root)}: UTF-8/HTML issue: {e}')
        parser_map[file.resolve()]=p
    total=0
    for file,p in parser_map.items():
        for typ,raw in p.refs:
            sp=urlsplit(raw)
            if sp.scheme or sp.netloc or not sp.path and not sp.fragment: continue
            if sp.path.startswith('/'):
                failures.append(f'{file.relative_to(root)}: root-absolute URL not portable on project Pages: {raw}')
                continue
            if sp.path:
                target=(file.parent / unquote(sp.path)).resolve()
                if root != target and root not in target.parents:
                    failures.append(f'{file.relative_to(root)}: path traverses outside site: {raw}')
                    continue
                if target.is_dir(): target=target/'index.html'
                if not target.is_file():
                    failures.append(f'{file.relative_to(root)}: missing {raw}')
                    continue
                total+=1
            else: target=file
            if sp.fragment and target.suffix.lower()=='.html' and target in parser_map and sp.fragment not in parser_map[target].ids:
                # Unit 02 uses #step-N as a JavaScript router; these are not DOM ids.
                if target == (root/'lab/02-grid/index.html') and re.fullmatch(r'step-[1-5]',sp.fragment): continue
                failures.append(f'{file.relative_to(root)}: anchor not found: {raw}')
    for mandatory in ('lab/01-cnn/index.html','lab/02-grid/index.html','lab/03-block/index.html',
                      'learn/01-cnn.html','learn/02-stride-grid.html','learn/03-block.html',
                      'downloads/unit01_numpy_starter.zip','downloads/unit02_numpy_starter.zip','downloads/unit03_numpy_starter.zip',
                      '.nojekyll'):
        if not (root/mandatory).is_file(): failures.append(f'missing essential: {mandatory}')
    for file in root.rglob('*'):
        if file.is_symlink(): failures.append(f'symlink prohibited: {file.relative_to(root)}')
    print(f'HTML: {len(htmlfiles)} / local references: {total} / site files: {sum(1 for f in root.rglob("*") if f.is_file())}')
    if failures:
        print('FAILURES:')
        for f in failures: print(' -',f)
        return 1
    print('PASS: site構造とHTML内リンクを検証しました')
    return 0

if __name__ == '__main__': sys.exit(main())
