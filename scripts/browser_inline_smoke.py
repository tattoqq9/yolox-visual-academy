#!/usr/bin/env python3
"""Browser UI test without navigation to localhost/file URLs.
Reads *our own local project files* and embeds dependencies into the page with data:
URIs, so Chromium can render with page.set_content(). It does not test HTTP routing;
GitHub Actions browser_smoke.py provides that after publication.
"""
from __future__ import annotations
import base64, mimetypes, os, re, shutil, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]/'site'
SNAP=Path(__file__).resolve().parents[1]/'checks'

def packed(path: Path):
    mime=mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}'

def inline(rel):
    file=ROOT/rel
    html=file.read_text()
    directory=file.parent
    def link(m):
        href=m.group(1)
        return '<style>\n'+(directory/href).resolve().read_text()+'\n</style>'
    html=re.sub(r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*/?>',link,html)
    def script(m):
        src=m.group(1)
        return '<script>\n'+(directory/src).resolve().read_text()+'\n</script>'
    html=re.sub(r'<script\s+src="([^"]+)"\s*>\s*</script>',script,html)
    # Replace literal image references both in markup and in JavaScript URLs.
    pattern=r'(?<![\w/])(?:\.\./)*(?:assets|01-cnn/assets)/[\w.-]+\.(?:png|jpe?g|webp)'
    def datauri(m):
        name=m.group(0)
        asset=(directory/name).resolve()
        return packed(asset) if asset.is_file() else name
    html=re.sub(pattern,datauri,html)
    return html

def run():
    SNAP.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=shutil.which('chromium'),headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        exceptions=[]
        page.on('pageerror',lambda e:exceptions.append(str(e)))
        page.set_content(inline('lab/03-blocks/index.html'),wait_until='load')
        page.wait_for_function('window.YVA03?.getDiagnostics?.() !== null')
        assert page.evaluate('YVA03.getState().image')=='photo'
        assert page.evaluate('YVA03.getDiagnostics().shape')==[32,48,3]
        page.screenshot(path=str(SNAP/'unit03a-desktop.png'))
        for stage in range(1,6):
            page.locator(f'[data-stage="{stage}"]').click()
            assert page.evaluate('YVA03.getState().stage')==stage
            assert page.evaluate('YVA03.getDiagnostics().shape')==[16,24,4]
            if stage==1:
                d=page.evaluate('YVA03.getDiagnostics()')
                assert len(d['convTerms'])==27
                assert abs(sum(q['product'] for q in d['convTerms'])-d['value'])<1e-9
                page.locator('#termNext').click()
                assert '1/27' in page.locator('#explanation').inner_text()
            if stage==2:
                before=page.evaluate('YVA03.getDiagnostics().value')
                page.locator('#bn_beta').fill('0.3')
                after=page.evaluate('YVA03.getDiagnostics().value')
                assert abs(after-before-.3)<1e-8,(before,after)
            if stage==4:
                page.screenshot(path=str(SNAP/'unit03a-csp.png'))
                before=page.evaluate('YVA03.getDiagnostics().value')
                page.locator('#shortcutToggle').click()
                after=page.evaluate('YVA03.getDiagnostics().value')
                assert abs(before-after)>1e-5,(before,after)
            if stage==5:
                for id,value in [('answerW','24'),('answerH','16'),('answerC','4')]:page.locator('#'+id).fill(value)
                page.locator('#checkAnswer').click()
                assert '正解' in page.locator('#answerFeedback').inner_text()
        page.set_viewport_size({'width':390,'height':844})
        page.set_content(inline('lab/03-blocks/index.html'),wait_until='load')
        page.wait_for_function('window.YVA03?.getDiagnostics?.() !== null')
        page.screenshot(path=str(SNAP/'unit03a-mobile.png'),full_page=True)
        overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 3')
        assert not overflow,'horizontal overflow on mobile'
        assert not exceptions,exceptions
        browser.close()
    print('PASS: inline Chromium UI — 6 stages, slider, CSP toggle, responsive layout')

if __name__=='__main__':
    try: run()
    except Exception as e:
        print(f'FAIL: {e}',file=sys.stderr);raise
