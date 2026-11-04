#!/usr/bin/env python3
"""Run local browser smoke tests on Units 01-03 using Playwright Chromium.
Requires: pip install playwright; playwright install chromium.
Usage: python3 scripts/browser_smoke.py [--base http://127.0.0.1:8765/]
The server should already be running: bash scripts/preview.sh
"""
import argparse, os, shutil, sys
from pathlib import Path
from playwright.sync_api import sync_playwright


def check(base, snapshot_dir=None):
    with sync_playwright() as p:
        launch={'headless':True,'args':['--no-sandbox']}
        binary=os.getenv('YVA_CHROMIUM')
        if binary is None and os.getenv('YVA_USE_SYSTEM_CHROMIUM')=='1':
            binary=shutil.which('chromium') or shutil.which('google-chrome')
        if binary: launch['executable_path']=binary
        browser=p.chromium.launch(**launch)
        page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        paths=['','lab/01-cnn/index.html','lab/02-grid/index.html','lab/03-blocks/index.html','learn/03-blocks.html']
        for path in paths:
            resp=page.goto(base.rstrip('/')+'/'+path,wait_until='load')
            assert resp is not None and resp.status==200,(path,resp.status if resp else None)
            assert page.locator('body').inner_text().strip(),path
            if snapshot_dir:
                Path(snapshot_dir).mkdir(parents=True,exist_ok=True)
                key=path.replace('/','-') or 'home'
                page.screenshot(path=str(Path(snapshot_dir)/(key+'.png')),full_page=False)
        page.goto(base.rstrip('/')+'/lab/01-cnn/index.html')
        page.wait_for_function("window.YOLOXLab?.getCurrent?.().image === 'zselfie'")
        page.locator('.chapter[data-step="1"]').click()
        assert page.locator('#lessonTitle').inner_text()
        page.goto(base.rstrip('/')+'/lab/02-grid/index.html')
        page.locator('.image-choice.selected').wait_for()
        assert page.locator('.image-choice.selected').get_attribute('data-image')=='zselfie'
        page.locator('[data-step="2"]').click()
        assert page.locator('#gridMapShape').inner_text().startswith('52×52')
        page.goto(base.rstrip('/')+'/lab/03-blocks/index.html')
        page.wait_for_function('window.YVA03?.getDiagnostics?.()?.shape?.[0] === 32')
        assert page.evaluate('YVA03.getState().image')=='photo'
        assert page.locator('.hero-image img').get_attribute('src')=='../../assets/003Z-selfie.png'
        page.locator('[data-stage="1"]').click()
        page.wait_for_function('window.YVA03?.getDiagnostics?.()?.shape?.[0] === 16')
        d=page.evaluate('YVA03.getDiagnostics()')
        assert d['shape']==[16,24,4]
        assert len(d['convTerms'])==27
        assert abs(sum(t['product'] for t in d['convTerms'])-d['value'])<1e-10
        page.locator('#termNext').click()
        assert '1/27' in page.locator('#explanation').inner_text()
        page.locator('[data-stage="2"]').click()
        bn1=page.evaluate('YVA03.getDiagnostics().value')
        page.locator('#bn_beta').evaluate("e => { e.value='0.3'; e.dispatchEvent(new Event('input', {bubbles:true})); }")
        bn2=page.evaluate('YVA03.getDiagnostics().value')
        assert abs(bn2-bn1-.3)<1e-10,(bn1,bn2)
        page.locator('[data-stage="4"]').click()
        x=page.evaluate('YVA03.getDiagnostics().value')
        page.locator('#shortcutToggle').click()
        y=page.evaluate('YVA03.getDiagnostics().value')
        assert abs(x-y)>1e-5,(x,y)
        page.locator('[data-stage="5"]').click()
        page.locator('#answerW').fill('24')
        page.locator('#answerH').fill('16')
        page.locator('#answerC').fill('4')
        page.locator('#checkAnswer').click()
        assert '正解' in page.locator('#answerFeedback').inner_text()
        if snapshot_dir:
            page.goto(base.rstrip('/')+'/lab/03-blocks/index.html#step-5');page.wait_for_function('window.YVA03?.getState?.().stage === 4');page.screenshot(path=str(Path(snapshot_dir)/'03-csp.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            page.goto(base.rstrip('/')+'/lab/03-blocks/index.html');page.wait_for_function('window.YVA03?.getDiagnostics?.() !== null');page.screenshot(path=str(Path(snapshot_dir)/'03-mobile.png'),full_page=True)
            overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 3')
            assert not overflow,'unexpected horizontal overflow on mobile'
        assert not errors,errors
        browser.close()
        print('PASS: browser smoke Units 01, 02, 03-A; no JavaScript exceptions')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8765/');parser.add_argument('--snapshots');args=parser.parse_args()
    try: check(args.base,args.snapshots)
    except Exception as e:
        print(f'FAIL: browser smoke: {e}',file=sys.stderr)
        raise
