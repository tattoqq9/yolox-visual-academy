#!/usr/bin/env python3
"""Offline deterministic browser smoke for Unit 06. Optional Playwright + local Chromium.

The local test environment blocks file:// and localhost. Embed the exact shipped
CSS/JS and sample PNG bytes in an otherwise identical HTML document to test UI logic.
This is NOT a substitute for a public GitHub Pages GUI test.
"""
from __future__ import annotations
import base64
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/'site/lab/06-tiny'
OUT=ROOT/'out/unit06/gui'
OUT.mkdir(parents=True,exist_ok=True)
html=(LAB/'index.html').read_text('utf-8')
css=(LAB/'styles.css').read_text('utf-8')
math=(LAB/'math.js').read_text('utf-8')
app=(LAB/'app.js').read_text('utf-8')
data='data:image/png;base64,'+base64.b64encode((LAB/'synthetic-tiny.png').read_bytes()).decode('ascii')
html=html.replace('<link rel="stylesheet" href="styles.css">','<style>'+css+'</style>')
html=html.replace('src="synthetic-tiny.png"',f'src="{data}"')
html=html.replace('<script src="math.js"></script><script src="app.js?v=20261105-unit06"></script>','')
app=app.replace("bg.src='synthetic-tiny.png'",f"bg.src='{data}'")

with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    for name,w,h in (('desktop',1365,900),('mobile',390,844)):
        page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.set_content(html,wait_until='domcontentloaded')
        page.add_script_tag(content=math)
        page.add_script_tag(content=app)
        assert page.locator('#geometryCount').inner_text().strip().isdigit()
        assert page.locator('#candidateRows tr').count()>0
        assert 'Score' in page.locator('#diagnosis').inner_text()
        score_low=page.locator('#scorePass').inner_text()
        page.locator('#scoreLevel').select_option('high')
        assert page.locator('#scorePass').inner_text()!=score_low
        page.locator('#stride').select_option('4')
        assert page.locator('#geometryCount').inner_text().isdigit()
        page.locator('#boxSize').fill('2')
        assert page.locator('#boxSizeLabel').inner_text()=='2 px'
        page.locator('#centerX').fill('61')
        assert page.locator('#centerXLabel').inner_text()=='61 px'
        page.locator('#reveal').click()
        assert page.locator('#comparison').is_hidden()
        page.locator('#idea').fill('4つの数値を参照点から左右上下それぞれへの符号付き距離として予測する仮説です。')
        page.locator('#reveal').click()
        assert page.locator('#comparison').is_visible()
        assert 'IoU 1.000' in page.locator('#signedIoU').inner_text()
        page.locator('#rightDelta').fill('-8')
        assert 'INVALID' in page.locator('#signedIoU').inner_text()
        page.locator('#twDelta').fill('1')
        assert 'IoU' in page.locator('#yoloIoU').inner_text()
        page.locator('#rightDelta').fill('0')
        page.locator('#twDelta').fill('0')
        page.screenshot(path=str(OUT/f'unit06_{name}.png'),full_page=True)
        overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth+1')
        assert not overflow,f'{name} horizontal overflow'
        assert not errors,errors
        print('PASS',name,'GUI, no JS error, no horizontal overflow, screenshot',OUT/f'unit06_{name}.png')
        page.close()
    browser.close()
