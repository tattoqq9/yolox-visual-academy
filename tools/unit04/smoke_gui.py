#!/usr/bin/env python3
"""Offline embedded-assets GUI smoke for environments blocking local HTTP origin.

This test inlines the *same* HTML/CSS/JS + real untrained trace and preprocessed photo.
It bypasses URL/network routing restrictions; it is not a deployed Pages test.
Requires Playwright and a local Chromium executable.
"""
from __future__ import annotations
import base64, json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/'site/lab/04-detection'
OUT=ROOT/'out/unit04'
OUT.mkdir(parents=True,exist_ok=True)


def bundle():
    html=(LAB/'index.html').read_text(encoding='utf-8')
    css=(LAB/'styles.css').read_text(encoding='utf-8')
    mathjs=(LAB/'math.js').read_text(encoding='utf-8')
    appjs=(LAB/'app.js').read_text(encoding='utf-8')
    trace=(LAB/'trace04.json').read_text(encoding='utf-8')
    self_image=base64.b64encode((ROOT/'site/assets/003Z-selfie.png').read_bytes()).decode()
    input_image=base64.b64encode((ROOT/'site/lab/03-nano/input416.jpg').read_bytes()).decode()
    assert '</script' not in mathjs.lower() and '</script' not in appjs.lower()
    html=html.replace('<link rel="stylesheet" href="styles.css">',f'<style>{css}</style>')
    html=html.replace('../../assets/003Z-selfie.png','data:image/png;base64,'+self_image)
    shim='''<script>window.fetch=async function(url){if(url==='trace04.json')return {ok:true,status:200,json:async()=>window.__YVA_TRACE};throw new Error('offline smoke fetch not allowed: '+url)};window.__YVA_TRACE='''+trace+';</script>'
    html=html.replace('</head>',shim+'</head>')
    appjs=appjs.replace("img.src='../03-nano/input416.jpg';",f"img.src='data:image/jpeg;base64,{input_image}';")
    html=html.replace('<script src="math.js?v=20261104-unit04"></script>',f'<script>{mathjs}</script>')
    html=html.replace('<script src="app.js?v=20261104-unit04"></script>',f'<script>{appjs}</script>')
    return html


def test_page(page,device):
    errs=[]
    page.on('pageerror',lambda err:errs.append(str(err)))
    page.set_content(bundle(),wait_until='load',timeout=40000)
    page.locator('#originFlag').wait_for()
    page.wait_for_function("document.querySelector('#headLogits').textContent.includes('obj_raw')",timeout=10000)
    assert page.locator('#originFlag').inner_text()=='実PyTorch値'
    assert page.locator('#decodedMetrics').locator('.metric').count()==4
    page.locator('#decodeLevel').select_option('p4')
    assert page.locator('#gridTotal').inner_text()=='26×26'
    page.locator('#rawTx').evaluate('(e)=>{e.value="0.5";e.dispatchEvent(new Event("input",{bubbles:true}))}')
    assert '仮想編集' in page.locator('#originFlag').inner_text()
    page.locator('#restoreRaw').click()
    assert page.locator('#originFlag').inner_text()=='実PyTorch値'
    page.locator('[data-step="nms"]').click()
    assert page.locator('#nmsRows tr').count()==8
    count_before=page.locator('#nmsSummary').inner_text()
    page.locator('#nmsClassAware').uncheck()
    count_after=page.locator('#nmsSummary').inner_text()
    assert count_before != count_after,(count_before,count_after)
    page.locator('[data-step="simota"]').click()
    assert page.locator('#simotaRows tr').count()>0
    dynamic=page.locator('#simotaMetrics').inner_text()
    assert 'GT A' in dynamic
    page.locator('#simotaGT').select_option('1')
    page.locator('#simotaX').evaluate('(e)=>{e.value="280";e.dispatchEvent(new Event("input",{bubbles:true}))}')
    assert '280px' in page.locator('#simotaXOut').inner_text()
    page.locator('#simotaRows tr').first.click()
    assert '選択候補' in page.locator('#simotaExplanation').inner_text() or '候補 ' in page.locator('#simotaExplanation').inner_text()
    page.locator('[data-step="loss"]').click()
    assert 'Loss合計' in page.locator('#lossMetrics').inner_text()
    loss_before=page.locator('#lossMetrics').inner_text()
    page.locator('#lossBoxSize').select_option('4')
    page.locator('#lossShift').evaluate('(e)=>{e.value="1";e.dispatchEvent(new Event("input",{bubbles:true}))}')
    loss_after=page.locator('#lossMetrics').inner_text()
    assert loss_before!=loss_after
    assert 'd L_obj' in page.locator('#lossGradients').inner_text()
    doc_width=page.evaluate('document.documentElement.scrollWidth'),page.evaluate('window.innerWidth')
    assert doc_width[0]<=doc_width[1]+2,f'mobile horizontal overflow: {doc_width}'
    page.screenshot(path=str(OUT/f'unit04_{device}_verified.png'),full_page=True,animations='disabled')
    assert not errs,f'Uncaught browser exceptions: {errs}'
    return {'device':device,'title':page.title(),'nms_before':count_before,'nms_after':count_after,'final_overflow':doc_width,'errors':errs}


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
        try:
            for device,view in [('desktop',{'width':1440,'height':900}),('mobile',{'width':390,'height':844})]:
                page=browser.new_page(viewport=view,device_scale_factor=1)
                print(json.dumps(test_page(page,device),ensure_ascii=False,indent=2),flush=True)
                page.close()
        finally:browser.close()
    print('PASS: embedded-assets UI operations, no network/deployment claims')

if __name__=='__main__':main()
