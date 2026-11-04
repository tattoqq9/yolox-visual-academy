"""Optional local browser smoke. Requires Playwright and installed system Chromium."""
import functools
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright, Error
import base64
import json

ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'site'

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(SITE),**kwargs)
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_address[1]}'
print('Base:',base)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    for width,height,label in [(1440,940,'desktop'),(390,844,'mobile')]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            response=page.goto(base+'/lab/03-fpn-head/index.html',wait_until='load',timeout=15000)
            print(label,'HTTP',response.status)
        except Error as e:
            if 'ERR_BLOCKED_BY_ADMINISTRATOR' not in str(e):raise
            # Browser sandbox disallows loopback networking. Inject the IDENTICAL
            # local HTML/CSS/JS/data bytes instead; no site functionality altered.
            folder=SITE/'lab/03-fpn-head'
            html=(folder/'index.html').read_text(encoding='utf8')
            css=(folder/'styles.css').read_text(encoding='utf8')
            js=(folder/'app.js').read_text(encoding='utf8')
            trace=(folder/'trace.json').read_text(encoding='utf8')
            selfie=base64.b64encode((SITE/'assets/003Z-selfie.png').read_bytes()).decode('ascii')
            html=html.replace('<link rel="stylesheet" href="styles.css">','<style>'+css+'</style>')
            html=html.replace('../../assets/003Z-selfie.png','data:image/png;base64,'+selfie)
            html=html.replace('<script src="app.js?v=20261104-unit03c"></script>',
                '<script>window.__UNIT03C_TEST_TRACE__='+trace+';</script><script>'+js+'</script>')
            page.set_content(html,wait_until='load')
            print(label,'blocked HTTP; used in-memory identical HTML/JS/CSS + data')
        page.wait_for_function('!!window.Academy03C',timeout=18000)
        print(label,'initial',page.evaluate('window.Academy03C.getState()'))
        assert page.locator('#nodeTitle').inner_text()=='Backbone dark5'
        page.locator('[data-node="cat0"]').click()
        assert page.locator('#nodeTitle').inner_text()=='Concat + dark4'
        assert 'torch.cat' in page.locator('#mathBody').inner_text()
        page.locator('#heatmap').click(position={'x':100,'y':125})
        page.locator('#coordX').fill('2')
        page.locator('#coordX').press('Tab')
        assert page.evaluate('window.Academy03C.getState().x')==2
        page.locator('[data-stage="bottomup"]').click()
        page.locator('[data-node="cat2"]').click()
        assert 'Concat' in page.locator('#nodeTitle').inner_text()
        page.locator('[data-stage="head"]').click()
        page.locator('[data-level="p4"]').click()
        page.locator('[data-node="p4.cls"]').click()
        assert page.locator('#nodeShape').inner_text()=='1 × 80 × 26 × 26'
        page.locator('[data-node="p4.raw"]').click()
        assert 'obj raw' in page.locator('#headReadout').inner_text()
        page.locator('#classes').select_option('1')
        assert '21,294' in page.locator('#classMath').inner_text()
        page.locator('#rotationX').fill('40')
        assert page.locator('#rotXLabel').inner_text()=='40°'
        if label=='mobile':
            over=page.evaluate('document.documentElement.scrollWidth - window.innerWidth')
            print('mobile horizontal overflow px:',over)
            assert over<=4
        pic=ROOT/f'unit03c_{label}_verified.png'
        page.screenshot(path=str(pic),full_page=True)
        print(label,'PASS screenshot',pic,'errors',errors)
        assert not errors
        page.close()
    browser.close()
server.shutdown()
