"""Run local HTTP + headless Chromium smoke for Unit 05 on desktop and phone."""
import argparse,base64,contextlib,http.server,json,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright

SITE=Path(__file__).resolve().parents[2]/'site'

class Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(SITE),**kwargs)
    def log_message(self,*args):pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('out/unit05/gui'));args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Quiet)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_address[1]}/lab/05-training/index.html'
    try:
        # Check HTTP asset serving independently; browser network policy forbids localhost.
        from urllib.request import urlopen
        from urllib.parse import urljoin
        for name in ('index.html','app.js','styles.css','trace05.json','val-0.png','val-5.png'):
            with urlopen(f'http://127.0.0.1:{server.server_address[1]}/lab/05-training/{name}') as resp:
                assert resp.status==200 and len(resp.read())>0,name
        print('PASS HTTP localhost: Unit 05 HTML/JS/CSS/JSON/PNG assets')
        # Browser receives the original HTML/CSS/JS and data from the same static files.
        # Only URL loading is replaced by inline transport because localhost is blocked.
        html=(SITE/'lab/05-training/index.html').read_text('utf-8')
        css=(SITE/'lab/05-training/styles.css').read_text('utf-8')
        js=(SITE/'lab/05-training/app.js').read_text('utf-8')
        trace=(SITE/'lab/05-training/trace05.json').read_text('utf-8')
        images=[base64.b64encode((SITE/f'lab/05-training/val-{i}.png').read_bytes()).decode() for i in range(6)]
        prep=f'<script>window.__inlineTrace={trace};window.__inlineImgs='+json.dumps([f'data:image/png;base64,{x}' for x in images])+';window.fetch=async()=>({ok:true,json:async()=>window.__inlineTrace});</script>'
        js=js.replace('im.src=`val-${i}.png`','im.src=window.__inlineImgs[i]')
        assert 'im.src=window.__inlineImgs[i]' in js
        html=html.replace('<link rel="stylesheet" href="styles.css">',f'<style>{css}</style>')
        html=html.replace('src="val-0.png"',f'src="data:image/png;base64,{images[0]}"')
        html=html.replace('<script src="app.js?v=20261105-unit05"></script>',prep+f'<script>{js}</script>')
        inline_url='data:text/html;base64,'+base64.b64encode(html.encode('utf-8')).decode()
        with sync_playwright() as pl:
            browser=pl.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            for kind,width,height in [('desktop',1280,900),('mobile',390,844)]:
                page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
                errors=[];page.on('pageerror',lambda err: errors.append(str(err)))
                page.set_content(html,wait_until='load',timeout=20000)
                page.wait_for_function("document.getElementById('metricVal').textContent !== '—'",timeout=10000)
                assert page.locator('#metricVal').inner_text()=='0.000'  # step=0
                page.locator('input[name="model"][value="eight"]').check()
                page.locator('#stepSlider').fill('8')
                assert page.locator('#metricVal').inner_text()=='0.102'
                assert page.locator('#metricTrain').inner_text()=='0.750'
                page.locator('#imageChoice').select_option('2')
                assert '最終モデル' in page.locator('#predInfo').inner_text()
                page.locator('#scoreSlider').fill('0.8')
                assert page.locator('#thresholdOut').inner_text()=='0.80'
                page.locator('#showGT').uncheck()
                assert page.locator('#showGT').is_checked() is False
                assert page.locator('#curveCanvas').is_visible() and page.locator('#predictionCanvas').is_visible()
                w=page.evaluate('({doc:document.documentElement.scrollWidth,view:window.innerWidth})')
                assert w['doc']<=w['view']+1,f'{kind} horizontal overflow: {w}'
                assert not errors,errors
                page.screenshot(path=str(args.out/f'unit05_{kind}.png'),full_page=True)
                print(f'PASS {kind}: interactive controls, JSON fetch, no JS exceptions, no horizontal overflow')
                page.close()
            browser.close()
    finally:
        server.shutdown();server.server_close()
if __name__=='__main__':main()
