"""Desktop and mobile browser smoke using exactly the site's HTTP-hosted authored assets.

Run: python3 tools/unit00/smoke_gui.py
Requires playwright Python and locally installed Chromium; never uploads images.
"""
import base64
import json
import http.server
from pathlib import Path
import socketserver
import threading
import urllib.request
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/unit00/gui'
OUT.mkdir(parents=True, exist_ok=True)

class ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT/'site'), **kwargs)
    def log_message(self, *args): pass


def data_url(path):
    mime = 'image/png' if path.suffix == '.png' else 'image/jpeg'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}'


def inline_lab(page, lab, is_unit00):
    """Browser environment blocks direct localhost. Preserve exact JS/CSS/JSON, inline assets only."""
    root = ROOT/'site'
    html_file = root/'lab'/lab/'index.html'
    html = html_file.read_text('utf8')
    css = (html_file.parent/'styles.css').read_text('utf8')
    js = (html_file.parent/'app.js').read_text('utf8')
    image_root = root/'assets/003Z-selfie.png'
    sources = {'../../assets/003Z-selfie.png': image_root}
    if not is_unit00:
        sources.update({f'assets/{name}.jpg': html_file.parent/'assets'/f'{name}.jpg'
                        for name in ('person','room','park')})
    # Both the HTML and existing source JS are unchanged apart from replacing
    # relative asset URLs with their original exact bytes for the inline harness.
    for url, actual in sources.items():
        encoded = data_url(actual)
        html = html.replace(url, encoded)
        js = js.replace(url, encoded)
    html = html.replace('<link rel="stylesheet" href="styles.css">', '<style>'+css+'</style>')
    html = html.replace('<script defer src="app.js"></script>', '')
    html = html.replace('<script src="app.js?v=20261105-unit00-bridge"></script>', '')
    page.goto('about:blank')
    page.set_content(html, wait_until='load')
    if is_unit00:
        payload = json.loads((html_file.parent/'actual-untrained-trace.json').read_text('utf8'))
        # This mock only resolves the static public trace read. The data is exact.
        page.evaluate('(data) => {window.fetch = async () => ({ok:true, json:async()=> data});}', payload)
    page.add_script_tag(content=js)


def exercise(page, mobile=False):
    js_errors=[]
    page.on('pageerror',lambda e:js_errors.append(str(e)))
    inline_lab(page, '00-overview', True)
    page.locator('h1').wait_for()
    assert 'AIは、どうやって' in page.locator('h1').inner_text()
    assert page.evaluate('YVAUnit00.getState().stage') == 0
    for n in range(6):
        page.locator(f'[data-stage="{n}"]').click()
        assert page.evaluate('YVAUnit00.getState().stage') == n
        assert page.locator(f'[data-stage="{n}"]').get_attribute('aria-current') == 'step'
    assert page.locator('#illustrativeBox').is_visible()
    page.locator('[data-stage="0"]').click()
    assert not page.locator('#illustrativeBox').is_visible()
    page.locator('#playFlow').click()
    assert page.locator('#playFlow').get_attribute('aria-pressed')=='true'
    page.locator('#playFlow').click()
    assert page.locator('#playFlow').get_attribute('aria-pressed')=='false'
    page.locator('#loadTrace').click()
    page.wait_for_function('YVAUnit00.getState().trained === false',timeout=10000)
    assert page.locator('#tracePlaceholder').is_hidden()
    assert 'ch 0' in page.locator('#traceShape').inner_text()
    page.locator('#traceStage').select_option('p3')
    assert '52×52' in page.locator('#traceShape').inner_text()
    page.locator('#traceCanvas').click(position={'x':70,'y':50})
    page.screenshot(path=str(OUT/('unit00_mobile.png' if mobile else 'unit00_desktop.png')),full_page=True)
    overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 1')
    assert not overflow, 'Unit00 horizontal overflow'
    assert not js_errors,js_errors
    # Unit 01 via actual page and source image (browser must load JS and sample)
    inline_lab(page, '01-cnn', False)
    page.wait_for_function('YOLOXLab.inputGray() != null',timeout=15000)
    assert page.locator('.unit01-orientation').is_visible()
    assert 'グレー値' in page.locator('#lessonContent').inner_text()
    assert page.locator('#originalSample').is_visible()
    assert page.locator('#sampleCanvas').is_visible()
    assert page.locator('#zoomSample').is_visible()
    page.locator('#photoCanvas').click(position={'x':160,'y':130})
    assert page.locator('#zoomSample').is_visible()
    page.screenshot(path=str(OUT/('unit01_input_mobile.png' if mobile else 'unit01_input_desktop.png')),full_page=True)
    page.locator('[data-step="1"]').click()
    assert 'Kernel' in page.locator('#lessonContent').inner_text()
    assert 'グレー値1チャネル' in page.locator('#lessonContent').inner_text()
    page.locator('[data-action="rgb"]').click()
    assert '27項' in page.locator('#lessonContent').inner_text()
    page.screenshot(path=str(OUT/('unit01_mobile.png' if mobile else 'unit01_desktop.png')),full_page=True)
    assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 1'), 'Unit01 horizontal overflow'
    assert not js_errors,js_errors
    return {'device':'mobile' if mobile else 'desktop','js_errors':js_errors,'horizontal_overflow':overflow}


def main():
    with ReuseServer(('127.0.0.1',0),Handler) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f'http://127.0.0.1:{server.server_address[1]}'
        for path in ('/lab/00-overview/index.html','/lab/00-overview/app.js',
                     '/lab/00-overview/actual-untrained-trace.json','/lab/01-cnn/index.html'):
            with urllib.request.urlopen(base+path,timeout=5) as response:
                assert response.status == 200 and response.read(), path
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
            for size,mobile in [({'width':1400,'height':920},False),({'width':390,'height':844},True)]:
                context=browser.new_context(viewport=size,device_scale_factor=1,is_mobile=mobile,has_touch=mobile)
                page=context.new_page()
                page.server_base=base
                print(exercise(page,mobile))
                context.close()
            browser.close()
        server.shutdown()
    print('PASS: Unit00 and Unit01 local HTTP assets and browser inline-asset interactions')

if __name__=='__main__':main()
