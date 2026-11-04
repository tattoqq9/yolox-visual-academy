import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestSiteContract(unittest.TestCase):
    def test_unit03_files(self):
        for name in ['site/lab/03-block/index.html','site/lab/03-block/app.js','site/lab/03-block/styles.css','site/learn/03-block.html','site/code/unit03/student.py']:
            self.assertTrue((ROOT/name).is_file(),name)
    def test_default_selfie(self):
        txt=(ROOT/'site/lab/03-block/app.js').read_text()
        self.assertIn("imgFromUrl('../../assets/003Z-selfie.png','selfie')",txt)
    def test_sandbox_does_not_send_images(self):
        txt=(ROOT/'site/lab/03-block/app.js').read_text()
        for x in ['fetch(', 'XMLHttpRequest','sendBeacon(','localStorage']:
            self.assertNotIn(x,txt)
    def test_home_link(self):self.assertIn('lab/03-block/index.html',(ROOT/'site/index.html').read_text())
if __name__=='__main__': unittest.main()
