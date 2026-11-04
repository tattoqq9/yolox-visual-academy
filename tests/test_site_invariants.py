"""Non-regression gate for the already-published Units 01 and 02.
Update hashes only in a reviewed commit after explicitly verifying changes.
"""
from pathlib import Path
import hashlib, json, unittest, zipfile

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
FROZEN={
    'site/lab/01-cnn/index.html':'3a818411e6b4a93a52975779ea558d8f0a158b80',
    'site/lab/01-cnn/app.js':'f769f6529d41f417003a9021ceee23fc291c4d66',
    'site/lab/02-grid/index.html':'4b8c82420be7e5cd4dbbc78972450bbcdad33f51',
    'site/lab/02-grid/app.js':'313b13c126c18544576452e6895655ede2d44bc3',
    'site/assets/003Z-selfie.png':'3ca6d647c675f7aa0293ecc0f61e04c4b2c6ee69',
}

def git_blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()

class SiteInvariants(unittest.TestCase):
    def test_previous_units_byte_identical(self):
        for rel,sha in FROZEN.items():
            with self.subTest(rel=rel):
                self.assertEqual(git_blob_sha((ROOT/rel).read_bytes()),sha)

    def test_unit03_materials_and_zip(self):
        data=json.loads((SITE/'assets/curriculum.json').read_text())
        unit=next(u for u in data['units'] if u['id']=='03a-blocks')
        self.assertEqual(unit['status'],'preview-ready')
        for field in ('lab','textbook','starter'):
            self.assertTrue((SITE/unit[field]).is_file())
        with zipfile.ZipFile(SITE/'downloads/unit03a_numpy_starter.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertIn('student.py',z.namelist())
            self.assertNotIn('reference_numpy.py',z.namelist())

    def test_no_teacher_solutions_or_remote_js_in_public_unit03(self):
        self.assertFalse((SITE/'code/unit03a/reference').exists())
        html=(SITE/'lab/03-blocks/index.html').read_text()
        self.assertNotIn('src="https://',html)
        self.assertIn('教育用',html)

if __name__=='__main__': unittest.main()
