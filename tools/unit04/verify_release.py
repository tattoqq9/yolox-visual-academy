#!/usr/bin/env python3
"""Release package integrity. Prints objective file count and source trace hash."""
import json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main(path):
 with zipfile.ZipFile(path) as z:
  bad=z.testzip();assert bad is None,bad
  names=set(z.namelist())
  for p in ['site/index.html','site/lab/04-detection/trace04.json','site/lab/04-detection/math.js','site/lab/04-detection/app.js','site/learn/04-detection.html','site/downloads/unit04_python_starter.zip','tests/test_unit04_contract.py','tests/unit04_math.test.cjs','.github/workflows/deploy-pages.yml']:
   assert p in names,p
  p=json.loads(z.read('site/lab/04-detection/trace04.json'))
  assert p['provenance']['learned'] is False
  assert b'UNTRAINED' in z.read('site/lab/04-detection/index.html') or b'\xe6\x9c\xaa\xe5\xad\xa6\xe7\xbf\x92' in z.read('site/lab/04-detection/index.html')
  print(f'PASS ZIP CRC: {len(names)} files / {path.stat().st_size:,} bytes / source_sha256={p["provenance"]["source_sha256"]}')
if __name__=='__main__':main(Path(sys.argv[1]))
