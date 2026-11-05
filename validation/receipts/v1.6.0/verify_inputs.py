import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
for row in json.loads((ROOT/'validation/work/release-1.6.0-gpu/prepared-inputs-sha256.json').read_text()):
 p=ROOT/row['destination'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],str(p)
print('all prepared input SHA-256 values verified')
