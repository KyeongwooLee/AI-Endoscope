"""Read-only status; does not inspect held-out outcomes."""
import json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parents[2]/'runs/20261012_followup';manifest=json.loads((p/'manifest.json').read_text());counts=Counter();total=Counter()
for r in manifest:
 total[r['split']]+=1
 if (p/'records'/f"{r['id']}.json").exists():counts[r['split']]+=1
print(datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'),dict(counts),'of',dict(total));print('Models frozen:',(p/'selection_frozen.json').exists(),'Final evaluated:',(p/'test_results.json').exists())
