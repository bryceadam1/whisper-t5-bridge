"""Print TTS progress on the VM: clips per split, audio hours, unfinished clips, last log lines."""

import collections
import json
from pathlib import Path

manifest = Path("data/audio/manifest.jsonl")
rows = [json.loads(line) for line in manifest.open()] if manifest.exists() else []
counts = collections.Counter(r["split"] for r in rows)
hours = sum(r["duration"] for r in rows) / 3600
unfinished = sum(not r["finished"] for r in rows)
print(dict(counts), "%.2f h audio, %d unfinished" % (hours, unfinished))
for log in ("logs/tts.log", "logs/teacher.log", "logs/features.log", "logs/train.log"):
    p = Path(log)
    if p.exists() and p.stat().st_size:
        print(log, "|", p.read_text().strip().splitlines()[-1])
