"""Print one line whenever a sweep run gains iterations slower than 5 s. Polls every 120 s."""

import json
import time
from pathlib import Path

OUT = Path("/tmp/claude-1001/d210ea4f/runs/output")
reported: dict[str, int] = {}
while True:
    for f in sorted(OUT.glob("*/iterations.jsonl")):
        run = f.parent.name
        try:
            L = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        except OSError, json.JSONDecodeError:
            continue
        el = [e["elapsed"] for e in L]
        slow = [
            L[i + 1]["loop"] for i, (a, b) in enumerate(zip(el, el[1:])) if b - a > 5
        ]
        if len(slow) > reported.get(run, 0):
            print(
                f"SLOW {run}: {len(slow)} iterations >5s, latest at loop {slow[-1]} (now at loop {L[-1]['loop']})",
                flush=True,
            )
            reported[run] = len(slow)
    time.sleep(120)
