#!/bin/bash
# usage: bash copy_back.sh <destination dir in the repo>
# Copies each run's config, per-iteration log (gzipped), final design, console log and
# the analysis artefacts. Intermediate design_it*.npz snapshots stay in /tmp: they are
# ~3 MB per run and nothing in the report reads them.
set -e
SRC=/tmp/claude-1001/d210ea4f/runs
DEST=$1
mkdir -p "$DEST"
for d in "$SRC"/output/*/; do
  run=$(basename "$d")
  mkdir -p "$DEST/$run"
  cp "$d/config.json" "$DEST/$run/" 2>/dev/null || true
  cp "$d/final_design.npz" "$DEST/$run/" 2>/dev/null || true
  gzip -c "$d/iterations.jsonl" > "$DEST/$run/iterations.jsonl.gz"
  cp "$SRC/logs/$run.log" "$DEST/$run/console.log" 2>/dev/null || true
done
mkdir -p "$DEST/analysis"
cp "$SRC"/analysis/*.png "$SRC"/analysis/summary.tsv "$DEST/analysis/" 2>/dev/null || true
cp "$SRC"/logs/progress_*.txt "$DEST/analysis/" 2>/dev/null || true
du -sh "$DEST"
