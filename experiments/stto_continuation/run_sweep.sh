#!/bin/bash
# usage: run_sweep.sh <snapshot dir> <stream name> <config.json>...
# Runs each config in order from the frozen snapshot; records failures and continues.
SNAP=$1; STREAM=$2; shift 2
ROOT=/tmp/claude-1001/d210ea4f/runs
mkdir -p "$ROOT/logs"
CONFIGS=()
for cfg in "$@"; do CONFIGS+=("$(realpath "$cfg")"); done
cd "$ROOT" || exit 1
for cfg in "${CONFIGS[@]}"; do
  tag=$(basename "$cfg" .json)
  if [ -f "output/$tag/final_design.npz" ]; then
    echo "$(date -Iseconds) SKIP $tag" >> "logs/progress_$STREAM.txt"; continue
  fi
  echo "$(date -Iseconds) START $tag" >> "logs/progress_$STREAM.txt"
  PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$SNAP" \
    python3 -m sttopt.stto_cli --config "$cfg" --tag "$tag" --tag-force --snapshot-every 25 \
    > "logs/$tag.log" 2>&1
  rc=$?
  echo "$(date -Iseconds) END $tag exit=$rc" >> "logs/progress_$STREAM.txt"
done
echo "$(date -Iseconds) STREAMDONE" >> "logs/progress_$STREAM.txt"
