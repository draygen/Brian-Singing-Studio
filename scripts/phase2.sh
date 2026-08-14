#!/usr/bin/env bash
# Phase 2 driver: excerpts -> their annotations -> custom scores -> battery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/SoulX-Singer"
PY="$ROOT/engine/.venv/bin/python"
REF="$ROOT/references/IndexTTS_Brian_Speaking_Script.wav"

# 1. Reference excerpts (first ~12 s, last ~12 s)
[[ -f "$ROOT/references/brian_early.wav" ]] || ffmpeg -y -loglevel error -i "$REF" -t 12 "$ROOT/references/brian_early.wav"
[[ -f "$ROOT/references/brian_late.wav"  ]] || ffmpeg -y -loglevel error -i "$REF" -ss 15.5 "$ROOT/references/brian_late.wav"

# 2. Annotate excerpts (same pipeline as the full reference)
cd "$ENGINE"; export PYTHONPATH="$ENGINE:${PYTHONPATH:-}"
for name in early late; do
  if [[ ! -f "$ROOT/outputs/transcriptions/brian_$name/metadata.json" ]]; then
    "$PY" -m preprocess.pipeline \
      --audio_path "$ROOT/references/brian_$name.wav" \
      --save_dir "$ROOT/outputs/transcriptions/brian_$name" \
      --language English --device cuda --vocal_sep False \
      --max_merge_duration 30000 --midi_transcribe True
  fi
done

# 3. Custom scores (sustain / rock / soft)
mkdir -p "$ROOT/outputs/scores"
"$PY" "$ROOT/scripts/make_score.py" "$ROOT/outputs/scores"

# 4. Battery
"$PY" "$ROOT/scripts/phase2_battery.py"
