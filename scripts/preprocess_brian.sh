#!/usr/bin/env bash
# Annotate Brian's real speech reference for use as the SVS timbre prompt:
# F0 (rmvpe) + English lyrics transcription (parakeet) + note transcription
# (rosvot). vocal_sep=False: the reference is already a clean dry recording.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/SoulX-Singer"
PY="$ROOT/engine/.venv/bin/python"
REF="$ROOT/references/IndexTTS_Brian_Speaking_Script.wav"

[[ -f "$REF" ]] || { echo "missing reference: $REF" >&2; exit 1; }

cd "$ENGINE"
export PYTHONPATH="$ENGINE:${PYTHONPATH:-}"

"$PY" -m preprocess.pipeline \
    --audio_path "$REF" \
    --save_dir "$ROOT/outputs/transcriptions/brian_prompt" \
    --language English \
    --device cuda \
    --vocal_sep False \
    --max_merge_duration 30000 \
    --midi_transcribe True

echo PREPROCESS_DONE
find "$ROOT/outputs/transcriptions/brian_prompt" -maxdepth 2 -type f | head -20
