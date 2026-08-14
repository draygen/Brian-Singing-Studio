#!/usr/bin/env bash
# Phase 1 smoke test: Brian's timbre singing the official English example song.
#
#   bash scripts/sing.sh [prompt_metadata.json] [prompt.wav]
#
# Defaults to the preprocessed Brian metadata; target melody/lyrics is the
# repo's en_target.json (official example pair). --auto_shift transposes the
# score into the prompt's register.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/SoulX-Singer"
PY="$ROOT/engine/.venv/bin/python"

PROMPT_META="${1:-$ROOT/outputs/transcriptions/brian_prompt/metadata.json}"
PROMPT_WAV="${2:-$ROOT/references/IndexTTS_Brian_Speaking_Script.wav}"
SAVE_DIR="$ROOT/outputs/brian_singing"

cd "$ENGINE"
export PYTHONPATH="$ENGINE:${PYTHONPATH:-}"

"$PY" -m cli.inference \
    --device cuda \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path "$PROMPT_WAV" \
    --prompt_metadata_path "$PROMPT_META" \
    --target_metadata_path example/audio/en_target.json \
    --phoneset_path soulxsinger/utils/phoneme/phone_set.json \
    --save_dir "$SAVE_DIR" \
    --auto_shift \
    --pitch_shift 0 \
    --fp16

echo SING_DONE
ls -la "$SAVE_DIR"
