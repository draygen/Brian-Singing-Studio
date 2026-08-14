# Brian-Singing-Studio

Zero-shot singing synthesis with Brian's real voice as the timbre reference.

**Phase 1 (this repo, current state):** isolated sung acapella from
SoulX-Singer (Soul-AILab), melody+lyrics from the official English example,
timbre from `references/IndexTTS_Brian_Speaking_Script.wav` (27.8 s dry mono
speech, 44.1 kHz). Later phases (presets, ComfyUI nodes, ACE-Step, Renoise)
are deliberately not built yet.

## Layout

- `engine/SoulX-Singer` — upstream checkout + `engine/.venv` (py3.10,
  torch 2.8.0+cu128; the repo's torch 2.2 pin predates Blackwell and cannot
  run on the RTX 5070). Both are gitignored.
- `references/` — Brian's real speaker reference (the ONLY timbre source).
- `scripts/setup.sh` — venv, deps, model downloads (SVS model + rmvpe/rosvot/
  parakeet EN preprocessing; no SVC model, no zh ASR, no vocal-sep models).
- `scripts/preprocess_brian.sh` — one-time annotation of Brian's wav
  (F0 + English transcription + note transcription) → SVS prompt metadata.
- `scripts/sing.sh` — the Phase 1 smoke test: Brian prompt × `en_target.json`
  official melody/lyric pair, `--auto_shift --fp16` → `outputs/brian_singing/`.
- `outputs/` — transcriptions and generated audio (gitignored).

## Run order

```bash
bash scripts/setup.sh
bash scripts/preprocess_brian.sh
bash scripts/sing.sh
```

Final deliverable is copied to `C:\sounds\brian_singing_acapella_test.wav`.

## Fallback (only if SVS fails on 12 GB)

SoulX-Singer-SVC (same repo, `model-svc.pt`, transcription-free audio-to-audio)
with an official example singing clip as the source performance and Brian's
wav as the prompt timbre. Not downloaded unless needed.
