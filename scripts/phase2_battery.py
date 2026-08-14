"""Phase 2 battery: load SoulX-Singer once, sweep every useful knob.

Axes covered (see MANIFEST):
  - reference excerpt (full / early / late)
  - seed variation
  - register (explicit pitch_shift vs auto)
  - cfg (speaker similarity vs expressiveness)
  - n_steps (quality vs speed)
  - sustained vowels / vibrato (custom score)
  - style preset candidates (Neutral / Soft / Rock / Emotional)

Writes outputs/phase2/<name>.wav + outputs/phase2/results.json.
Run via scripts/phase2.sh (sets PYTHONPATH and cwd).
"""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE = os.path.join(ROOT, "engine", "SoulX-Singer")
OUT = os.path.join(ROOT, "outputs", "phase2")
SCORES = os.path.join(ROOT, "outputs", "scores")

sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

from cli.inference import build_model  # noqa: E402
from soulxsinger.utils.data_processor import DataProcessor  # noqa: E402
from soulxsinger.utils.file_utils import load_config  # noqa: E402

REF_FULL = os.path.join(ROOT, "references", "IndexTTS_Brian_Speaking_Script.wav")
REF_EARLY = os.path.join(ROOT, "references", "brian_early.wav")
REF_LATE = os.path.join(ROOT, "references", "brian_late.wav")
META_FULL = os.path.join(ROOT, "outputs", "transcriptions", "brian_prompt", "metadata.json")
META_EARLY = os.path.join(ROOT, "outputs", "transcriptions", "brian_early", "metadata.json")
META_LATE = os.path.join(ROOT, "outputs", "transcriptions", "brian_late", "metadata.json")
EN_TARGET = os.path.join(ENGINE, "example", "audio", "en_target.json")
SUSTAIN = os.path.join(SCORES, "brian_sustain.json")
ROCK = os.path.join(SCORES, "brian_rock.json")
SOFT = os.path.join(SCORES, "brian_soft.json")

# name, prompt_meta, prompt_wav, target, seed, cfg, n_steps, pitch_shift(None=auto)
MANIFEST = [
    # A. reference excerpts
    ("A_ref_full",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 32, None),
    ("A_ref_early",       META_EARLY, REF_EARLY, EN_TARGET, 42,   3.0, 32, None),
    ("A_ref_late",        META_LATE,  REF_LATE,  EN_TARGET, 42,   3.0, 32, None),
    # B. seeds (full ref)
    ("B_seed_7",          META_FULL,  REF_FULL,  EN_TARGET, 7,    3.0, 32, None),
    ("B_seed_1234",       META_FULL,  REF_FULL,  EN_TARGET, 1234, 3.0, 32, None),
    ("B_seed_777",        META_FULL,  REF_FULL,  EN_TARGET, 777,  3.0, 32, None),
    # C. register (explicit shift disables auto)
    ("C_pitch_-7",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 32, -7),
    ("C_pitch_-4",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 32, -4),
    ("C_pitch_-2",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 32, -2),
    ("C_pitch_+2",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 32, 2),
    # D. cfg: lower = looser/more natural, higher = tighter/more score-faithful
    ("D_cfg_1.5",         META_FULL,  REF_FULL,  EN_TARGET, 42,   1.5, 32, None),
    ("D_cfg_5",           META_FULL,  REF_FULL,  EN_TARGET, 42,   5.0, 32, None),
    # E. steps
    ("E_steps_16",        META_FULL,  REF_FULL,  EN_TARGET, 42,   3.0, 16, None),
    # F. sustained vowels / vibrato
    ("F_sustain",         META_FULL,  REF_FULL,  SUSTAIN,   42,   3.0, 32, None),
    # G. style preset candidates
    ("Brian_Singing_Neutral",   META_FULL, REF_FULL, EN_TARGET, 42, 3.0, 32, None),
    ("Brian_Singing_Soft",      META_FULL, REF_FULL, SOFT,      42, 2.0, 32, None),
    ("Brian_Singing_Rock",      META_FULL, REF_FULL, ROCK,      42, 5.0, 32, None),
    ("Brian_Singing_Emotional", META_FULL, REF_FULL, SUSTAIN,   7,  4.0, 32, None),
]


def run_one(model, dp, name, prompt_meta_path, prompt_wav, target_path,
            seed, cfg, n_steps, pitch_shift):
    prompt_meta = json.load(open(prompt_meta_path, encoding="utf-8"))[0]
    target_list = json.load(open(target_path, encoding="utf-8"))
    prompt_data = dp.process(prompt_meta, prompt_wav)

    total_len = int(target_list[-1]["time"][1] / 1000 * 24000)
    merged = np.zeros(total_len, dtype=np.float32)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    for target_meta in target_list:
        target_data = dp.process(target_meta, None)
        with torch.no_grad():
            audio = model.infer(
                {"prompt": prompt_data, "target": target_data},
                auto_shift=pitch_shift is None,
                pitch_shift=pitch_shift or 0,
                n_steps=n_steps,
                cfg=cfg,
                control="score",
                use_fp16=True,
            )
        audio = audio.squeeze().cpu().numpy()
        start = int(target_meta["time"][0] / 1000 * 24000)
        n = min(len(audio), total_len - start)
        merged[start:start + n] = audio[:n]
    wall = time.time() - t0

    path = os.path.join(OUT, name + ".wav")
    sf.write(path, merged, 24000)
    dur = total_len / 24000.0
    peak = float(np.abs(merged).max())
    return {"name": name, "target": os.path.basename(target_path),
            "prompt": os.path.basename(prompt_wav), "seed": seed, "cfg": cfg,
            "n_steps": n_steps,
            "pitch_shift": "auto" if pitch_shift is None else pitch_shift,
            "wall_s": round(wall, 2), "dur_s": round(dur, 2),
            "rtf": round(wall / dur, 3), "peak": round(peak, 3)}


def main():
    os.makedirs(OUT, exist_ok=True)
    config = load_config("soulxsinger/config/soulxsinger.yaml")
    t0 = time.time()
    model = build_model("pretrained_models/SoulX-Singer/model.pt", config,
                        "cuda", use_fp16=True)
    load_s = time.time() - t0
    dp = DataProcessor(hop_size=480, sample_rate=24000,
                       phoneset_path="soulxsinger/utils/phoneme/phone_set.json",
                       device="cuda")

    results = {"model_load_s": round(load_s, 1), "runs": []}
    for row in MANIFEST:
        try:
            r = run_one(model, dp, *row)
            print("%-26s %5.1fs rtf=%.2f peak=%.2f" %
                  (r["name"], r["wall_s"], r["rtf"], r["peak"]))
        except Exception as exc:
            r = {"name": row[0], "error": "%s: %s" % (type(exc).__name__, exc)}
            print("%-26s FAILED %s" % (row[0], r["error"]))
        results["runs"].append(r)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("BATTERY_DONE", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
