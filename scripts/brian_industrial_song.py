"""Industrial/heavy song built around Brian's chosen acapella (C_pitch_-7).

Reverse Phase-4 workflow: Brian acapella -> arranged vocal skeleton ->
ACE-Step audio2audio re-imagines a full industrial track around it, keeping
the melodic/vocal contour, with matching lyrics in the prompt.

The 6.9 s phrase is arranged into a ~41 s skeleton (intro gap, phrase x2,
breakdown gap, phrase x2, outro) so the a2a reference has song-like structure.
Three variants: two a2a strengths + one alternate seed.
"""

import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACAPELLA = os.path.join(ROOT, "outputs", "phase2", "C_pitch_-7.wav")
OUT = os.path.join(ROOT, "outputs", "acestep")
SKELETON = os.path.join(OUT, "brian_vocal_skeleton.wav")

PROMPT = (
    "industrial metal, heavy distorted guitars, pounding mechanical drums, "
    "dark electronic synthesizers, aggressive, melodic chorus, soaring male "
    "vocals, anthemic, clean modern production, polished mix, 110 bpm"
)

LYRICS = """[verse]
Who says you're not pretty
Who says you're not beautiful

[chorus]
Who says you're not pretty
Who says you're not beautiful
Who says
"""


def build_skeleton():
    voc, sr = sf.read(ACAPELLA)
    voc = voc.astype(np.float32)
    gap = lambda s: np.zeros(int(s * sr), dtype=np.float32)  # noqa: E731
    arrangement = np.concatenate([
        gap(2.0), voc, gap(1.2), voc,           # verse-ish
        gap(3.0),                                # breakdown
        voc, gap(1.2), voc, gap(2.0),            # chorus-ish + outro
    ])
    os.makedirs(OUT, exist_ok=True)
    sf.write(SKELETON, arrangement, sr)
    return len(arrangement) / sr


def main():
    duration = build_skeleton()
    print("skeleton %.1fs" % duration)

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(dtype="bfloat16", cpu_offload=True,
                           overlapped_decode=True)

    variants = [
        ("brian_industrial_A_s0p55", 0.55, 42),
        ("brian_industrial_B_s0p70", 0.70, 42),
        ("brian_industrial_C_s0p55_seed7", 0.55, 7),
    ]
    for name, strength, seed in variants:
        save = os.path.join(OUT, name + ".wav")
        pipe(
            format="wav",
            audio_duration=duration,
            prompt=PROMPT,
            lyrics=LYRICS,
            infer_step=60,
            guidance_scale=15.0,
            manual_seeds=[seed],
            audio2audio_enable=True,
            ref_audio_strength=strength,
            ref_audio_input=SKELETON,
            save_path=save,
        )
        print("WROTE", save)
    print("SONG_DONE")


if __name__ == "__main__":
    sys.exit(main())
