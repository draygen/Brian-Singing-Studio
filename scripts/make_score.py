"""Author a SoulX-Singer score-control metadata JSON from a simple note list.

A note is (word, duration_s, midi_pitch, note_type):
  note_type 1 = rest (word must be "<SP>", pitch 0)
  note_type 2 = sung note starting a word
  note_type 3 = melisma: same word continues on a new pitch (repeat the word)

Phonemes come from g2p_en with the repo's en_ prefix + hyphen joining, matching
example/audio/en_target.json exactly. This is the seed of the later
Renoise-MIDI → score path (Phase 5).
"""

import json
import re
import sys


def phonemize(word):
    from g2p_en import G2p

    if not hasattr(phonemize, "_g2p"):
        phonemize._g2p = G2p()
    phones = [p for p in phonemize._g2p(word) if re.match(r"^[A-Z]", p)]
    return "en_" + "-".join(phones)


def build_segment(notes, language="English", index="custom_0"):
    text, phoneme, pitches, types, durs = [], [], [], [], []
    for word, dur, pitch, ntype in notes:
        durs.append("%.2f" % dur)
        pitches.append(str(int(pitch)))
        types.append(str(int(ntype)))
        if ntype == 1:
            text.append("<SP>")
            phoneme.append("<SP>")
        else:
            text.append(word)
            phoneme.append(phonemize(word))
    total_ms = int(round(sum(float(d) for d in durs) * 1000))
    return {
        "index": index,
        "language": language,
        "time": [0, total_ms],
        "duration": " ".join(durs),
        "text": " ".join(text),
        "phoneme": " ".join(phoneme),
        "note_pitch": " ".join(pitches),
        "note_type": " ".join(types),
    }


# --- Phase 2 scores -------------------------------------------------------

# Sustained-vowel / vibrato test: long open-vowel notes with melismas.
SUSTAIN = [
    ("<SP>", 0.40, 0, 1),
    ("I", 0.45, 57, 2),
    ("am", 0.45, 59, 2),
    ("singing", 0.50, 60, 2),
    ("singing", 0.50, 62, 3),
    ("all", 0.70, 64, 2),
    ("alone", 0.50, 62, 2),
    ("alone", 1.90, 64, 3),
    ("tonight", 0.50, 62, 2),
    ("tonight", 2.30, 59, 3),
    ("<SP>", 0.40, 0, 1),
    ("hold", 1.10, 60, 2),
    ("me", 0.80, 62, 2),
    ("now", 1.20, 64, 2),
    ("now", 1.60, 62, 3),
    ("<SP>", 0.50, 0, 1),
]

# Rock-leaning phrase: short punchy notes, upper register, driving rhythm.
ROCK = [
    ("<SP>", 0.30, 0, 1),
    ("We", 0.28, 62, 2),
    ("are", 0.28, 64, 2),
    ("running", 0.30, 66, 2),
    ("running", 0.30, 66, 3),
    ("through", 0.28, 64, 2),
    ("the", 0.28, 62, 2),
    ("fire", 0.45, 66, 2),
    ("fire", 0.65, 67, 3),
    ("<SP>", 0.25, 0, 1),
    ("never", 0.30, 66, 2),
    ("never", 0.30, 64, 3),
    ("gonna", 0.30, 62, 2),
    ("gonna", 0.30, 62, 3),
    ("stop", 0.50, 64, 2),
    ("tonight", 0.40, 62, 2),
    ("tonight", 1.30, 59, 3),
    ("<SP>", 0.40, 0, 1),
]

# Soft ballad phrase: gentle low-register line, moderate sustains.
SOFT = [
    ("<SP>", 0.40, 0, 1),
    ("Close", 0.60, 55, 2),
    ("your", 0.45, 57, 2),
    ("eyes", 0.90, 59, 2),
    ("eyes", 0.70, 57, 3),
    ("and", 0.40, 55, 2),
    ("drift", 0.60, 57, 2),
    ("away", 0.45, 59, 2),
    ("away", 1.40, 55, 3),
    ("<SP>", 0.35, 0, 1),
    ("I", 0.50, 54, 2),
    ("will", 0.50, 55, 2),
    ("stay", 0.80, 57, 2),
    ("stay", 1.50, 55, 3),
    ("<SP>", 0.50, 0, 1),
]

SCORES = {"sustain": SUSTAIN, "rock": ROCK, "soft": SOFT}


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, notes in SCORES.items():
        seg = build_segment(notes, index="brian_%s_0" % name)
        path = "%s/brian_%s.json" % (out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([seg], fh, ensure_ascii=False, indent=1)
        print(path, "dur=%.1fs" % (seg["time"][1] / 1000.0))
