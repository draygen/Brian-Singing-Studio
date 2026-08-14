"""Phase 2b: Brian's requested grid, all at pitch_shift = -7.

  seeds x CFG : {42, 7, 777, 1234} x {1.5, 3.0, 5.0}   (32 steps)
  steps check : the CFG 3.0 row repeated at 16 steps

Reuses the loaded-once model runner from phase2_battery.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_battery import (  # noqa: E402  (module import performs engine chdir)
    EN_TARGET, META_FULL, OUT, REF_FULL, DataProcessor, build_model,
    load_config, run_one,
)

OUT2 = os.path.join(os.path.dirname(OUT), "phase2b")

SEEDS = [42, 7, 777, 1234]
CFGS = [1.5, 3.0, 5.0]


def main():
    os.makedirs(OUT2, exist_ok=True)
    # run_one writes into OUT; point it at our folder instead
    import phase2_battery
    phase2_battery.OUT = OUT2

    config = load_config("soulxsinger/config/soulxsinger.yaml")
    model = build_model("pretrained_models/SoulX-Singer/model.pt", config,
                        "cuda", use_fp16=True)
    dp = DataProcessor(hop_size=480, sample_rate=24000,
                       phoneset_path="soulxsinger/utils/phoneme/phone_set.json",
                       device="cuda")

    results = []
    for seed in SEEDS:
        for cfg in CFGS:
            name = "seed%d_cfg%s" % (seed, ("%g" % cfg).replace(".", "p"))
            r = run_one(model, dp, name, META_FULL, REF_FULL, EN_TARGET,
                        seed, cfg, 32, -7)
            print("%-22s %5.1fs rtf=%.2f peak=%.2f" %
                  (r["name"], r["wall_s"], r["rtf"], r["peak"]))
            results.append(r)
    for seed in SEEDS:
        name = "seed%d_cfg3_steps16" % seed
        r = run_one(model, dp, name, META_FULL, REF_FULL, EN_TARGET,
                    seed, 3.0, 16, -7)
        print("%-22s %5.1fs rtf=%.2f peak=%.2f" %
              (r["name"], r["wall_s"], r["rtf"], r["peak"]))
        results.append(r)

    with open(os.path.join(OUT2, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("GRID_DONE")


if __name__ == "__main__":
    main()
