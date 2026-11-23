"""Writes a deterministic set of labelled synthetic puzzle photos: NNNN.jpg + NNNN.json per image.

The `train` split uses fonts/train, the `eval` split only fonts/eval, so eval digits are in typefaces
the classifier has never seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import synth


def main() -> None:
    """Generate a labelled photo set from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unsolvable-fraction", type=float, default=0.08)
    args = parser.parse_args()

    fonts = synth.EVAL_FONTS if args.split == "eval" else synth.TRAIN_FONTS
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.count):
        size = 3 if index % 2 == 0 else 4
        solvable = bool(rng.random() >= args.unsolvable_fraction)
        state = synth.generate_random_state(rng, size, solvable)
        landscape = bool(rng.random() < 0.5)
        width, height = (960, 720) if landscape else (720, 960)
        image, quad, font = synth.render_photo(rng, size, state, fonts, width, height)
        stem = f"{index:04d}"
        image.save(args.out / f"{stem}.jpg", quality=95)  # already degraded; keep it as is
        label = {
            "size": size,
            "state": state,
            "solvable": solvable,
            "corners": quad.round(1).tolist(),
            "font": font,
        }
        (args.out / f"{stem}.json").write_text(json.dumps(label))
    print(f"wrote {args.count} images to {args.out}")


if __name__ == "__main__":
    main()
