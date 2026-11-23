"""Writes test/fixtures/crop-reference.json: Python's sample_crop on a formula-defined image.

test/vision.test.ts rebuilds the same image from the same formula and checks the TypeScript
sampleCrop agrees, so training crops and pipeline crops cannot silently drift apart.
"""

from __future__ import annotations

import json

import numpy as np
import synth

SIZE = 96
CASES = [(48.0, 40.0, 50.0), (30.5, 61.25, 37.0), (70.0, 22.0, 64.0)]


def make_reference_image() -> np.ndarray:
    """Return a deterministic grey test pattern for the crop reference."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    pattern: np.ndarray = np.clip(
        128 + 90 * np.sin(xs * 0.31) * np.cos(ys * 0.17) + (xs * 3 + ys * 5) % 23, 0, 255
    )
    return pattern.astype(np.uint8)


def main() -> None:
    """Write reference crops that the TypeScript tests compare their crops against."""
    image = make_reference_image()
    cases = []
    for cx, cy, side in CASES:
        crop = synth.sample_crop(image, cx, cy, side)
        cases.append(
            {
                "cx": cx,
                "cy": cy,
                "side": side,
                "crop": [round(float(v), 5) for v in crop.reshape(-1)],
            }
        )
    out = synth.ROOT / "test" / "fixtures" / "crop-reference.json"
    out.write_text(json.dumps({"size": SIZE, "cases": cases}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
