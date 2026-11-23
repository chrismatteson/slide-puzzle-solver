import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import synth

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_is_solvable_follows_the_parity_rule() -> None:
    assert synth.is_solvable([1, 2, 3, 4, 5, 6, 7, 8, 0], 3)
    assert not synth.is_solvable([2, 1, 3, 4, 5, 6, 7, 8, 0], 3)
    solved_4 = [*range(1, 16), 0]
    assert synth.is_solvable(solved_4, 4)
    # On an even width the blank's row counts: one slide up from solved stays solvable.
    up_one = solved_4[:11] + [0] + solved_4[12:15] + [12]
    assert synth.is_solvable(up_one, 4)
    swapped = solved_4.copy()
    swapped[13], swapped[14] = swapped[14], swapped[13]
    assert not synth.is_solvable(swapped, 4)


@pytest.mark.parametrize("size", [3, 4])
@pytest.mark.parametrize("solvable", [True, False])
def test_generate_random_state_gives_a_permutation_of_the_requested_parity(
    size: int, solvable: bool
) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        state = synth.generate_random_state(rng, size, solvable)
        assert sorted(state) == list(range(size * size))
        assert synth.is_solvable(state, size) == solvable


def test_normalise_crop_has_zero_mean_and_keeps_blank_cells_quiet() -> None:
    rng = np.random.default_rng(2)
    crop = rng.uniform(0, 1, (synth.CROP_PIXELS, synth.CROP_PIXELS)).astype(np.float32)
    normalised = synth.normalise_crop(crop)
    assert normalised.dtype == np.float32
    assert abs(float(normalised.mean())) < 1e-5
    assert abs(float(normalised.std()) - 1) < 1e-4
    flat = np.full((synth.CROP_PIXELS, synth.CROP_PIXELS), 0.5, np.float32)
    flat[0, 0] = 0.51
    assert float(np.abs(synth.normalise_crop(flat)).max()) < 0.2


def test_eval_and_training_fonts_are_disjoint() -> None:
    assert synth.EVAL_FONTS and synth.TRAIN_FONTS
    assert not synth.EVAL_FONT_NAMES & {font.stem for font in synth.TRAIN_FONTS}


def write_crop_set(path: Path, fonts: list[str], source: str) -> np.lib.npyio.NpzFile:
    np.savez(
        path, x=np.zeros((1, 4)), y=np.zeros(1), fonts=np.array(fonts), source=np.array(source)
    )
    crops: np.lib.npyio.NpzFile = np.load(path)
    return crops


def test_check_training_crops_refuses_eval_fonts_and_accepts_training_fonts(
    tmp_path: Path,
) -> None:
    eval_font = sorted(synth.EVAL_FONT_NAMES)[0]
    with pytest.raises(SystemExit, match="eval"):
        synth.check_training_crops(
            tmp_path / "a.npz", write_crop_set(tmp_path / "a.npz", [eval_font], "hard cases")
        )
    with pytest.raises(SystemExit, match="eval"):
        train_font = synth.TRAIN_FONTS[0].stem
        synth.check_training_crops(
            tmp_path / "b.npz", write_crop_set(tmp_path / "b.npz", [train_font], "eval photos")
        )
    with pytest.raises(SystemExit, match="not in fonts/train"):
        synth.check_training_crops(
            tmp_path / "c.npz", write_crop_set(tmp_path / "c.npz", ["NoSuchFont"], "hard cases")
        )
    good = write_crop_set(tmp_path / "d.npz", [synth.TRAIN_FONTS[0].stem], "hard cases")
    synth.check_training_crops(tmp_path / "d.npz", good)


def generate(out: Path, seed: int) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / "gen_photos.py"), "--split", "eval", "--count", "4"]
        + ["--seed", str(seed), "--out", str(out)],
        check=True,
        capture_output=True,
    )


def test_gen_photos_is_deterministic_and_labels_match_the_photos(tmp_path: Path) -> None:
    generate(tmp_path / "a", 11)
    generate(tmp_path / "b", 11)
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == [f"{i:04d}.{ext}" for i in range(4) for ext in ("jpg", "json")]
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    for index in range(4):
        label = json.loads((tmp_path / "a" / f"{index:04d}.json").read_text())
        size = label["size"]
        assert size == (3 if index % 2 == 0 else 4)
        assert sorted(label["state"]) == list(range(size * size))
        assert synth.is_solvable(label["state"], size) == label["solvable"]
        assert label["font"] in synth.EVAL_FONT_NAMES
        corners = np.asarray(label["corners"])
        assert corners.shape == (4, 2)
        assert synth.is_convex(corners)
