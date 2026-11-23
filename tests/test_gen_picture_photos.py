import hashlib
import json
from pathlib import Path

import gen_picture_photos as gen
import numpy as np
import pytest
import synth
from PIL import Image, ImageFont


def make_picture() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [3, 4])
def test_scramble_gives_a_permutation_that_is_not_solved(size: int) -> None:
    rng = np.random.default_rng(1)
    for home in (0, size * size - 1, size + 1):
        cells = gen.scramble(rng, size, home)
        assert sorted(cells) == [-1, *[p for p in range(size * size) if p != home]]
        assert cells != [-1 if p == home else p for p in range(size * size)]


def test_label_matches_the_rendered_puzzle() -> None:
    rng = np.random.default_rng(2)
    photo, label = gen.render_picture_photo(rng, make_picture(), 3, 640, 480)
    assert photo.size == (640, 480)
    solved = label["solved"]
    assert isinstance(solved, list)
    assert sorted(solved) == [-1, *[c for c in range(9) if c != label["blank"]]]
    quad = label["tilesQuad"]
    assert isinstance(quad, list)
    assert len(quad) == 4
    json.dumps(label)  # plain JSON types only


def test_numbered_photos_are_unchanged_by_the_shared_photo_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The hash is the render by scripts/synth.py from before photograph_flat_board was split out of
    # render_photo (commit e5e623a); the old and current code both give it. Pillow lays text out with
    # raqm only where the system has libfribidi, which moves glyphs by a pixel, so the basic layout
    # engine is pinned here to make the hash the same in every environment.
    truetype = ImageFont.truetype

    def truetype_basic(font: str, size: int) -> ImageFont.FreeTypeFont:
        return truetype(font, size, layout_engine=ImageFont.Layout.BASIC)

    monkeypatch.setattr(ImageFont, "truetype", truetype_basic)
    rng = np.random.default_rng(3)
    state = synth.generate_random_state(rng, 3, True)
    image, quad, _ = synth.render_photo(rng, 3, state, synth.EVAL_FONTS, 320, 240)
    assert hashlib.sha256(image.tobytes()).hexdigest()[:16] == "be478fa5383e8c70"
    assert quad.round(3).tolist()[0] == [115.284, 72.222]


def test_generator_writes_jpg_and_json_pairs(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    photo, label = gen.render_picture_photo(rng, make_picture(), 4, 480, 640)
    photo.save(tmp_path / "0000.jpg")
    (tmp_path / "0000.json").write_text(json.dumps(label))
    assert json.loads((tmp_path / "0000.json").read_text())["size"] == 4
