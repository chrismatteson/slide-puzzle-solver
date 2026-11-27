import json
from pathlib import Path

import gen_correction_photos as gen
import gen_picture_photos as base
import numpy as np
import pytest
import synth
from PIL import Image


def make_picture(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8))


def tiles_of(label: dict[str, object]) -> list[int]:
    """Rebuild the photo's tile order (the position of each photo cell's tile, -1 for the slot)."""
    solved = label["solved"]
    assert isinstance(solved, list)
    cells = [-2] * len(solved)
    for position, cell in enumerate(solved):
        if cell != -1:
            cells[cell] = position
    cells[int(str(label["blank"]))] = -1
    return cells


def is_reachable(cells: list[int], size: int) -> bool:
    """Whether sliding can turn the solved picture into `cells`.

    Each slide is one swap of the slot with a tile, so the permutation's parity must equal the parity of
    the slot's Manhattan distance from its home.
    """
    home = next(p for p in range(size * size) if p not in cells)
    permutation = [home if position == -1 else position for position in cells]
    swaps = len(permutation) - len(cycles(permutation))
    (slot_row, slot_col), (home_row, home_col) = divmod(cells.index(-1), size), divmod(home, size)
    return swaps % 2 == (abs(slot_row - home_row) + abs(slot_col - home_col)) % 2


def cycles(permutation: list[int]) -> list[list[int]]:
    """Return the cycles of a permutation of 0..n-1."""
    seen: set[int] = set()
    found = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        at = start
        while at not in seen:
            seen.add(at)
            cycle.append(at)
            at = permutation[at]
        found.append(cycle)
    return found


def centre_is_inside(quad: np.ndarray, point: np.ndarray) -> bool:
    """Whether `point` lies inside the convex `quad`."""
    signs = []
    for i in range(4):
        edge, to_point = quad[(i + 1) % 4] - quad[i], point - quad[i]
        signs.append(edge[0] * to_point[1] - edge[1] * to_point[0] > 0)
    return all(signs) or not any(signs)


def corners_of(label: dict[str, object], key: str) -> np.ndarray:
    """The label's four {"x", "y"} points as a 4x2 array."""
    points = label[key]
    assert isinstance(points, list)
    return np.array([[p["x"], p["y"]] for p in points])


@pytest.mark.parametrize("mode", gen.MODES)
@pytest.mark.parametrize("size", [3, 4])
def test_label_matches_the_rendered_puzzle(mode: str, size: int) -> None:
    rng = np.random.default_rng(10 + size)
    photo, label = gen.render_correction_photo(rng, mode, make_picture(), size, 640, 480)
    assert photo.size == (640, 480)
    assert label["mode"] == mode
    assert label["size"] == size
    solved = label["solved"]
    assert isinstance(solved, list)
    assert sorted(solved) == [-1, *[c for c in range(size * size) if c != label["blank"]]]
    for key in ("corners", "tilesQuad"):
        assert corners_of(label, key).shape == (4, 2)
    json.dumps(label)  # plain JSON types only


@pytest.mark.parametrize("mode", gen.MODES)
@pytest.mark.parametrize("size", [3, 4])
def test_boards_are_solvable_by_sliding(mode: str, size: int) -> None:
    for seed in range(3):
        rng = np.random.default_rng(seed)
        _, label = gen.render_correction_photo(rng, mode, make_picture(), size, 640, 480)
        assert is_reachable(tiles_of(label), size)


@pytest.mark.parametrize("mode", gen.MODES)
def test_same_seed_gives_the_same_photo_and_label(mode: str) -> None:
    first = gen.render_correction_photo(np.random.default_rng(7), mode, make_picture(), 3, 320, 240)
    second = gen.render_correction_photo(
        np.random.default_rng(7), mode, make_picture(), 3, 320, 240
    )
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1] == second[1]
    other = gen.render_correction_photo(np.random.default_rng(8), mode, make_picture(), 3, 320, 240)
    assert other[1] != first[1]


def test_thumbnail_modes_label_the_print_and_angled_mode_does_not() -> None:
    rng = np.random.default_rng(3)
    for mode in gen.MODES:
        _, label = gen.render_correction_photo(rng, mode, make_picture(), 3, 640, 480)
        assert ("thumbQuad" in label) == (mode != "angled-blank")
        assert ("thumbShowsBlank" in label) == (mode != "angled-blank")


def test_white_frame_is_wide_and_the_thumbnail_lies_in_it() -> None:
    rng = np.random.default_rng(4)
    for _ in range(5):
        photo, label = gen.render_correction_photo(
            rng, "white-frame-thumbnail", make_picture(), 3, 960, 960
        )
        tiles = corners_of(label, "tilesQuad")
        outer = corners_of(label, "corners")
        thumb = corners_of(label, "thumbQuad")
        # Frame width as a share of the tile area, from the side lengths (the view is square-on enough).
        tile_side = np.linalg.norm(tiles[1] - tiles[0])
        outer_side = np.linalg.norm(outer[1] - outer[0])
        assert 0.14 <= (outer_side - tile_side) / 2 / tile_side <= 0.45
        assert not centre_is_inside(tiles, thumb.mean(axis=0))
        assert centre_is_inside(outer, thumb.mean(axis=0))
        assert photo.size == (960, 960)


def test_tilt_is_between_30_and_50_degrees() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        quad = gen.make_tilted_quad(rng, 960, 960, base.BOARD_SCALE)
        assert synth.is_convex(quad)
        # Seen square-on the diagonals are equal; a 30-50 degree tilt makes the near/far edges clearly differ.
        near, far = sorted(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4))[::3]
        assert far / near > 1.1


def test_empty_slot_is_unlike_plain_frame() -> None:
    rng = np.random.default_rng(6)
    frame = (120, 110, 100)
    for _ in range(10):
        slot = np.asarray(gen.make_textured_slot(rng, 80, frame), float)
        assert np.abs(slot.mean(axis=(0, 1)) - np.array(frame)).max() > 10 or slot.std() > 5


def test_drawings_are_deterministic_and_flat_looking() -> None:
    first = gen.draw_drawing(np.random.default_rng(1))
    again = gen.draw_drawing(np.random.default_rng(1))
    assert first.tobytes() == again.tobytes()
    grey = np.asarray(first.convert("L"), float)
    # Line art: most pixels are paper or flat fill, a few are dark ink.
    assert (grey < 80).mean() < 0.15
    assert (grey < 80).mean() > 0.005


def test_flatten_shows_transparency_as_white() -> None:
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert gen.flatten(image).getpixel((0, 0)) == (255, 255, 255)


def test_line_art_fails_when_a_manifest_picture_is_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "picture-sources.toml"
    manifest.write_text(
        '[[picture]]\nname = "missing-bear"\nset = "line-art"\nsplit = "dev"\n'
        '[[picture]]\nname = "other-split"\nset = "line-art"\nsplit = "eval"\n'
    )
    monkeypatch.setattr(base, "MANIFEST", manifest)
    monkeypatch.setattr(base, "PICTURE_DIR", tmp_path / "pictures")
    with pytest.raises(FileNotFoundError, match="missing-bear.*pictures:fetch"):
        gen.load_line_art("dev")


def test_line_art_uses_downloaded_pictures_and_drawn_art(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "picture-sources.toml"
    manifest.write_text('[[picture]]\nname = "bear"\nset = "line-art"\nsplit = "dev"\n')
    (tmp_path / "pictures").mkdir()
    Image.new("RGB", (40, 30), "white").save(tmp_path / "pictures" / "bear.jpg", format="PNG")
    monkeypatch.setattr(base, "MANIFEST", manifest)
    monkeypatch.setattr(base, "PICTURE_DIR", tmp_path / "pictures")
    pictures = gen.load_line_art("dev")
    assert "bear" in pictures
    assert len(pictures) == 1 + gen.PROCEDURAL_DRAWINGS["dev"]
