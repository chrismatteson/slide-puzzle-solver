"""Writes a deterministic set of labelled synthetic photos of picture-tile sliding puzzles.

One photo is NNNN.jpg plus NNNN.json: `size`, `blank` (photo cell of the empty slot), `solved`
(photo cell of the tile that belongs at each solved position, -1 at the empty slot's home, as in the
owner's label), `corners` (the frame's outer quad) and `tilesQuad` (the tile area's quad), both TL, TR,
BR, BL in photo pixels. The pictures come from picture-sources.toml (`npm run pictures:fetch`);
`--split` picks the dev pictures (tuning) or the held-out eval pictures.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import synth
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "picture-sources.toml"
PICTURE_DIR = ROOT / "data" / "pictures"
# Where the empty slot's home is: mostly the bottom-right corner, as on a bought puzzle.
HOME_BOTTOM_RIGHT = 0.70
HOME_OTHER_CORNER = 0.18  # the rest: anywhere
# Phone photos of a puzzle are large and the board fills most of the frame; the numbered set's tiny
# boards would leave a tile only ~50 px.
BOARD_SCALE = (0.55, 0.92)
PHOTO_LONG_SIDE, PHOTO_SHORT_SIDE = 1280, 960
# `--real-like`: what the owner's photo has and the default boards lack. Each tile sits REAL_LIKE_SLOP
# pixels (of the 640 px flat board) out of line, is glossy (a bright, partly saturated highlight over
# each tile), and the whole board is soft.
REAL_LIKE_SLOP = (2.0, 5.0)
REAL_LIKE_BLUR = (1.0, 2.5)


def load_pictures(split: str) -> dict[str, Image.Image]:
    """Return the downloaded paintings and photos of one split of the manifest, by name."""
    entries = tomllib.loads(MANIFEST.read_text())["picture"]
    pictures: dict[str, Image.Image] = {}
    for entry in entries:
        if entry["split"] != split or entry.get("set", "painting") != "painting":
            continue
        path = PICTURE_DIR / f"{entry['name']}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run npm run pictures:fetch")
        pictures[entry["name"]] = Image.open(path).convert("RGB")
    if not pictures:
        raise ValueError(f"no {split} pictures in {MANIFEST.name}")
    return pictures


def choose_home(rng: np.random.Generator, size: int) -> int:
    """Pick the position where the empty slot sits when the puzzle is solved."""
    last = size * size - 1
    draw = float(rng.random())
    if draw < HOME_BOTTOM_RIGHT:
        return last
    if draw < HOME_BOTTOM_RIGHT + HOME_OTHER_CORNER:
        return int(rng.choice([0, size - 1, last - size + 1]))
    return int(rng.integers(0, last + 1))


def scramble(rng: np.random.Generator, size: int, home: int) -> list[int]:
    """Return `cells`: the position in the solved layout of the tile in each photo cell (-1: slot).

    A long random walk of the slot away from its home, so the state is always solvable.
    """
    cells = [*range(size * size)]
    cells[home] = -1
    slot = home
    previous = -1
    for _ in range(int(rng.integers(150, 400)) * size):
        row, col = divmod(slot, size)
        options = [
            r * size + c
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if 0 <= r < size and 0 <= c < size and r * size + c != previous
        ]
        target = int(rng.choice(options))
        cells[slot], cells[target] = cells[target], -1
        previous, slot = slot, target
    if all(cell in (-1, index) for index, cell in enumerate(cells)):
        return scramble(rng, size, home)
    return cells


def crop_square(rng: np.random.Generator, picture: Image.Image) -> Image.Image:
    """Cut a random square (60-100% of the short side) out of the picture."""
    width, height = picture.size
    side = int(min(width, height) * float(rng.uniform(0.6, 1.0)))
    left = int(rng.integers(0, width - side + 1))
    top = int(rng.integers(0, height - side + 1))
    return picture.crop((left, top, left + side, top + side))


def add_glare(rng: np.random.Generator, flat: Image.Image) -> Image.Image:
    """Lay a soft bright band over part of the board, like a lamp reflecting off glossy tiles."""
    size = flat.size[0]
    layer = Image.new("L", flat.size, 0)
    draw = ImageDraw.Draw(layer)
    offset = float(rng.uniform(0.0, 1.0)) * size
    width = float(rng.uniform(0.05, 0.25)) * size
    draw.polygon(
        [
            (offset, 0),
            (offset + width, 0),
            (offset + width - 0.4 * size, size),
            (offset - 0.4 * size, size),
        ],
        fill=int(rng.uniform(40, 140)),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(8, 30))))
    white = Image.new("RGB", flat.size, (255, 255, 255))
    return Image.composite(white, flat, layer)


def add_tile_gloss(rng: np.random.Generator, tile: Image.Image) -> Image.Image:
    """Lay a bright, blurred highlight over one glossy tile; its core clips to white."""
    width, height = tile.size
    layer = Image.new("L", tile.size, 0)
    draw = ImageDraw.Draw(layer)
    cx = float(rng.uniform(0.0, 1.0)) * width
    cy = float(rng.uniform(0.0, 1.0)) * height
    rx = float(rng.uniform(0.15, 0.6)) * width
    ry = float(rng.uniform(0.05, 0.3)) * height
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(rng.uniform(60, 255)))
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(4, 15))))
    white = Image.new("RGB", tile.size, (255, 255, 255))
    return Image.composite(white, tile, layer)


def draw_colour(rng: np.random.Generator, low: int, high: int) -> tuple[int, int, int]:
    """Return a random RGB colour with every channel in [low, high)."""
    red, green, blue = (int(v) for v in rng.integers(low, high, size=3))
    return red, green, blue


def paste_tiles(
    rng: np.random.Generator,
    board: Image.Image,
    art: Image.Image,
    size: int,
    cells: list[int],
    origin: float,
    pitch: float,
    make_slot: Callable[[], Image.Image],
    real_like: bool = False,
) -> None:
    """Paste the tiles of `art` onto `board` in `cells` order, each with its gap, slop and shadow.

    `make_slot` makes the image shown where the slot is empty (`pitch` pixels square).
    """
    gap = float(rng.uniform(0.006, 0.03)) * pitch
    slop = float(rng.uniform(0.0, 0.02)) * pitch  # tiles sit loosely in the frame
    if real_like:
        slop = float(rng.uniform(*REAL_LIKE_SLOP))
    for index, home in enumerate(cells):
        row, col = divmod(index, size)
        left = origin + col * pitch
        top = origin + row * pitch
        if home == -1:
            tile = make_slot()
        else:
            h_row, h_col = divmod(home, size)
            box = (
                int(round(h_col * pitch)),
                int(round(h_row * pitch)),
                int(round((h_col + 1) * pitch)),
                int(round((h_row + 1) * pitch)),
            )
            tile = art.crop(box)
            brightness = float(rng.uniform(0.93, 1.05))
            tile = tile.point(lambda v, b=brightness: min(255, int(v * b)))
            if real_like and rng.random() < 0.7:
                tile = add_tile_gloss(rng, tile)
        dx, dy = (float(v) for v in rng.uniform(-slop, slop, size=2))
        x = int(round(left + gap / 2 + dx))
        y = int(round(top + gap / 2 + dy))
        inset = max(int(round(gap)), 1)
        tile = tile.resize(
            (max(int(pitch) - inset, 1), max(int(pitch) - inset, 1)), Image.Resampling.BILINEAR
        )
        shadow = Image.new("L", tile.size, 90)
        shadow = shadow.filter(ImageFilter.GaussianBlur(2))
        board.paste((0, 0, 0), (x + 2, y + 2), shadow)
        board.paste(tile, (x, y))


def render_flat_picture_board(
    rng: np.random.Generator,
    picture: Image.Image,
    size: int,
    cells: list[int],
    real_like: bool = False,
) -> tuple[Image.Image, float, tuple[int, int, int]]:
    """Render the board square: frame, tiles of `picture` in `cells` order, gaps, shadows, empty slot.

    Returns the image, the frame's share of the edge and the frame colour.
    """
    edge = synth.FLAT_SIZE
    margin = float(rng.uniform(0.05, 0.10))
    frame = draw_colour(rng, 8, 70)
    light_slot = rng.random() < 0.3
    slot_colour = draw_colour(rng, 190, 250) if light_slot else draw_colour(rng, 0, 30)
    inner = int(round(edge * (1 - 2 * margin)))
    pitch = inner / size
    origin = edge * margin
    art = crop_square(rng, picture).resize((inner, inner), Image.Resampling.LANCZOS)
    board = Image.new("RGB", (edge, edge), frame)
    paste_tiles(
        rng,
        board,
        art,
        size,
        cells,
        origin,
        pitch,
        lambda: Image.new("RGB", (int(pitch),) * 2, slot_colour),
        real_like,
    )
    if rng.random() < 0.55 or real_like:
        board = add_glare(rng, board)
    if real_like:
        board = board.filter(ImageFilter.GaussianBlur(float(rng.uniform(*REAL_LIKE_BLUR))))
    return board, margin, frame


def list_solved_layout(cells: list[int]) -> list[int]:
    """Return the label's `solved`: the photo cell of the tile that belongs at each position (-1: home)."""
    solved = [-1] * len(cells)
    for cell, position in enumerate(cells):
        if position != -1:
            solved[position] = cell
    return solved


def to_points(quad: np.ndarray) -> list[dict[str, float]]:
    """Corners as {"x", "y"} objects, the form of the owner's label."""
    return [{"x": round(float(x), 1), "y": round(float(y), 1)} for x, y in quad]


def render_picture_photo(
    rng: np.random.Generator,
    picture: Image.Image,
    size: int,
    width: int,
    height: int,
    real_like: bool = False,
) -> tuple[Image.Image, dict[str, object]]:
    """Render one photo of a scrambled picture puzzle and its label."""
    home = choose_home(rng, size)
    cells = scramble(rng, size, home)
    flat, margin, frame = render_flat_picture_board(rng, picture, size, cells, real_like)
    photo, quad, to_photo = synth.photograph_flat_board(
        rng, flat, margin, size, frame, width, height, BOARD_SCALE
    )
    low, high = synth.FLAT_SIZE * margin, synth.FLAT_SIZE * (1 - margin)
    tiles_quad = to_photo(np.array([[low, low], [high, low], [high, high], [low, high]]))
    label: dict[str, object] = {
        "size": size,
        "blank": cells.index(-1),
        "solved": list_solved_layout(cells),
        "corners": to_points(quad),
        "tilesQuad": to_points(tiles_quad),
    }
    return photo, label


def main() -> None:
    """Generate a labelled picture-puzzle photo set from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "eval"], required=True)
    parser.add_argument("--count", type=int, default=144)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--real-like",
        action="store_true",
        help="tiles 2-5 px out of line, glossy highlights, soft focus (see REAL_LIKE_*)",
    )
    args = parser.parse_args()

    pictures = load_pictures(args.split)
    names = sorted(pictures)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.count):
        size = 3 if index % 2 == 0 else 4
        name = names[(index // 2) % len(names)]  # every picture appears at both sizes
        landscape = bool(rng.random() < 0.5)
        width, height = (
            (PHOTO_LONG_SIDE, PHOTO_SHORT_SIDE)
            if landscape
            else (PHOTO_SHORT_SIDE, PHOTO_LONG_SIDE)
        )
        photo, label = render_picture_photo(
            rng, pictures[name], size, width, height, args.real_like
        )
        stem = f"{index:04d}"
        photo.save(args.out / f"{stem}.jpg", quality=95)  # already degraded; keep it as is
        (args.out / f"{stem}.json").write_text(json.dumps({**label, "picture": name}))
    print(f"wrote {args.count} images to {args.out}")


if __name__ == "__main__":
    main()
