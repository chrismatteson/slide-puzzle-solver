"""Writes seeded, labelled photo sets that reproduce three ways picture-puzzle reading fails.

Each set goes to its own folder of NNNN.jpg + NNNN.json pairs under --out, with the label of
gen_picture_photos.py (`size`, `blank`, `solved`, `corners`, `tilesQuad`) plus `picture`, `mode` and,
where a printed copy of the solved picture is in the photo, `thumbQuad` (its corners, TL, TR, BR, BL)
and `thumbShowsBlank` (whether the copy leaves the empty slot's home blank):

- angled-blank: 30-50 degree tilt; the empty slot shows a dark, textured base with shadows from its
  neighbours, and the frame is close to the picture's tones, so it does not look like plain frame;
- white-frame-thumbnail: a wide white frame (15-40% of the tile area) with a printed thumbnail of the
  solved picture and text-like decoration on it, so the tile area is easy to mistake;
- line-art: dark line drawings on white or flat pastel, little detail for edges to continue; a printed
  thumbnail sits on the frame as the reference. The drawings are the line art of picture-sources.toml
  (it requires `npm run pictures:fetch`) plus procedurally drawn ones.

`--split` picks the dev pictures (tuning) or the held-out eval pictures.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gen_picture_photos as base
import numpy as np
import synth
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

MODES = ("angled-blank", "white-frame-thumbnail", "line-art")
TILT_DEGREES = (30.0, 50.0)
# Camera distance in board sides: close enough that a 40 degree tilt gives the near edge ~1.4x the far one.
CAMERA_DISTANCE = 2.0
# Share of the tile area the frame is wide, per side.
WHITE_FRAME_WIDTH = (0.15, 0.40)
LINE_ART_FRAME_WIDTH = (0.25, 0.40)
PROCEDURAL_DRAWINGS = {"dev": 4, "eval": 8}
# The drawings are seeded apart from the photos, so every run of a split uses the same pictures.
DRAWING_SEED = 770031
ART_SIDE = 1024
PASTELS = ((255, 214, 220), (255, 236, 179), (200, 230, 255), (212, 240, 207), (235, 215, 250))


def flatten(image: Image.Image) -> Image.Image:
    """Return `image` as RGB, with any transparency shown as white paper (not black)."""
    rgba = image.convert("RGBA")
    paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(paper, rgba).convert("RGB")


def draw_drawing(rng: np.random.Generator) -> Image.Image:
    """Draw a simple scene (creature, sun, clouds, tree, ground) in dark outline on flat fills."""
    scale = 2  # drawn at twice the size and shrunk, for smooth lines
    side = ART_SIDE * scale
    paper = (252, 252, 252) if rng.random() < 0.6 else PASTELS[int(rng.integers(len(PASTELS)))]
    ink = (int(rng.integers(0, 45)),) * 3
    line = int(rng.integers(4, 8)) * scale
    image = Image.new("RGB", (side, side), paper)
    draw = ImageDraw.Draw(image)

    def pick_fill_colour() -> tuple[int, int, int]:
        return (252, 252, 252) if rng.random() < 0.5 else PASTELS[int(rng.integers(len(PASTELS)))]

    def draw_ellipse(cx: float, cy: float, rx: float, ry: float) -> None:
        draw.ellipse(
            (cx - rx, cy - ry, cx + rx, cy + ry), fill=pick_fill_colour(), outline=ink, width=line
        )

    horizon = float(rng.uniform(0.62, 0.8)) * side
    draw.rectangle((0, horizon, side, side), fill=pick_fill_colour())
    draw.line((0, horizon, side, horizon), fill=ink, width=line)
    sun_x, sun_y, sun_r = (float(v) * side for v in (rng.uniform(0.1, 0.9), 0.14, 0.07))
    for k in range(int(rng.integers(8, 13))):
        angle = 2 * math.pi * k / 10
        draw.line(
            (
                sun_x + 1.3 * sun_r * math.cos(angle),
                sun_y + 1.3 * sun_r * math.sin(angle),
                sun_x + 1.9 * sun_r * math.cos(angle),
                sun_y + 1.9 * sun_r * math.sin(angle),
            ),
            fill=ink,
            width=line,
        )
    draw_ellipse(sun_x, sun_y, sun_r, sun_r)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = float(rng.uniform(0.15, 0.85)) * side, float(rng.uniform(0.25, 0.4)) * side
        for dx in (-0.05, 0.0, 0.05):
            draw_ellipse(cx + dx * side, cy, 0.05 * side, 0.035 * side)
    tree_x = float(rng.uniform(0.1, 0.9)) * side
    draw.rectangle(
        (
            tree_x - 0.015 * side,
            horizon - 0.18 * side,
            tree_x + 0.015 * side,
            horizon + 0.03 * side,
        ),
        fill=pick_fill_colour(),
        outline=ink,
        width=line,
    )
    draw_ellipse(tree_x, horizon - 0.24 * side, 0.09 * side, 0.09 * side)
    # The creature: body, head, ears, eyes, smile and legs.
    bx, by = float(rng.uniform(0.35, 0.65)) * side, horizon - 0.02 * side
    rx, ry = float(rng.uniform(0.14, 0.22)) * side, float(rng.uniform(0.08, 0.13)) * side
    for leg in (-0.6, -0.25, 0.25, 0.6):
        draw.line(
            (bx + leg * rx, by + ry * 0.7, bx + leg * rx, by + ry * 1.6), fill=ink, width=2 * line
        )
    draw_ellipse(bx, by, rx, ry)
    hx, hy, hr = bx + rx * 0.95, by - ry * 1.1, ry * 0.9
    for ear in (-0.6, 0.6):
        ex = hx + ear * hr
        draw.polygon(
            [(ex - 0.25 * hr, hy - 0.8 * hr), (ex, hy - 1.7 * hr), (ex + 0.25 * hr, hy - 0.8 * hr)],
            fill=pick_fill_colour(),
            outline=ink,
            width=line,
        )
    draw_ellipse(hx, hy, hr, hr)
    for eye in (-0.35, 0.35):
        draw.ellipse(
            (
                hx + eye * hr - 0.08 * hr,
                hy - 0.2 * hr - 0.08 * hr,
                hx + eye * hr + 0.08 * hr,
                hy - 0.2 * hr + 0.08 * hr,
            ),
            fill=ink,
        )
    draw.arc(
        (hx - 0.4 * hr, hy - 0.1 * hr, hx + 0.4 * hr, hy + 0.5 * hr), 20, 160, fill=ink, width=line
    )
    return image.resize((ART_SIDE, ART_SIDE), Image.Resampling.LANCZOS)


def load_line_art(split: str) -> dict[str, Image.Image]:
    """Return the line drawings of one split by name: downloaded ones of the manifest, then drawn ones.

    Raises FileNotFoundError when a manifest picture of the split is not downloaded.
    """
    entries = tomllib.loads(base.MANIFEST.read_text())["picture"]
    pictures: dict[str, Image.Image] = {}
    for entry in entries:
        if entry["split"] != split or entry.get("set") != "line-art":
            continue
        path = base.PICTURE_DIR / f"{entry['name']}.jpg"  # may hold a PNG; Pillow goes by content
        if path.exists():
            pictures[entry["name"]] = flatten(Image.open(path))
        else:
            # Drawn art alone would silently change the set the held-out numbers are measured on.
            msg = f"{path.name} is not downloaded; run `npm run pictures:fetch`"
            raise FileNotFoundError(msg)
    split_index = ("dev", "eval").index(split)
    for index in range(PROCEDURAL_DRAWINGS[split]):
        rng = np.random.default_rng([DRAWING_SEED, split_index, index])
        pictures[f"drawn-{split}-{index:02d}"] = draw_drawing(rng)
    return pictures


def make_tilted_quad(
    rng: np.random.Generator, width: int, height: int, scale: tuple[float, float]
) -> np.ndarray:
    """Board corners (TL, TR, BR, BL) of a square board tilted TILT_DEGREES about a random hinge line."""
    short = min(width, height)
    for _ in range(1000):
        tilt = math.radians(float(rng.uniform(*TILT_DEGREES)))
        hinge = float(rng.uniform(0, 2 * math.pi))
        along_axis = np.array([math.cos(hinge), math.sin(hinge)])
        across_axis = np.array([-along_axis[1], along_axis[0]])
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        along, across = corners @ along_axis, corners @ across_axis
        # Tilting about the hinge line moves a corner away from the camera by its distance from the line.
        depth = CAMERA_DISTANCE + across * math.sin(tilt)
        flat = across * math.cos(tilt)
        points = np.stack(
            [
                along * along_axis[0] + flat * across_axis[0],
                along * along_axis[1] + flat * across_axis[1],
            ],
            axis=1,
        )
        points = points / depth[:, None]
        roll = math.radians(float(rng.uniform(-20, 20)))  # the camera is not held square either
        points = points @ np.array(
            [[math.cos(roll), math.sin(roll)], [-math.sin(roll), math.cos(roll)]]
        )
        points = points * (
            float(rng.uniform(*scale)) * short / float((points.max(0) - points.min(0)).max())
        )
        centre = np.array([rng.uniform(0.35, 0.65) * width, rng.uniform(0.35, 0.65) * height])
        quad = points + centre
        inside = (
            (quad >= 2).all()
            and (quad[:, 0] <= width - 2).all()
            and (quad[:, 1] <= height - 2).all()
        )
        if inside and synth.is_convex(quad):
            return np.asarray(quad, np.float64)
    raise RuntimeError("could not place a tilted board in the frame")


def make_textured_slot(
    rng: np.random.Generator, pitch: int, frame: tuple[int, int, int]
) -> Image.Image:
    """Return an empty slot unlike plain frame: darker base, grain, shadow from its neighbours."""
    factor = float(rng.uniform(0.45, 0.8)) if rng.random() < 0.7 else float(rng.uniform(1.1, 1.3))
    grain = rng.normal(0, float(rng.uniform(8, 22)), size=(pitch, pitch, 1))
    pixels = np.asarray(frame, np.float64) * factor + grain
    # The raised neighbours shade the recess along its top and left edges.
    ramp = np.clip(1 - np.arange(pitch) / (pitch * float(rng.uniform(0.15, 0.3))), 0, 1)
    shade = 1 - float(rng.uniform(0.3, 0.6)) * np.maximum(ramp[:, None], ramp[None, :])
    slot = Image.fromarray(np.clip(pixels * shade[..., None], 0, 255).astype(np.uint8))
    return slot.filter(ImageFilter.GaussianBlur(1))


def draw_decoration(rng: np.random.Generator, length: int, thickness: int) -> Image.Image:
    """Print a logo and lines of text-like blocks on a transparent-white strip, length x thickness."""
    strip = Image.new("RGBA", (length, thickness), (0, 0, 0, 0))
    draw = ImageDraw.Draw(strip)
    logo = (int(rng.integers(20, 200)), int(rng.integers(20, 200)), int(rng.integers(20, 200)), 255)
    if rng.random() < 0.5:
        draw.ellipse((0, 0, thickness, thickness), fill=logo)
    else:
        draw.polygon([(0, thickness), (thickness / 2, 0), (thickness, thickness)], fill=logo)
    line_height = max(thickness // 7, 2)
    left = thickness + max(thickness // 4, 4)
    for k in range(int(rng.integers(2, 5))):
        y = k * 2 * line_height + line_height
        x = left
        while x < length - 12:
            block = int(rng.integers(8, 40))
            draw.rectangle((x, y, min(x + block, length), y + line_height), fill=(70, 70, 70, 255))
            x += block + int(rng.integers(5, 10))
    return strip


@dataclass
class FramedBoard:
    """A flat board render with the printed thumbnail's place in it (if it has one)."""

    image: Image.Image
    margin: float  # the frame's share of the edge
    thumb_quad: np.ndarray | None  # flat-board pixels, TL, TR, BR, BL
    thumb_shows_blank: bool


def render_framed_board(
    rng: np.random.Generator,
    picture: Image.Image,
    size: int,
    cells: list[int],
    margin: float,
    frame: tuple[int, int, int],
    make_slot: Callable[[int], Image.Image],
    thumbnail: bool,
    decorate: bool,
) -> FramedBoard:
    """Render the flat board square; with `thumbnail`, print the solved picture small on the frame.

    `margin` is the frame's share of the edge on each side. The home of the empty slot is blank in the
    thumbnail half of the time.
    """
    edge = synth.FLAT_SIZE
    inner = int(round(edge * (1 - 2 * margin)))
    pitch = inner / size
    origin = edge * margin
    art = base.crop_square(rng, picture).resize((inner, inner), Image.Resampling.LANCZOS)
    board = Image.new("RGB", (edge, edge), frame)
    base.paste_tiles(rng, board, art, size, cells, origin, pitch, lambda: make_slot(int(pitch)))
    thumb_quad = None
    shows_blank = False
    band = int(origin)
    if thumbnail:
        side = int(band * float(rng.uniform(0.8, 0.95)))
        along = int(rng.integers(0, edge - side + 1))
        across = (band - side) // 2
        place = int(rng.integers(4))  # which side of the grid: top, bottom, left, right
        x0, y0 = (
            (along, across),
            (along, edge - band + across),
            (across, along),
            (edge - band + across, along),
        )[place]
        thumb = art.resize((side, side), Image.Resampling.BOX)
        shows_blank = bool(rng.random() < 0.5)
        if shows_blank:
            # Paint the home of the empty slot in the frame colour: the print of a puzzle with its
            # slot empty.
            home = base.list_solved_layout(cells).index(-1)
            row, col = divmod(home, size)
            box = (
                col * side // size,
                row * side // size,
                (col + 1) * side // size,
                (row + 1) * side // size,
            )
            thumb.paste(frame, box)
        # Print: duller colours than the tiles', as a paper copy of a photo is.
        thumb = ImageEnhance.Color(thumb).enhance(float(rng.uniform(0.7, 1.0)))
        thumb = ImageEnhance.Contrast(thumb).enhance(float(rng.uniform(0.85, 1.0)))
        thumb = ImageEnhance.Brightness(thumb).enhance(float(rng.uniform(1.0, 1.1)))
        card = (x0 - 4, y0 - 4, x0 + side + 4, y0 + side + 4)
        ImageDraw.Draw(board).rectangle(card, fill=(250, 250, 250))
        board.paste(thumb, (x0, y0))
        thumb_quad = np.array(
            [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], float
        )
        if decorate:
            opposite = place ^ 1  # the side across the grid: top<->bottom, left<->right
            inset = band // 5
            strip = draw_decoration(rng, edge - 2 * inset, band - 2 * inset)
            offset = inset if opposite in (0, 2) else edge - band + inset
            if opposite < 2:
                board.paste(strip, (inset, offset), strip)
            else:
                board.paste(
                    strip.rotate(90, expand=True), (offset, inset), strip.rotate(90, expand=True)
                )
    if rng.random() < 0.55:
        board = base.add_glare(rng, board)
    return FramedBoard(board, margin, thumb_quad, shows_blank)


def render_correction_photo(
    rng: np.random.Generator,
    mode: str,
    picture: Image.Image,
    size: int,
    width: int,
    height: int,
) -> tuple[Image.Image, dict[str, object]]:
    """Render one photo of a scrambled puzzle failing in `mode`, and its label."""
    home = base.choose_home(rng, size)
    cells = base.scramble(rng, size, home)
    quad = None
    if mode == "angled-blank":
        margin = float(rng.uniform(0.05, 0.10))
        mean = np.asarray(picture.resize((32, 32)), np.float64).reshape(-1, 3).mean(axis=0)
        jitter = rng.uniform(-18, 18, size=3)
        red, green, blue = (int(v) for v in np.clip(mean + jitter, 0, 255))
        frame = (red, green, blue)
        framed = render_framed_board(
            rng,
            picture,
            size,
            cells,
            margin,
            frame,
            lambda pitch: make_textured_slot(rng, pitch, frame),
            thumbnail=False,
            decorate=False,
        )
        quad = make_tilted_quad(rng, width, height, base.BOARD_SCALE)
    elif mode == "white-frame-thumbnail":
        share = float(rng.uniform(*WHITE_FRAME_WIDTH))  # of the tile area
        margin = share / (1 + 2 * share)
        paper = int(rng.integers(232, 253))
        frame = (paper, paper - int(rng.integers(0, 5)), paper - int(rng.integers(0, 12)))
        slot = (
            int(frame[0] * rng.uniform(0.88, 1.0)),
            int(frame[1] * rng.uniform(0.88, 1.0)),
            int(frame[2] * rng.uniform(0.88, 1.0)),
        )
        framed = render_framed_board(
            rng,
            picture,
            size,
            cells,
            margin,
            frame,
            lambda pitch: Image.new("RGB", (pitch, pitch), slot),
            thumbnail=True,
            decorate=True,
        )
    elif mode == "line-art":
        share = float(rng.uniform(*LINE_ART_FRAME_WIDTH))
        margin = share / (1 + 2 * share)
        frame = base.draw_colour(rng, 8, 70)
        slot_colour = (
            base.draw_colour(rng, 190, 250) if rng.random() < 0.3 else base.draw_colour(rng, 0, 30)
        )
        framed = render_framed_board(
            rng,
            picture,
            size,
            cells,
            margin,
            frame,
            lambda pitch: Image.new("RGB", (pitch, pitch), slot_colour),
            thumbnail=True,
            decorate=False,
        )
    else:
        raise ValueError(f"unknown mode {mode}")
    photo, outer, to_photo = synth.photograph_flat_board(
        rng, framed.image, framed.margin, size, frame, width, height, base.BOARD_SCALE, quad
    )
    low, high = synth.FLAT_SIZE * framed.margin, synth.FLAT_SIZE * (1 - framed.margin)
    tiles_quad = to_photo(np.array([[low, low], [high, low], [high, high], [low, high]]))
    label: dict[str, object] = {
        "size": size,
        "blank": cells.index(-1),
        "solved": base.list_solved_layout(cells),
        "corners": base.to_points(outer),
        "tilesQuad": base.to_points(tiles_quad),
        "mode": mode,
    }
    if framed.thumb_quad is not None:
        label["thumbQuad"] = base.to_points(to_photo(framed.thumb_quad))
        label["thumbShowsBlank"] = framed.thumb_shows_blank
    return photo, label


def main() -> None:
    """Generate the three failure-mode photo sets from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--split", choices=["dev", "eval"], required=True)
    parser.add_argument("--count", type=int, default=48, help="photos per mode, half 3x3, half 4x4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for number, mode in enumerate(MODES):
        pictures = (
            base.load_pictures(args.split) if mode != "line-art" else load_line_art(args.split)
        )
        names = sorted(pictures)
        folder = args.out / mode
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng([args.seed, number])
        for index in range(args.count):
            size = 3 if index % 2 == 0 else 4
            name = names[(index // 2) % len(names)]  # every picture appears at both sizes
            landscape = bool(rng.random() < 0.5)
            width, height = (
                (base.PHOTO_LONG_SIDE, base.PHOTO_SHORT_SIDE)
                if landscape
                else (base.PHOTO_SHORT_SIDE, base.PHOTO_LONG_SIDE)
            )
            photo, label = render_correction_photo(rng, mode, pictures[name], size, width, height)
            stem = f"{index:04d}"
            photo.save(folder / f"{stem}.jpg", quality=95)  # already degraded; keep it as is
            (folder / f"{stem}.json").write_text(json.dumps({**label, "picture": name}))
        print(f"wrote {args.count} {mode} images to {folder}")


if __name__ == "__main__":
    main()
