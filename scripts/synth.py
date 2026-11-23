"""Synthetic sliding-puzzle photos: flat board rendering, photographic degradation, cell crops.

Shared by gen_photos.py (labelled photos) and train_classifier.py (cell crops), so the classifier
trains on the same kind of tile the eval photographs show. Everything is driven by a numpy Generator
so a seed fully determines the output.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FONTS = sorted((ROOT / "fonts" / "train").glob("*.ttf"))
EVAL_FONTS = sorted((ROOT / "fonts" / "eval").glob("*.ttf"))
EVAL_FONT_NAMES = {font.stem for font in EVAL_FONTS}
FLAT_SIZE = 640  # edge of the flat board render, pixels


def compute_luminance(rgb: tuple[int, int, int]) -> float:
    """Return the perceived brightness (Rec. 601 luma) of an RGB colour."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def generate_random_colour(rng: np.random.Generator) -> tuple[int, int, int]:
    """Return a random colour, sometimes saturated and sometimes near grey."""
    # Mix of saturated and near-grey colours, as real boards are either.
    if rng.random() < 0.3:
        v = int(rng.integers(20, 240))
        return (v, v, v)
    return tuple(int(x) for x in rng.integers(10, 250, size=3))  # type: ignore[return-value]


def pick_colour_with_contrast(
    rng: np.random.Generator, against: tuple[int, int, int], minimum: float
) -> tuple[int, int, int]:
    """Return a random colour at least `minimum` luminance levels from `against`."""
    for _ in range(200):
        c = generate_random_colour(rng)
        if abs(compute_luminance(c) - compute_luminance(against)) >= minimum:
            return c
    # Fall back to black or white, whichever is further away.
    return (0, 0, 0) if compute_luminance(against) > 128 else (255, 255, 255)


@dataclass
class BoardStyle:
    """Everything that decides how one board looks: colours, font, digit style and texture."""

    size: int
    base: tuple[int, int, int]
    tile: tuple[int, int, int]
    digit: tuple[int, int, int]
    gap: float  # fraction of the pitch
    margin: float  # frame margin, fraction of the board edge
    radius: float  # tile corner radius, fraction of the tile
    tile_jitter: int  # per-tile luminance jitter
    font_path: Path
    bevel: bool
    weight: float  # 0..1 position along a variable font's weight axis
    stroke: int  # extra stroke px at 320 px tile edge, thickens digits of fixed-weight fonts
    oldstyle: bool  # per-digit vertical offsets, imitating old-style numerals
    slant: float
    stretch: float
    # Realism. The defaults reproduce the old flat printed look.
    digit_style: str = "flat"  # flat, emboss, deboss, engrave, engrave_shadow, metal, low_contrast
    glossy: bool = False  # tile surface gradient and specular highlights
    specular: float = 0.0  # probability that a tile carries a highlight
    tile_shadow: bool = False  # soft drop shadow, so the gap between tiles is darker
    texture: str = "flat"  # board base: flat, wood, brushed, plastic
    metal_tint: tuple[int, int, int] = (0, 0, 0)  # gold or silver, for digit_style "metal"


DIGIT_STYLES = (
    ("flat", 0.40),
    ("emboss", 0.12),
    ("deboss", 0.10),
    ("engrave", 0.08),
    ("engrave_shadow", 0.07),
    ("metal", 0.11),
    ("low_contrast", 0.12),
)
TEXTURES = (("flat", 0.5), ("wood", 0.18), ("brushed", 0.16), ("plastic", 0.16))
METAL_TINTS = ((212, 175, 55), (192, 192, 200), (184, 115, 51))  # gold, silver, bronze


def choose_weighted(rng: np.random.Generator, options: tuple[tuple[str, float], ...]) -> str:
    """Pick one name from (name, weight) pairs with probability proportional to weight."""
    names, weights = zip(*options, strict=True)
    return str(rng.choice(names, p=np.asarray(weights) / sum(weights)))


def make_low_contrast_digit(
    rng: np.random.Generator, tile: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Return a digit colour 18-40 luminance levels from its tile: darker if there is room, else lighter."""
    delta = float(rng.uniform(18, 40))
    sign = -1 if compute_luminance(tile) > delta + 20 else 1
    return tuple(int(np.clip(c + sign * delta, 0, 255)) for c in tile)  # type: ignore[return-value]


def generate_random_style(rng: np.random.Generator, size: int, fonts: list[Path]) -> BoardStyle:
    """Return a random board style for a `size` x `size` board using `fonts`."""
    base = generate_random_colour(rng)
    tile = pick_colour_with_contrast(rng, base, 50)
    digit = pick_colour_with_contrast(rng, tile, 75)
    style = BoardStyle(
        size=size,
        base=base,
        tile=tile,
        digit=digit,
        gap=float(rng.uniform(0.02, 0.07)),
        margin=float(rng.uniform(0.02, 0.08)),
        radius=float(rng.uniform(0.0, 0.18)),
        tile_jitter=int(rng.choice([0, 0, 0, 12, 25])),
        font_path=fonts[int(rng.integers(len(fonts)))],
        bevel=bool(rng.random() < 0.5),
        weight=float(rng.random()),
        stroke=int(rng.choice([0, 0, 0, 3, 6])),
        oldstyle=bool(rng.random() < 0.35),
        slant=float(rng.normal(0, 0.08)) if rng.random() < 0.4 else 0.0,
        stretch=float(rng.uniform(0.85, 1.15)),
    )
    # Drawn last, so the older fields keep their position in the random stream.
    style.digit_style = choose_weighted(rng, DIGIT_STYLES)
    style.texture = choose_weighted(rng, TEXTURES)
    style.glossy = bool(rng.random() < 0.35)
    style.specular = (
        float(rng.choice([0.0, 0.3, 0.7])) if style.glossy or style.digit_style == "metal" else 0.0
    )
    style.tile_shadow = bool(rng.random() < 0.4)
    style.metal_tint = METAL_TINTS[int(rng.integers(len(METAL_TINTS)))]
    if style.digit_style == "low_contrast":
        style.digit = make_low_contrast_digit(rng, style.tile)
    if (
        style.digit_style == "metal"
        and abs(compute_luminance(style.metal_tint) - compute_luminance(style.tile)) < 50
    ):
        style.metal_tint = (
            style.digit
        )  # gold on a gold tile would be unreadable to anyone; use the contrasting colour
    return style


@lru_cache(maxsize=256)  # bounded: every (font, size, weight) is a separate open FreeType face
def load_font(path: Path, px: int, weight_step: int = 2) -> ImageFont.FreeTypeFont:
    """Load a font; a variable font's weight axis is set to weight_step/4 of its range (0 = lightest)."""
    font = ImageFont.truetype(str(path), px)
    try:
        axes = font.get_variation_axes()
    except OSError:
        axes = []  # not a variable font
    for index, axis in enumerate(axes):
        if axis["name"] in (b"Weight", "Weight"):
            values = [float(a["default"] or 0) for a in axes]
            lowest, highest = float(axis["minimum"] or 0), float(axis["maximum"] or 0)
            values[index] = lowest + weight_step / 4 * (highest - lowest)
            font.set_variation_by_axes(values)
    return font


def fit_font(
    path: Path, text: str, max_w: float, max_h: float, weight: float
) -> ImageFont.FreeTypeFont:
    """Largest size whose rendering of `text` fits in max_w x max_h (measured once at a reference size)."""
    step = round(weight * 4)
    reference = 100
    left, top, right, bottom = load_font(path, reference, step).getbbox(text)
    ratio = min(max_w / max(1, right - left), max_h / max(1, bottom - top))
    return load_font(path, max(8, int(reference * ratio)), step)


def draw_tile(rng: np.random.Generator, style: BoardStyle, number: int, edge: int) -> Image.Image:
    """One tile on a transparent square of `edge` pixels (the tile itself is smaller by the gap)."""
    scale = 2
    big = edge * scale
    tile_edge = big * (1 - style.gap)
    pad = (big - tile_edge) / 2
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    jitter = (
        int(rng.integers(-style.tile_jitter, style.tile_jitter + 1)) if style.tile_jitter else 0
    )
    colour = tuple(int(np.clip(c + jitter, 0, 255)) for c in style.tile)
    box = (pad, pad, big - pad, big - pad)
    radius = style.radius * tile_edge
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    if style.tile_shadow:
        add_drop_shadow(rng, img, box, radius, big)
        draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=radius, fill=colour + (255,))
    if style.glossy:
        add_gloss_gradient(rng, img, mask)
        draw = ImageDraw.Draw(img)
    if style.bevel:
        darker = tuple(int(c * 0.8) for c in colour)
        draw.rounded_rectangle(box, radius=radius, outline=darker + (255,), width=max(1, big // 80))
    text = str(number)
    max_h = tile_edge * float(rng.uniform(0.38, 0.62))
    max_w = tile_edge * float(rng.uniform(0.6, 0.8))
    font = fit_font(style.font_path, text, max_w, max_h, style.weight)
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    stroke = int(style.stroke * big / 640)
    if style.oldstyle and len(text) == 2:
        # Digits drawn one by one with their own vertical offset, like old-style numerals.
        advances = [font.getlength(ch) for ch in text]
        x = big / 2 - sum(advances) / 2
        for ch, advance in zip(text, advances, strict=True):
            dy = float(rng.normal(0, 0.07)) * max_h
            layer_draw.text(
                (x, big / 2 + dy),
                ch,
                font=font,
                fill=style.digit + (255,),
                anchor="lm",
                stroke_width=stroke,
                stroke_fill=style.digit + (255,),
            )
            x += advance
    else:
        dy = float(rng.normal(0, 0.07)) * max_h if style.oldstyle else 0.0
        layer_draw.text(
            (big / 2, big / 2 + dy),
            text,
            font=font,
            fill=style.digit + (255,),
            anchor="mm",
            stroke_width=stroke,
            stroke_fill=style.digit + (255,),
        )
    centre = big / 2
    stretch, slant = style.stretch, style.slant
    layer = layer.transform(
        layer.size,
        Image.Transform.AFFINE,
        (1 / stretch, -slant, centre - centre / stretch + slant * centre, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
    )
    angle = float(rng.normal(0, 2.5))
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(big / 2, big / 2))
    dx = int(rng.normal(0, 0.03) * big)
    dy = int(rng.normal(0, 0.03) * big)
    layer = layer.transform(layer.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy))
    # Clip the digit to the tile so a large glyph cannot spill into the gap.
    layer.putalpha(Image.fromarray(np.minimum(np.asarray(layer.getchannel("A")), np.asarray(mask))))
    composite_digit(rng, style, img, layer, colour)
    if style.specular and rng.random() < style.specular:
        add_specular(rng, img, mask)
    return img.resize((edge, edge), Image.Resampling.LANCZOS)


def shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Return `a` moved by (dx, dy) pixels, zero-filled.

    np.roll would wrap; the tile has transparent padding anyway.
    """
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = a[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def blend(img: Image.Image, rgb: np.ndarray | tuple[int, ...], alpha: np.ndarray) -> None:
    """Composites a colour (one RGB or a per-pixel array) over `img` where `alpha` (0..1) says so."""
    base = np.asarray(img, np.float32)
    a = alpha[..., None]
    base[..., :3] = base[..., :3] * (1 - a) + np.asarray(rgb, np.float32) * a
    base[..., 3] = np.maximum(base[..., 3], alpha * 255)
    img.paste(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)))


def add_drop_shadow(
    rng: np.random.Generator, img: Image.Image, box: tuple[float, ...], radius: float, big: int
) -> None:
    """Darken a blurred, offset copy of `box` on `img` so the tile looks raised."""
    offset = float(rng.uniform(0.008, 0.02)) * big
    shadow = Image.new("L", (big, big), 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (box[0] + offset, box[1] + offset, box[2] + offset, box[3] + offset),
        radius=radius,
        fill=255,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.008, 0.02)) * big))
    blend(img, (0, 0, 0), np.asarray(shadow, np.float32) / 255 * float(rng.uniform(0.3, 0.6)))


def add_gloss_gradient(rng: np.random.Generator, img: Image.Image, mask: Image.Image) -> None:
    """Lighter on one side of the tile, darker on the other, as a curved glossy surface reflects the room."""
    h, w = mask.size[1], mask.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = float(rng.uniform(0, 2 * math.pi))
    t = (xx * math.cos(angle) + yy * math.sin(angle)) / max(w, h)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    strength = float(rng.uniform(0.15, 0.45))
    factor = 1 + strength * (0.5 - t)  # 1 + s/2 .. 1 - s/2
    inside = np.asarray(mask, np.float32) / 255
    base = np.asarray(img, np.float32)
    base[..., :3] = base[..., :3] * (1 + (factor[..., None] - 1) * inside[..., None])
    img.paste(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)))


def add_specular(rng: np.random.Generator, img: Image.Image, mask: Image.Image) -> None:
    """Paint a bright blob or streak anywhere on the tile, so it can cover part of the digit."""
    h, w = mask.size[1], mask.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = float(rng.uniform(0.2, 0.8)) * w, float(rng.uniform(0.2, 0.8)) * h
    angle = float(rng.uniform(0, math.pi))
    u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
    v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
    if rng.random() < 0.5:  # blob
        su, sv = (float(rng.uniform(0.08, 0.25)) * w,) * 2
    else:  # streak
        su, sv = float(rng.uniform(0.3, 0.7)) * w, float(rng.uniform(0.03, 0.09)) * w
    strength = float(rng.uniform(0.45, 0.95))
    highlight = (
        strength
        * np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))
        * np.asarray(mask, np.float32)
        / 255
    )
    blend(img, (255, 255, 255), highlight)


def composite_digit(
    rng: np.random.Generator,
    style: BoardStyle,
    img: Image.Image,
    layer: Image.Image,
    tile: tuple[int, ...],
) -> None:
    """Draws the digit (whose shape is `layer`'s alpha) onto the tile in the board's digit style."""
    kind = style.digit_style
    if kind in ("flat", "low_contrast"):
        img.alpha_composite(layer)
        return
    big = img.size[0]
    alpha = np.asarray(layer.getchannel("A"), np.float32) / 255
    o = max(2, round(big / 110))
    edge_width = max(1, round(big / 160))
    # Inner edge band: the digit minus the digit shrunk by `edge_width`.
    shrunk = (
        np.asarray(
            layer.getchannel("A").filter(ImageFilter.MinFilter(2 * edge_width + 1)), np.float32
        )
        / 255
    )
    rim = np.clip(alpha - shrunk, 0, 1)
    if kind in ("emboss", "deboss"):
        light, dark = (-1, 1) if kind == "emboss" else (1, -1)
        lit = np.clip(shift(alpha, light * o, light * o) - alpha, 0, 1)
        shaded = np.clip(shift(alpha, dark * o, dark * o) - alpha, 0, 1)
        blend(img, (255, 255, 255), lit * float(rng.uniform(0.45, 0.8)))
        blend(img, (0, 0, 0), shaded * float(rng.uniform(0.35, 0.65)))
        # The body is the tile's own material, a touch lighter or darker.
        body = tuple(int(np.clip(c * float(rng.uniform(0.9, 1.1)), 0, 255)) for c in tile[:3])
        blend(img, body, alpha)
    elif kind == "engrave":
        blend(img, style.digit, rim * float(rng.uniform(0.7, 1.0)))
    elif kind == "engrave_shadow":
        inner = np.clip(alpha - shift(alpha, o, o), 0, 1)
        blend(img, tuple(int(c * 0.75) for c in tile[:3]), alpha * 0.5)
        blend(img, (0, 0, 0), inner * float(rng.uniform(0.5, 0.85)))
        blend(img, (255, 255, 255), np.clip(shift(alpha, o, o) - alpha, 0, 1) * 0.35)
    elif kind == "metal":
        rows = np.nonzero(alpha.max(axis=1) > 0.1)[0]
        top, bottom = (rows[0], rows[-1]) if len(rows) else (0, big - 1)
        t = np.clip((np.arange(big, dtype=np.float32) - top) / max(1, bottom - top), 0, 1)[
            :, None
        ] * np.ones((1, big), np.float32)
        tint = np.asarray(style.metal_tint, np.float32)
        gradient = tint * (1.25 - 0.7 * t[..., None])
        centre = float(rng.uniform(0.15, 0.85))
        gradient = gradient + 110 * np.exp(-(((t - centre) / 0.1) ** 2))[..., None]
        blend(img, gradient, alpha)
        blend(img, tuple(int(c * 0.4) for c in tint), rim * 0.8)
    else:
        raise ValueError(f"unknown digit style {kind}")


def make_base_texture(rng: np.random.Generator, style: BoardStyle, edge: int) -> Image.Image:
    """Return the texture of the board's frame and the floor of its empty slot.

    Flat colour, wood grain, brushed metal or plastic.
    """
    base = np.asarray(style.base, np.float32)
    if style.texture == "flat":
        return Image.new("RGB", (edge, edge), style.base)
    yy, xx = np.mgrid[0:edge, 0:edge].astype(np.float32)
    if style.texture == "wood":
        # Rings along one axis, wobbled by low-frequency noise, over fine fibres.
        wobble = (
            np.asarray(
                Image.fromarray(rng.random((6, 6)).astype(np.float32) * 255).resize(
                    (edge, edge), Image.Resampling.BICUBIC
                ),
                np.float32,
            )
            / 255
        )
        fibres = rng.normal(0, 1, size=(1, edge)).astype(np.float32) * np.ones(
            (edge, 1), np.float32
        )
        rings = np.sin((xx + 60 * wobble) * float(rng.uniform(0.05, 0.14)))
        pattern = 0.09 * rings + 0.04 * fibres
    elif style.texture == "brushed":
        streaks = rng.normal(0, 1, size=(edge, edge)).astype(np.float32)
        # Blur along x only: a long thin kernel makes the grain of brushed metal.
        kernel = np.ones(41, np.float32) / 41
        streaks = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, streaks)
        pattern = 0.5 * streaks / (streaks.std() + 1e-6) * 0.07
    else:  # plastic: a soft mottle and a faint sheen
        mottle = (
            np.asarray(
                Image.fromarray(rng.random((8, 8)).astype(np.float32) * 255).resize(
                    (edge, edge), Image.Resampling.BICUBIC
                ),
                np.float32,
            )
            / 255
        )
        pattern = 0.12 * (mottle - 0.5) + 0.05 * np.sin(xx / edge * math.pi)
    arr = base * (1 + pattern[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_flat_board(
    rng: np.random.Generator, style: BoardStyle, state: list[int]
) -> tuple[Image.Image, float]:
    """Render the board square (frame included).

    Returns the image and the frame margin fraction.
    """
    edge = FLAT_SIZE
    img = make_base_texture(rng, style, edge).convert("RGBA")
    inner = edge * (1 - 2 * style.margin)
    pitch = inner / style.size
    origin = edge * style.margin
    # Shade the base a little so the empty slot is not a perfectly flat patch.
    for index, number in enumerate(state):
        if number == 0:
            continue
        row, col = divmod(index, style.size)
        tile = draw_tile(rng, style, number, int(round(pitch)))
        img.alpha_composite(
            tile, (int(round(origin + col * pitch)), int(round(origin + row * pitch)))
        )
    return img.convert("RGB"), style.margin


def compute_homography_coefficients(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Coefficients for PIL's PERSPECTIVE transform: maps output (dst) coordinates to input (src)."""
    rows = []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    rhs = src.reshape(-1)
    return np.linalg.solve(np.array(rows, dtype=float), rhs.astype(float))


def generate_random_background(
    rng: np.random.Generator, width: int, height: int, avoid: tuple[int, int, int]
) -> Image.Image:
    """Return a table-like background with a gradient, stripes and clutter."""
    base = pick_colour_with_contrast(rng, avoid, 35)
    other = pick_colour_with_contrast(rng, avoid, 20)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = float(rng.uniform(0, 2 * math.pi))
    t = (xx * math.cos(angle) + yy * math.sin(angle)) / max(width, height)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    arr = (
        np.asarray(base, np.float32) * (1 - t[..., None])
        + np.asarray(other, np.float32) * t[..., None]
    )
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")
    # Wood-grain-like stripes on some backgrounds.
    if rng.random() < 0.4:
        period = int(rng.integers(8, 40))
        stripe = pick_colour_with_contrast(rng, base, 10)
        for row in range(0, height, period):
            draw.line(
                [(0, row), (width, row + int(rng.integers(-10, 10)))],
                fill=stripe + (70,),
                width=int(rng.integers(1, 4)),
            )
    # Clutter: random shapes of random colour, including a few rectangles that look like books.
    for _ in range(int(rng.integers(3, 12))):
        colour = generate_random_colour(rng) + (int(rng.integers(120, 255)),)
        x, y = float(rng.uniform(0, width)), float(rng.uniform(0, height))
        w, h = float(rng.uniform(20, width * 0.4)), float(rng.uniform(20, height * 0.4))
        kind = int(rng.integers(4))
        if kind == 0:
            draw.ellipse((x, y, x + w, y + h), fill=colour)
        elif kind == 1:
            draw.line((x, y, x + w, y + h), fill=colour, width=int(rng.integers(2, 14)))
        else:
            theta = float(rng.uniform(0, math.pi))
            pts = [
                (
                    x + math.cos(theta) * px - math.sin(theta) * py,
                    y + math.sin(theta) * px + math.cos(theta) * py,
                )
                for px, py in ((0, 0), (w, 0), (w, h), (0, h))
            ]
            draw.polygon(pts, fill=colour)
    return img


def generate_random_quad(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Board corners (TL, TR, BR, BL) in the photo: scale 30-90% of the short side, rotation, perspective."""
    short = min(width, height)
    for _ in range(1000):
        side = float(rng.uniform(0.30, 0.88)) * short
        theta = math.radians(float(rng.uniform(-20, 20)))
        cx = float(rng.uniform(0.3, 0.7)) * width
        cy = float(rng.uniform(0.3, 0.7)) * height
        half = side / 2
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        quad = corners @ rot.T
        # Perspective: pull the corners in or out independently (keystone and skew).
        strength = float(rng.choice([0.0, 0.03, 0.06, 0.10]))
        quad = quad + rng.uniform(-strength, strength, size=(4, 2)) * side
        quad = quad + np.array([cx, cy])
        inside = (quad[:, 0] >= 2).all() and (quad[:, 0] <= width - 2).all()
        inside = inside and (quad[:, 1] >= 2).all() and (quad[:, 1] <= height - 2).all()
        if inside and is_convex(quad):
            return np.asarray(quad, np.float64)
    raise RuntimeError("could not place a board in the frame")


def is_convex(quad: np.ndarray) -> bool:
    """Return whether the four-point polygon `quad` is convex."""
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        signs.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0)
    return all(signs) or not any(signs)


def degrade(
    rng: np.random.Generator, img: Image.Image, jpeg_quality: tuple[int, int] = (55, 92)
) -> Image.Image:
    """Lighting gradient, blur, noise, contrast and JPEG artefacts."""
    arr = np.asarray(img, np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = float(rng.uniform(0, 2 * math.pi))
    ramp = (xx * math.cos(angle) + yy * math.sin(angle)) / max(w, h)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
    low = float(rng.uniform(0.55, 1.0))
    arr = arr * (low + (1.15 - low) * ramp)[..., None]
    if rng.random() < 0.3:  # soft shadow band
        centre = float(rng.uniform(0.2, 0.8))
        band = np.exp(-(((ramp - centre) / float(rng.uniform(0.05, 0.2))) ** 2))
        arr = arr * (1 - float(rng.uniform(0.2, 0.5)) * band)[..., None]
    if rng.random() < 0.2:  # hard shadow edge: an object between the lamp and the board
        theta = float(rng.uniform(0, 2 * math.pi))
        distance = (xx - w / 2) * math.cos(theta) + (yy - h / 2) * math.sin(theta)
        offset = float(rng.uniform(-0.25, 0.25)) * max(w, h)
        softness = float(rng.uniform(1.5, 6))
        in_shadow = 1 / (1 + np.exp(-np.clip((distance - offset) / softness, -30, 30)))
        arr = arr * (1 - float(rng.uniform(0.3, 0.6)) * in_shadow)[..., None]
    arr = (arr - 128) * float(rng.uniform(0.7, 1.15)) + 128 + float(rng.uniform(-20, 20))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    sigma = float(rng.choice([0, 0.5, 1.0, 1.5, 2.2]))
    if sigma:
        out = out.filter(ImageFilter.GaussianBlur(sigma))
    noise = float(rng.choice([0, 3, 6, 10]))
    if noise:
        arr = np.asarray(out, np.float32) + rng.normal(0, noise, size=arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(rng.integers(jpeg_quality[0], jpeg_quality[1] + 1)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_barrel_distortion(img: Image.Image, k: float) -> Image.Image:
    """Radial lens distortion.

    Output pixel p samples the input at c + (p - c)(1 + k r²), r = 1 at the edge midpoints.

    Positive k is barrel (straight lines bow outwards, the image edges are squeezed). The input outside
    the frame is transparent for RGBA input and black otherwise. Done as a PIL mesh transform: one
    bilinear patch per 16 px, which follows the smooth radial map closely.
    """
    w, h = img.size
    step = 16
    cx, cy = w / 2, h / 2

    def compute_source_point(x: float, y: float) -> tuple[float, float]:
        r2 = ((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2
        f = 1 + k * r2
        return cx + (x - cx) * f, cy + (y - cy) * f

    mesh = []
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1, y1 = min(x0 + step, w), min(y0 + step, h)
            quad = (
                *compute_source_point(x0, y0),
                *compute_source_point(x0, y1),
                *compute_source_point(x1, y1),
                *compute_source_point(x1, y0),
            )
            mesh.append(((x0, y0, x1, y1), quad))
    return img.transform(img.size, Image.Transform.MESH, mesh, Image.Resampling.BILINEAR)


def apply_barrel_to_points(points: np.ndarray, k: float, width: int, height: int) -> np.ndarray:
    """Where input-image points land after draw_barrel_distortion.

    The inverse of its sampling map, found by fixed-point iteration.
    """
    centre = np.array([width / 2, height / 2])
    half = centre
    out = points.astype(float).copy()
    for _ in range(30):
        r2 = (((out - centre) / half) ** 2).sum(axis=1, keepdims=True)
        out = centre + (points - centre) / (1 + k * r2)
    return out


def make_finger_polygon(
    rng: np.random.Generator, margin_px: float, pitch: float, edge: float
) -> np.ndarray:
    """Return a thumb tip entering the board from outside one side, in flat-board pixels.

    It reaches at most 0.3 of a tile past the frame, so no tile is ever covered whole. Points beyond
    the board are kept; the caller maps them with the same transform as the board.
    """
    reach = float(rng.uniform(0.0, margin_px + 0.3 * pitch))
    width = float(rng.uniform(0.7, 1.3)) * pitch
    along = float(rng.uniform(0.15, 0.85)) * edge
    length = 0.6 * edge  # runs well out of the board, so only the tip is visible
    angle = float(rng.normal(0, 0.15))
    # Built pointing down (+y) into a board whose top edge is y = 0, then rotated to a random side.
    tip_y = reach - width / 2
    arc = [
        (math.cos(t) * width / 2, tip_y + math.sin(t) * width / 2)
        for t in np.linspace(0, math.pi, 12)
    ]
    pts = np.array([(width / 2, tip_y - length), *arc, (-width / 2, tip_y - length)])
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pts = (pts - np.array([0, tip_y])) @ rot.T + np.array([0, tip_y])
    pts[:, 0] += along
    for _ in range(int(rng.integers(4))):  # quarter turns: top, right, bottom, left
        pts = np.stack([edge - pts[:, 1], pts[:, 0]], axis=1)
    return pts


def draw_finger(rng: np.random.Generator, img: Image.Image, polygon: np.ndarray) -> Image.Image:
    """Paint a skin-coloured, softly shaded thumb over `img` inside `polygon`."""
    skin = np.array(
        [float(rng.uniform(190, 245)), float(rng.uniform(140, 200)), float(rng.uniform(110, 170))]
    )
    skin = skin * float(rng.uniform(0.45, 1.0))  # darker skin tones too
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(
        [tuple(p) for p in polygon], fill=tuple(int(c) for c in skin) + (255,)
    )
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.8, 2.5))))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def generate_random_state(rng: np.random.Generator, size: int, solvable: bool) -> list[int]:
    """Return a random board state (0 is the blank) that is solvable or not."""
    total = size * size
    state = [int(v) for v in rng.permutation(total)]
    if is_solvable(state, size) != solvable:
        # Swapping two tiles flips parity.
        idx = [i for i, v in enumerate(state) if v != 0][:2]
        state[idx[0]], state[idx[1]] = state[idx[1]], state[idx[0]]
    return state


def is_solvable(state: list[int], size: int) -> bool:
    """Return whether the sliding-puzzle state can reach the solved order."""
    tiles = [v for v in state if v != 0]
    inversions = sum(
        1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j]
    )
    # The blank's value is 0 here, so convert "blank row from top" to the goal's convention.
    if size % 2 == 1:
        return inversions % 2 == 0
    blank_row = state.index(0) // size
    return (inversions + blank_row) % 2 == (size - 1) % 2


FINGER_PROBABILITY = 0.12
BARREL_PROBABILITY = 0.3


def render_photo(
    rng: np.random.Generator,
    size: int,
    state: list[int],
    fonts: list[Path],
    width: int,
    height: int,
) -> tuple[Image.Image, np.ndarray, str]:
    """Render one labelled photo of a board."""
    style = generate_random_style(rng, size, fonts)
    flat, _ = render_flat_board(rng, style, state)
    background = generate_random_background(rng, width, height, style.base)
    quad = generate_random_quad(rng, width, height)
    src = np.array([[0, 0], [FLAT_SIZE, 0], [FLAT_SIZE, FLAT_SIZE], [0, FLAT_SIZE]], float)
    coeffs = compute_homography_coefficients(src, quad)
    warped = flat.transform(
        (width, height), Image.Transform.PERSPECTIVE, tuple(coeffs), Image.Resampling.BICUBIC
    )
    mask = Image.new("L", (FLAT_SIZE, FLAT_SIZE), 255).transform(
        (width, height), Image.Transform.PERSPECTIVE, tuple(coeffs), Image.Resampling.BILINEAR
    )
    finger = None
    if rng.random() < FINGER_PROBABILITY:
        # Flat-board pixels -> photo pixels is the inverse map of the one used to warp.
        forward = compute_homography_coefficients(quad, src)
        pitch = FLAT_SIZE * (1 - 2 * style.margin) / size
        polygon = make_finger_polygon(rng, style.margin * FLAT_SIZE, pitch, FLAT_SIZE)
        denom = forward[6] * polygon[:, 0] + forward[7] * polygon[:, 1] + 1
        finger = np.stack(
            [
                (forward[0] * polygon[:, 0] + forward[1] * polygon[:, 1] + forward[2]) / denom,
                (forward[3] * polygon[:, 0] + forward[4] * polygon[:, 1] + forward[5]) / denom,
            ],
            axis=1,
        )
    if rng.random() < BARREL_PROBABILITY:
        # Distort the board and its mask (the table behind it is clutter anyway)
        # and move the labelled corners with them.
        k = float(rng.uniform(0.02, 0.10))
        warped = draw_barrel_distortion(warped.convert("RGB"), k)
        mask = draw_barrel_distortion(mask, k)
        quad = apply_barrel_to_points(quad, k, width, height)
        if finger is not None:
            finger = apply_barrel_to_points(finger, k, width, height)
    shadow = mask.filter(ImageFilter.GaussianBlur(float(rng.uniform(4, 12))))
    shadow = shadow.transform(shadow.size, Image.Transform.AFFINE, (1, 0, -6, 0, 1, -8)).point(
        lambda v: int(v * 0.45)
    )
    background.paste(Image.new("RGB", background.size, (0, 0, 0)), (0, 0), shadow)
    background.paste(warped, (0, 0), mask)
    if finger is not None:
        background = draw_finger(rng, background, finger)
    return degrade(rng, background), quad, style.font_path.stem


def render_training_board(
    rng: np.random.Generator, size: int, fonts: list[Path], rectified: int = 400
) -> tuple[np.ndarray, list[int], float]:
    """Return a rectified-looking board for classifier training.

    Flat render, small corner jitter, degradation.

    Returns (grey image, state, true frame margin fraction).
    """
    state = [int(v) for v in rng.permutation(size * size)]
    style = generate_random_style(rng, size, fonts)
    flat, margin = render_flat_board(rng, style, state)
    # Rectification residue: the quad fit is off by a couple of percent at each corner.
    src = np.array([[0, 0], [FLAT_SIZE, 0], [FLAT_SIZE, FLAT_SIZE], [0, FLAT_SIZE]], float)
    jitter = rng.normal(0, 0.008 * FLAT_SIZE, size=(4, 2)).clip(-0.02 * FLAT_SIZE, 0.02 * FLAT_SIZE)
    jittered = src + jitter
    coeffs = compute_homography_coefficients(jittered, src * (rectified / FLAT_SIZE))
    img = flat.transform(
        (rectified, rectified), Image.Transform.PERSPECTIVE, tuple(coeffs), Image.Resampling.BICUBIC
    )
    if rng.random() < FINGER_PROBABILITY:
        pitch = FLAT_SIZE * (1 - 2 * style.margin) / size
        polygon = make_finger_polygon(rng, style.margin * FLAT_SIZE, pitch, FLAT_SIZE)
        # Same transform as the board: flat pixels -> rectified pixels (plus the corner jitter above).
        forward = compute_homography_coefficients(src * (rectified / FLAT_SIZE), jittered)
        denom = forward[6] * polygon[:, 0] + forward[7] * polygon[:, 1] + 1
        points = np.stack(
            [
                (forward[0] * polygon[:, 0] + forward[1] * polygon[:, 1] + forward[2]) / denom,
                (forward[3] * polygon[:, 0] + forward[4] * polygon[:, 1] + forward[5]) / denom,
            ],
            axis=1,
        )
        img = draw_finger(rng, img, points)
    if rng.random() < BARREL_PROBABILITY:
        img = draw_barrel_distortion(img, float(rng.uniform(0.02, 0.10)))
    # Colour cast (white balance) before the grey conversion, so the digit/tile luminance contrast varies.
    cast = rng.uniform(0.85, 1.15, size=3)
    img = Image.fromarray(np.clip(np.asarray(img, np.float32) * cast, 0, 255).astype(np.uint8))
    img = degrade(rng, img)
    return np.asarray(img.convert("L"), np.uint8), state, margin


CROP_PIXELS = 32
CROP_WINDOW = 0.8  # crop side as a fraction of the cell pitch; the TS pipeline uses the same value
CROP_SUBSAMPLES = 3  # per output pixel, per axis; the TS pipeline uses the same value
CROP_MIN_SPREAD = 0.08  # floor on the contrast normaliser, so a blank cell stays low-amplitude


def sample_crop(grey: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    """CROP_PIXELS² crop centred at (cx, cy).

    Each pixel is the mean of bilinear sub-samples, contrast-normalised.

    Mirrors src/vision/crop.ts; keep the two in step (a test compares them on a fixture).
    """
    offsets = (np.arange(CROP_PIXELS * CROP_SUBSAMPLES) + 0.5) / (
        CROP_PIXELS * CROP_SUBSAMPLES
    ) - 0.5
    xs = cx + offsets * side
    ys = cy + offsets * side
    h, w = grey.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    g = grey.astype(np.float32) / 255.0
    top = g[np.ix_(y0, x0)] * (1 - fx)[None, :] + g[np.ix_(y0, x0 + 1)] * fx[None, :]
    bottom = g[np.ix_(y0 + 1, x0)] * (1 - fx)[None, :] + g[np.ix_(y0 + 1, x0 + 1)] * fx[None, :]
    full = top * (1 - fy)[:, None] + bottom * fy[:, None]
    crop = full.reshape(CROP_PIXELS, CROP_SUBSAMPLES, CROP_PIXELS, CROP_SUBSAMPLES).mean(
        axis=(1, 3)
    )
    return normalise_crop(crop)


def augment_grey_window(rng: np.random.Generator, grey: np.ndarray) -> np.ndarray:
    """Crop-level photometric jitter applied to the grey pixels around one cell before it is sampled.

    Gamma and contrast (the crop's own normalisation removes only the linear part of these), and a
    specular highlight that saturates part of the cell. The classifier then sees digits that are
    washed out or half burnt out, whatever the board's own style.
    """
    g = grey.astype(np.float32) / 255
    if rng.random() < 0.5:
        g = g ** float(rng.uniform(0.6, 1.6))
    if rng.random() < 0.4:
        g = (g - 0.5) * float(rng.uniform(0.5, 1.3)) + 0.5 + float(rng.uniform(-0.1, 0.1))
    if rng.random() < 0.25:
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = float(rng.uniform(0.2, 0.8)) * w, float(rng.uniform(0.2, 0.8)) * h
        angle = float(rng.uniform(0, math.pi))
        u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        su = float(rng.uniform(0.1, 0.45)) * w
        sv = float(rng.uniform(0.05, 0.25)) * w
        g = g + float(rng.uniform(0.3, 1.0)) * np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)


def normalise_crop(crop: np.ndarray) -> np.ndarray:
    """Scale a crop to zero mean and unit spread, as the classifier expects."""
    normalised: np.ndarray = (crop - crop.mean()) / max(float(crop.std()), CROP_MIN_SPREAD)
    return normalised.astype(np.float32)


def compute_cell_centre(
    row: int, col: int, size: int, margin: float, edge: int
) -> tuple[float, float, float]:
    """Centre x, y and pitch of a cell in a rectified `edge`-pixel square whose frame margin is `margin`."""
    pitch = edge * (1 - 2 * margin) / size
    return edge * margin + (col + 0.5) * pitch, edge * margin + (row + 0.5) * pitch, pitch


def check_training_crops(path: Path, data: np.lib.npyio.NpzFile) -> None:
    """Refuses an extra crop set (train_classifier.py --extra) that could carry eval-split content.

    An .npz holds only pixels, so it cannot prove where they came from. The rule is therefore
    positive: the file must declare `fonts` (the font file stems its crops were rendered in) and
    `source`, every font must be a fonts/train font, and it must not live under data/eval*, where
    `npm run gen` writes the held-out photos. Crops cut from eval photos or eval fonts are refused.
    """
    in_data = (
        path.resolve().relative_to(ROOT / "data").parts
        if path.resolve().is_relative_to(ROOT / "data")
        else ()
    )
    if in_data and in_data[0].startswith("eval"):
        raise SystemExit(
            f"{path}: refusing crops from an eval path; eval data must never be trained on"
        )
    missing = {"x", "y", "fonts", "source"} - set(data.files)
    if missing:
        raise SystemExit(
            f"{path}: missing arrays {sorted(missing)}; extra crop sets must declare their fonts and source"
        )
    fonts = {str(f) for f in data["fonts"]}
    source = str(data["source"])
    train_names = {font.stem for font in TRAIN_FONTS}
    if "eval" in source.lower() or fonts & EVAL_FONT_NAMES:
        raise SystemExit(
            f"{path}: refusing crops derived from eval-split photos or fonts "
            f"({sorted(fonts & EVAL_FONT_NAMES) or source})"
        )
    unknown = fonts - train_names
    if unknown:
        raise SystemExit(f"{path}: fonts {sorted(unknown)} are not in fonts/train; refusing")
