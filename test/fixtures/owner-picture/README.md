# Owner photos of picture-tile puzzles

- `picture-4x4-01.jpg`: the owner's own photo of their puzzle (2026-11-22). It is a 4x4 sliding puzzle whose 15 tiles carry pieces of one printed picture, with no numbers. The empty slot is in the top row, third column. The photo is slightly blurred, taken at an angle, with glare on the right edge.

The owner approved using this photo as a test fixture in this repository. Test data only: it does not go into the app bundle.

`picture-4x4-01.json` is its hand label: the empty slot's photo cell, the solved arrangement (which photo
cell belongs at each solved position) and the tile area's corners. Numbering the tiles 1-15 by their solved
position, the solved puzzle and the photo are:

```
solved            photo
 .  1  2  3        9  4  .  2
 4  5  6  7        1  6 11  3
 8  9 10 11        8  5 10  7
12 13 14 15       12 13 14 15
```

How it was derived: the trash can with its lettering fills the whole bottom row, and the round white lid
runs to the top edge of tile 4 with no tile showing its top-left, so the empty slot belongs top-left. Tile
seams fix the rest: tile 4's right edge and tile 5's left edge are both white lid; tile 9's bottom carries
the worm's tail that continues on tile 13; tile 3 has a fur tip continuing tile 2's hair. Assembled, this
shows one Oscar (two eyes, one open mouth with tongue, lid behind him, worm in front). Tiles 7 and 11 are
told apart by their seams with 3 and 15 and by parity: a board scrambled by sliding must be reachable from
solved, and the other way round it is not. The optimal solver needs 24 moves from the photo.

`picture-4x4-01.chromium-960.png` is the same photo as the app's browser path decodes it: Chromium's
`createImageBitmap` of the JPEG drawn on a canvas scaled to fit 1280 px (960x1280, as src/app/photo.ts
does), read back with `getImageData` and saved lossless (RGB PNG, 1.7 MB). It differs from the jpeg-js decode
in colour rounding and resampling, which was enough to flip assembly before the edge cost compared
z-normalised profiles (test/picture-chromium.test.ts). Made with Playwright's Chromium 159: load the JPEG
into a blob, `createImageBitmap`, draw into a 960x1280 canvas, `getImageData`, write the RGBA with pngjs.
