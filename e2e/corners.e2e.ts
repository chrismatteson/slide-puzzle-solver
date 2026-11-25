// The two corrections of a read: "Adjust corners" (the board outline was wrong) and "Change empty slot"
// (a picture's empty cell was wrong). Like the other e2e tests, everything after load is offline.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";
import { type Page, expect, test } from "@playwright/test";
import { dragWithMouse, dragWithTouch, findHandleCentre, mapPhotoToScreen } from "./corner-drag.ts";
import {
  READ_TIMEOUT_MS,
  openApp,
  readImages,
  readNumbers,
  solveLabelledPuzzle,
  takeScreenshot,
  uploadOwnerPhoto,
} from "./helpers.ts";
import { parsePictureLabel } from "../scripts/labels.ts";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

const FIXTURES = `${import.meta.dirname}/../test/fixtures`;
const NUMBERED_PHOTO = `${FIXTURES}/photo-0000.jpg`;
const LOWCONF_PHOTO = `${FIXTURES}/photo-lowconf.jpg`;
const WHITE_FRAME_PHOTO = `${FIXTURES}/corrections/white-frame.jpg`;
const tables = await loadPatternTables();

const numberedLabel = JSON.parse(readFileSync(`${FIXTURES}/photo-0000.json`, "utf8")) as {
  size: 3;
  state: number[];
  corners: [number, number][];
};
const numberedJpeg = jpeg.decode(readFileSync(NUMBERED_PHOTO), { useTArray: true });
const pictureLabelPath = `${FIXTURES}/owner-picture/picture-4x4-01.json`;
const pictureLabel = parsePictureLabel(
  JSON.parse(readFileSync(pictureLabelPath, "utf8")),
  pictureLabelPath,
);

async function uploadNumbered(page: Page): Promise<void> {
  await page.locator("#photo-file").setInputFiles(NUMBERED_PHOTO);
  await expect(page.locator("#review .grid")).toBeVisible({ timeout: READ_TIMEOUT_MS });
}

/** A point inside the labelled quad of the numbered fixture, `u` of the way across and `v` down (bilinear). */
function inLabelledQuad(u: number, v: number): [number, number] {
  const [tl, tr, br, bl] = numberedLabel.corners as [
    [number, number],
    [number, number],
    [number, number],
    [number, number],
  ];
  const at = (k: 0 | 1): number =>
    (1 - v) * ((1 - u) * tl[k] + u * tr[k]) + v * ((1 - u) * bl[k] + u * br[k]);
  return [at(0), at(1)];
}

/** Drags each dot onto the matching corner, alternating mouse and touch. */
async function dragDotsTo(page: Page, corners: [number, number][]): Promise<void> {
  for (const [index, corner] of corners.entries()) {
    const drag = index % 2 === 0 ? dragWithMouse : dragWithTouch;
    await drag(page, index, await mapPhotoToScreen(page, numberedJpeg, corner));
  }
}

/** A dot can only be dragged inside the view shown, so a quad to be moved far is moved on the whole photo. */
async function showWholePhoto(page: Page): Promise<void> {
  const zoom = page.locator("#editor-zoom");
  if ((await zoom.textContent()) === "Show whole photo") await zoom.click();
  await expect(zoom).toHaveText("Zoom to board");
}

test.describe("Adjust corners", () => {
  test.use({ hasTouch: true });

  test("corners placed by hand are what is read: a wrong area reads wrong, the labelled corners read the labelled board and solve", async ({
    page,
  }) => {
    const errors = await openApp(page);
    await uploadNumbered(page);
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Adjust corners");
    await expect(page.locator("#corner-hint")).toHaveText(
      "Drag the four dots onto the outer corners of the tiles (not the frame). Grid: auto (faint lines show the 3×3 of the current reading).",
    );
    await expect(page.locator("#corner-size-auto")).toBeChecked();
    // Phone-sized: the editor never scrolls sideways, and a dot's hit area is at least 44 px.
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
    const hit = await page.locator("#corner-0").boundingBox();
    expect(hit?.width).toBeGreaterThanOrEqual(44);
    expect(hit?.height).toBeGreaterThanOrEqual(44);

    // First a deliberately wrong area, only the top-left 2x2 tiles of the labelled quad. If the corners
    // were ignored the board would be found anyway and read right, and this test would fail.
    await dragDotsTo(page, [
      inLabelledQuad(0, 0),
      inLabelledQuad(2 / 3, 0),
      inLabelledQuad(2 / 3, 2 / 3),
      inLabelledQuad(0, 2 / 3),
    ]);
    await page.locator("#read-corners").click();
    await expect(page.locator("#adjust-corners")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    // Either rejected or a board that is not the labelled one; both show a grid, or none at all.
    if ((await page.locator("#review .grid").count()) > 0) {
      expect(await readNumbers(page, "#review .grid")).not.toEqual(numberedLabel.state);
    }
    await takeScreenshot(page, "corners-wrong-area");

    // Then the labelled corners: two dots by mouse, two by touch.
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Adjust corners");
    // It opens zoomed on the wrong area just read, which the labelled corners lie outside of.
    await showWholePhoto(page);
    await dragDotsTo(page, numberedLabel.corners);
    for (let i = 0; i < 4; i++) {
      const at = await findHandleCentre(page, i);
      const want = await mapPhotoToScreen(
        page,
        numberedJpeg,
        numberedLabel.corners[i] as [number, number],
      );
      expect(Math.hypot(at.x - want.x, at.y - want.y)).toBeLessThan(3);
    }
    await takeScreenshot(page, "corners-editor");

    await page.locator("#read-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Detected board", {
      timeout: READ_TIMEOUT_MS,
    });
    expect(await readNumbers(page, "#review .grid")).toEqual(numberedLabel.state);
    await expect(page.locator("#check")).toContainText("consistent");
    await page.getByRole("button", { name: "Solve", exact: true }).click();
    const expected = solve({ size: 3, cells: numberedLabel.state }, { tables });
    if (expected.status !== "solved") throw new Error("fixture is not solvable");
    await expect(page.locator("#solution-heading")).toHaveText(
      `Solution: ${String(expected.moves.length)} moves`,
    );
    await takeScreenshot(page, "corners-solved");
    expect(errors).toEqual([]);
  });

  test("a numbered review re-read with adjusted corners stays a numbered puzzle", async ({
    page,
  }) => {
    await openApp(page);
    await page.locator("#photo-file").setInputFiles(LOWCONF_PHOTO);
    await expect(page.locator("#review-heading")).toHaveText("Detected board", {
      timeout: READ_TIMEOUT_MS,
    });
    await page.locator("#adjust-corners").click();
    for (const index of [0, 2]) {
      await page.locator(`#corner-${String(index)}`).focus();
      await page.keyboard.press("Shift+ArrowRight");
      await page.keyboard.press("Shift+ArrowRight");
    }
    await page.locator("#read-corners").click();
    await expect(page.locator("#adjust-corners")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    await expect(page.locator("#review-heading")).toHaveText("Detected board");
    await expect(page.locator("#review .grid")).toBeVisible();
  });

  test("Treat as picture puzzle keeps the corners the user placed", async ({ page }) => {
    await openApp(page);
    await uploadNumbered(page);
    await page.locator("#adjust-corners").click();
    // Only the top-left 2x2 tiles, so the board found by itself would put the bottom-right dot elsewhere.
    const wrongCorner = inLabelledQuad(2 / 3, 2 / 3);
    await dragDotsTo(page, [
      inLabelledQuad(0, 0),
      inLabelledQuad(2 / 3, 0),
      wrongCorner,
      inLabelledQuad(0, 2 / 3),
    ]);
    await page.locator("#read-corners").click();
    await expect(page.locator("#use-picture")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    await page.locator("#use-picture").click();
    await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
      timeout: READ_TIMEOUT_MS,
    });
    await page.locator("#adjust-corners").click();
    const at = await findHandleCentre(page, 2);
    const want = await mapPhotoToScreen(page, numberedJpeg, wrongCorner);
    // The picture reader may snap a corner a little (at most 4% of the side), never to the real board's.
    expect(Math.hypot(at.x - want.x, at.y - want.y)).toBeLessThan(25);
  });

  test("the loupe shows while a dot is dragged, above it, and is gone afterwards", async ({
    page,
  }) => {
    await openApp(page);
    await uploadNumbered(page);
    await page.locator("#adjust-corners").click();
    const loupe = page.locator("#corner-loupe");
    await expect(loupe).toBeHidden();
    const from = await findHandleCentre(page, 2);
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(from.x - 10, from.y - 10, { steps: 3 });
    await expect(loupe).toBeVisible();
    const box = await loupe.boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual(110);
    expect(box?.height).toBeGreaterThanOrEqual(110);
    // Above the finger, not under it.
    expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThan(from.y - 10);
    await takeScreenshot(page, "corners-loupe");
    await page.mouse.up();
    await expect(loupe).toBeHidden();

    // And by touch.
    const session = await page.context().newCDPSession(page);
    const start = await findHandleCentre(page, 2);
    await session.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [{ x: start.x, y: start.y }],
    });
    await session.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [{ x: start.x - 5, y: start.y - 5 }],
    });
    await expect(loupe).toBeVisible();
    await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await expect(loupe).toBeHidden();
    await session.detach();
  });

  test("stacked dots can be separated, and Read says why it is disabled when the corners fold", async ({
    page,
  }) => {
    await openApp(page);
    await uploadNumbered(page);
    await page.locator("#adjust-corners").click();
    const read = page.locator("#read-corners");
    await expect(read).toBeEnabled();
    await expect(page.locator("#corner-problem")).toBeHidden();
    const one = await findHandleCentre(page, 1);
    const three = await findHandleCentre(page, 3);
    // Dot 1 right on top of dot 3: the quad folds over itself.
    await dragWithMouse(page, 1, three);
    await expect(read).toBeDisabled();
    await expect(read).toHaveAttribute("aria-describedby", "corner-problem");
    await expect(page.locator("#corner-problem")).toHaveText(
      "The corners cross over or are too close together. Move them onto the four corners of the tiles.",
    );
    // Grabbing the stack takes the one touched last (dot 1) and leaves dot 3 where it was.
    await page.mouse.move(three.x, three.y);
    await page.mouse.down();
    await page.mouse.move(one.x, one.y, { steps: 5 });
    await page.mouse.up();
    const movedBack = await findHandleCentre(page, 1);
    expect(Math.hypot(movedBack.x - one.x, movedBack.y - one.y)).toBeLessThan(3);
    const stayed = await findHandleCentre(page, 3);
    expect(Math.hypot(stayed.x - three.x, stayed.y - three.y)).toBeLessThan(3);
    await expect(read).toBeEnabled();
    await expect(read).not.toHaveAttribute("aria-describedby", /.+/);
    await expect(page.locator("#corner-problem")).toBeHidden();
  });

  test("arrow keys move a dot by half a percent, shift by three, and stop at the photo's edge", async ({
    page,
  }) => {
    await openApp(page);
    await uploadNumbered(page);
    await page.locator("#adjust-corners").click();
    const handle = page.locator("#corner-0");
    await expect(handle).toBeFocused();
    await expect(handle).toHaveAccessibleName("Top-left corner: drag, or use the arrow keys");
    const position = async (): Promise<{ left: number; top: number }> =>
      handle.evaluate((element) => ({
        left: parseFloat((element as HTMLElement).style.left),
        top: parseFloat((element as HTMLElement).style.top),
      }));
    const start = await position();
    await page.keyboard.press("ArrowRight");
    expect((await position()).left).toBeCloseTo(start.left + 0.5, 3);
    await page.keyboard.press("Shift+ArrowDown");
    expect((await position()).top).toBeCloseTo(start.top + 3, 3);
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Shift+ArrowUp");
    expect(await position()).toEqual({
      left: expect.closeTo(start.left, 3),
      top: expect.closeTo(start.top, 3),
    });
    for (let i = 0; i < 40; i++) await page.keyboard.press("Shift+ArrowLeft");
    expect((await position()).left).toBe(0);
    await expect(page.locator("#corner-1")).toHaveAccessibleName(
      "Top-right corner: drag, or use the arrow keys",
    );
    await expect(page.locator("#corner-2")).toHaveAccessibleName(
      "Bottom-right corner: drag, or use the arrow keys",
    );
    await expect(page.locator("#corner-3")).toHaveAccessibleName(
      "Bottom-left corner: drag, or use the arrow keys",
    );
  });

  test("Cancel restores the review as it was", async ({ page }) => {
    await openApp(page);
    await uploadNumbered(page);
    const before = await readNumbers(page, "#review .grid");
    await page.locator("#adjust-corners").click();
    await page.locator("#corner-size-4").check();
    await dragWithMouse(page, 2, await mapPhotoToScreen(page, numberedJpeg, [100, 100]));
    await page.locator("#cancel-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Detected board");
    await expect(page.locator(".corner-editor")).toHaveCount(0);
    expect(await readNumbers(page, "#review .grid")).toEqual(before);
    await expect(page.locator("#adjust-corners")).toBeFocused();
    // Reopening starts from the detected board again, not from the abandoned edit.
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#corner-size-auto")).toBeChecked();
  });

  test("an outline past the photo's edge opens the editor with every dot inside the photo, and Read gives a review", async ({
    page,
  }) => {
    const errors = await openApp(page);
    // Lossless, so Chromium reads the same pixels as Node, where the numbered reader puts one corner of
    // this photo's outline below the photo (test/user-corners.test.ts).
    const decoded = jpeg.decode(readFileSync(WHITE_FRAME_PHOTO), { useTArray: true });
    const png = new PNG({ width: decoded.width, height: decoded.height });
    png.data.set(decoded.data);
    await page.locator("#photo-file").setInputFiles({
      name: "white-frame.png",
      mimeType: "image/png",
      buffer: PNG.sync.write(png),
    });
    await expect(page.locator("#adjust-corners")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    if ((await page.locator("#use-numbers").count()) > 0) {
      await page.locator("#use-numbers").click();
      await expect(page.locator("#use-picture")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    }
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Adjust corners");
    const box = await page.locator(".corner-wrap img").boundingBox();
    if (!box) throw new Error("the photo is not shown");
    for (let i = 0; i < 4; i++) {
      const at = await findHandleCentre(page, i);
      expect(at.x).toBeGreaterThanOrEqual(box.x - 1);
      expect(at.x).toBeLessThanOrEqual(box.x + box.width + 1);
      expect(at.y).toBeGreaterThanOrEqual(box.y - 1);
      expect(at.y).toBeLessThanOrEqual(box.y + box.height + 1);
    }
    await takeScreenshot(page, "corners-outline-past-edge");
    await expect(page.locator("#read-corners")).toBeEnabled();
    await page.locator("#read-corners").click();
    await expect(page.locator("#adjust-corners")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    await expect(page.locator("#review-heading")).not.toHaveText("Adjust corners");
    await expect(page.locator("#notice")).toBeHidden();
    expect(errors).toEqual([]);
  });

  test("the no-board screen offers it, starting from a rectangle inside the photo", async ({
    page,
  }) => {
    await openApp(page);
    const png = new PNG({ width: 320, height: 240 });
    png.data.fill(128);
    await page.locator("#photo-file").setInputFiles({
      name: "grey.png",
      mimeType: "image/png",
      buffer: PNG.sync.write(png),
    });
    await expect(page.locator("#review-heading")).toHaveText("No board found", {
      timeout: READ_TIMEOUT_MS,
    });
    await expect(page.locator("#review .notice")).toContainText("No board");
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#corner-size-auto")).toBeChecked();
    const box = await page.locator(".corner-wrap img").boundingBox();
    const first = await findHandleCentre(page, 0);
    expect(first.x - (box?.x ?? 0)).toBeCloseTo(0.15 * (box?.width ?? 0), 0);
    expect(first.y - (box?.y ?? 0)).toBeCloseTo(0.15 * (box?.height ?? 0), 0);
    await takeScreenshot(page, "corners-no-board");
  });
});

test.describe("Change empty slot", () => {
  test("a wrong empty cell changes the arrangement; the labelled one brings back the labelled puzzle", async ({
    page,
  }) => {
    const errors = await openApp(page);
    await uploadOwnerPhoto(page);
    const expected = solveLabelledPuzzle(pictureLabel, tables);
    const solvedGrid = "#review .grid[aria-label='Solved picture']";
    const original = await readImages(page, solvedGrid);
    await page.getByRole("button", { name: "Solve", exact: true }).click();
    await expect(page.locator("#solution-heading")).toBeVisible();

    await page.locator("#change-blank").click();
    await expect(page.locator("#blank-hint")).toHaveText(
      "Tap the cell that is empty in your puzzle.",
    );
    await expect(page.locator("#review .blank-chooser .cell")).toHaveCount(16);
    await expect(page.locator(`#blank-cell-${String(pictureLabel.blank)}`)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await takeScreenshot(page, "blank-chooser");

    // Tapping the current empty cell closes the chooser and changes nothing.
    await page.locator(`#blank-cell-${String(pictureLabel.blank)}`).click();
    await expect(page.locator(".blank-chooser")).toHaveCount(0);
    expect(await readImages(page, solvedGrid)).toEqual(original);

    const wrong = pictureLabel.blank === 5 ? 6 : 5;
    await page.locator("#change-blank").click();
    await page.locator(`#blank-cell-${String(wrong)}`).click();
    await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
      timeout: READ_TIMEOUT_MS,
    });
    await expect(page.locator("#solution")).toBeHidden();
    await expect(page.locator(".blank-chooser")).toHaveCount(0);
    await expect(page.locator("#check")).toBeVisible();
    expect(await readImages(page, solvedGrid)).not.toEqual(original);
    await takeScreenshot(page, "blank-changed");

    await page.locator("#change-blank").click();
    await page.locator(`#blank-cell-${String(pictureLabel.blank)}`).click();
    await expect(page.locator("#change-blank")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    await expect(page.locator(".blank-chooser")).toHaveCount(0);
    expect(await readNumbers(page, solvedGrid)).toEqual(expected.solvedNumbers);
    await expect(page.locator("#check")).toContainText(/can be solved|look the same/);
    await page.getByRole("button", { name: "Solve", exact: true }).click();
    await expect(page.locator("#solution-heading")).toHaveText(
      `Solution: ${String(expected.moves)} moves`,
    );
    expect(expected.moves).toBe(24);
    expect(errors).toEqual([]);
  });

  test("Cancel closes the chooser without a change, and the cells are keyboard buttons", async ({
    page,
  }) => {
    await openApp(page);
    await uploadOwnerPhoto(page);
    const solvedGrid = "#review .grid[aria-label='Solved picture']";
    const before = await readImages(page, solvedGrid);
    await page.locator("#change-blank").click();
    // Focus moves to the cell marked empty now, and the chooser is right under the button, near the top.
    await expect(page.locator(".blank-chooser .cell[aria-pressed=true]")).toBeFocused();
    await expect(page.locator("#blank-cell-0")).not.toBeFocused();
    const button = await page.locator("#change-blank").boundingBox();
    const chooser = await page.locator(".blank-chooser").boundingBox();
    expect(chooser?.y).toBeGreaterThan(button?.y ?? Infinity);
    expect((chooser?.y ?? 0) - (button?.y ?? 0)).toBeLessThan(150);
    const marked = Number(
      (await page.locator(".blank-chooser .cell[aria-pressed=true]").getAttribute("id"))?.replace(
        "blank-cell-",
        "",
      ),
    );
    await page.keyboard.press("Tab");
    await expect(page.locator(`#blank-cell-${String(marked + 1)}`)).toBeFocused();
    await page.locator("#cancel-blank").click();
    await expect(page.locator(".blank-chooser")).toHaveCount(0);
    await expect(page.locator("#change-blank")).toBeFocused();
    expect(await readImages(page, solvedGrid)).toEqual(before);
  });

  test("a picture photo's corners can be adjusted and read again as a picture", async ({
    page,
  }) => {
    await openApp(page);
    await uploadOwnerPhoto(page);
    await page.locator("#adjust-corners").click();
    await expect(page.locator("#corner-size-auto")).toBeChecked();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
    await page.locator("#read-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
      timeout: READ_TIMEOUT_MS,
    });
    await expect(page.locator("#change-blank")).toBeVisible();
  });

  test("adjusting the corners afterwards keeps the empty slot you chose", async ({ page }) => {
    await openApp(page);
    await uploadOwnerPhoto(page);
    const wrong = pictureLabel.blank === 5 ? 6 : 5;
    await page.locator("#change-blank").click();
    await page.locator(`#blank-cell-${String(wrong)}`).click();
    await expect(page.locator("#change-blank")).toBeEnabled({ timeout: READ_TIMEOUT_MS });
    await page.locator("#adjust-corners").click();
    await page.locator("#read-corners").click();
    await expect(page.locator("#picture-note")).toHaveText("Kept the empty slot you chose.", {
      timeout: READ_TIMEOUT_MS,
    });
    await page.locator("#change-blank").click();
    await expect(page.locator(`#blank-cell-${String(wrong)}`)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(page.locator(`#blank-cell-${String(pictureLabel.blank)}`)).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  test("a changed grid size drops the chosen empty slot and says so", async ({ page }) => {
    await openApp(page);
    await uploadOwnerPhoto(page);
    const wrong = pictureLabel.blank === 5 ? 6 : 5;
    await page.locator("#change-blank").click();
    await page.locator(`#blank-cell-${String(wrong)}`).click();
    await expect(page.locator("#change-blank")).toBeEnabled({ timeout: READ_TIMEOUT_MS });
    await page.locator("#adjust-corners").click();
    await page.locator("#corner-size-3").check();
    await page.locator("#read-corners").click();
    await expect(page.locator("#picture-note")).toHaveText(
      "The grid size changed, so the empty slot you chose was dropped.",
      { timeout: READ_TIMEOUT_MS },
    );
  });
});
