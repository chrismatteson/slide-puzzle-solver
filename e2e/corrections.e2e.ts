// The two failure photos of test/fixtures/corrections, corrected the way a person would on a phone:
// white-frame.jpg by dragging the corner dots (zooming as needed), angled-blank.jpg by tapping the
// empty cell. Like the other e2e tests, everything after the app has loaded is offline.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import { type Page, expect, test } from "@playwright/test";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { type PictureLabel, parsePictureLabel } from "../scripts/labels.ts";
import { dragWithTouch, findHandleCentre, mapPhotoToScreen } from "./corner-drag.ts";
import { openApp, readNumbers, solveLabelledPuzzle, takeScreenshot } from "./helpers.ts";

const FIXTURES = `${import.meta.dirname}/../test/fixtures/corrections`;
const tables = await loadPatternTables();

// Same margin as the other e2e files: a read takes 2-4 s, more on a loaded machine.
const READ_TIMEOUT_MS = 30_000;

function loadCorrectionPhoto(name: string): {
  file: string;
  width: number;
  height: number;
  label: PictureLabel;
  corners: [number, number][];
  solvedNumbers: number[];
  board: number[];
  moves: number;
} {
  const file = `${FIXTURES}/${name}.jpg`;
  const label = parsePictureLabel(
    JSON.parse(readFileSync(`${FIXTURES}/${name}.json`, "utf8")),
    name,
  );
  const { width, height } = jpeg.decode(readFileSync(file), { useTArray: true });
  const corners = label.tilesQuad.map((p): [number, number] => [p.x, p.y]);
  return { file, width, height, label, corners, ...solveLabelledPuzzle(label, tables) };
}

const solvedGrid = "#review .grid[aria-label='Solved picture']";
const photoGrid = "#review .grid[aria-label^='Your puzzle as photographed']";

async function upload(page: Page, file: string): Promise<void> {
  await page.locator("#photo-file").setInputFiles(file);
  await expect(page.locator("#adjust-corners")).toBeVisible({ timeout: READ_TIMEOUT_MS });
}

/** Whether a screen point is inside the editor's photo frame, so a dot can be dragged there. */
async function isShown(page: Page, at: { x: number; y: number }): Promise<boolean> {
  const box = await page.locator(".corner-wrap").boundingBox();
  if (!box) throw new Error("the editor is not shown");
  return at.x >= box.x && at.x <= box.x + box.width && at.y >= box.y && at.y <= box.y + box.height;
}

/**
 * Puts the four dots on `corners` by touch: first roughly, on the whole photo, wherever the board is,
 * then zoomed on the board for the last pixels.
 */
async function placeDots(
  page: Page,
  photo: { width: number; height: number },
  corners: [number, number][],
): Promise<void> {
  const zoom = page.locator("#editor-zoom");
  if ((await zoom.textContent()) === "Show whole photo") await zoom.click();
  await expect(zoom).toHaveText("Zoom to board");
  for (const [index, corner] of corners.entries()) {
    await dragWithTouch(page, index, await mapPhotoToScreen(page, photo, corner));
  }
  await zoom.click();
  await expect(zoom).toHaveText("Show whole photo");
  for (const [index, corner] of corners.entries()) {
    const target = await mapPhotoToScreen(page, photo, corner);
    expect(await isShown(page, target)).toBe(true);
    await dragWithTouch(page, index, target);
    const at = await findHandleCentre(page, index);
    // The zoomed view is why this is tighter than the whole photo allows.
    expect(Math.hypot(at.x - target.x, at.y - target.y)).toBeLessThan(3);
  }
}

test.describe("the failure photos, corrected by hand", () => {
  test.use({ hasTouch: true });

  test("white-frame: the dots on the labelled tile corners read the labelled 4x4 board and solve it", async ({
    page,
  }) => {
    const puzzle = loadCorrectionPhoto("white-frame");
    const errors = await openApp(page);
    await upload(page, puzzle.file);
    // The automatic reading takes the wrong area, whatever it shows.
    if ((await page.locator(photoGrid).count()) > 0) {
      expect(await readNumbers(page, photoGrid)).not.toEqual(puzzle.board);
    }
    await takeScreenshot(page, "corrections-white-frame-auto");

    await page.locator("#adjust-corners").click();
    await expect(page.locator("#corner-size-auto")).toBeChecked();
    await placeDots(page, puzzle, puzzle.corners);
    await takeScreenshot(page, "corrections-white-frame-editor");
    await page.locator("#read-corners").click();
    await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
      timeout: READ_TIMEOUT_MS,
    });
    expect(await readNumbers(page, solvedGrid)).toEqual(puzzle.solvedNumbers);
    expect(await readNumbers(page, photoGrid)).toEqual(puzzle.board);
    await expect(page.locator("#check")).toContainText("can be solved");
    await takeScreenshot(page, "corrections-white-frame-read");

    await page.getByRole("button", { name: "Solve", exact: true }).click();
    await expect(page.locator("#solution-heading")).toHaveText(
      `Solution: ${String(puzzle.moves)} moves`,
    );
    await takeScreenshot(page, "corrections-white-frame-solved");
    expect(errors).toEqual([]);
  });

  test("angled-blank: tapping the labelled empty cell gives the labelled 3x3 board and solves it", async ({
    page,
  }) => {
    const puzzle = loadCorrectionPhoto("angled-blank");
    const errors = await openApp(page);
    await upload(page, puzzle.file);
    await expect(page.locator("#change-blank")).toBeVisible();
    // The reader picked another cell as the empty one.
    expect(await readNumbers(page, photoGrid)).not.toEqual(puzzle.board);
    await takeScreenshot(page, "corrections-angled-blank-auto");

    await page.locator("#change-blank").click();
    await page.locator(`#blank-cell-${String(puzzle.label.blank)}`).click();
    await expect(page.locator("#change-blank")).toBeEnabled({ timeout: READ_TIMEOUT_MS });
    await expect(async () => {
      expect(await readNumbers(page, photoGrid)).toEqual(puzzle.board);
    }).toPass({ timeout: READ_TIMEOUT_MS });
    expect(await readNumbers(page, solvedGrid)).toEqual(puzzle.solvedNumbers);
    await expect(page.locator("#check")).toContainText("can be solved");
    await takeScreenshot(page, "corrections-angled-blank-read");

    await page.getByRole("button", { name: "Solve", exact: true }).click();
    await expect(page.locator("#solution-heading")).toHaveText(
      `Solution: ${String(puzzle.moves)} moves`,
    );
    await takeScreenshot(page, "corrections-angled-blank-solved");
    expect(errors).toEqual([]);
  });
});
