import { readFileSync } from "node:fs";
import { type Page, expect, test } from "@playwright/test";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

const FIXTURES = `${import.meta.dirname}/../test/fixtures`;
const OWNER_PHOTO = `${FIXTURES}/owner-picture/picture-4x4-01.jpg`;
const SHOTS = `${import.meta.dirname}/../e2e-artifacts`;
const tables = await loadPatternTables();

// Picture photos are read twice (numbered reader, then picture reader) and assembled: 2-4 s, more on a
// loaded machine. Same margin as app.e2e.ts.
const READ_TIMEOUT_MS = 30_000;

interface PictureLabel {
  size: 4;
  blank: number;
  /** solved[p]: photo cell of the tile that belongs at solved position p; -1 at the empty slot's home. */
  solved: number[];
}

const label = JSON.parse(
  readFileSync(`${FIXTURES}/owner-picture/picture-4x4-01.json`, "utf8"),
) as PictureLabel;

/** Tile number at each solved position (0 at the home), and the photo's board, from the label. */
function computeLabelledPuzzle(): { solvedNumbers: number[]; board: number[]; moves: number } {
  const home = label.solved.indexOf(-1);
  const solvedNumbers = label.solved.map((_, p) => (p === home ? 0 : p < home ? p + 1 : p));
  const board = new Array<number>(label.solved.length).fill(0);
  label.solved.forEach((cell, p) => {
    if (cell !== -1) board[cell] = solvedNumbers[p] as number;
  });
  const result = solve({ size: label.size, blankHome: home, cells: board }, { tables });
  if (result.status !== "solved") throw new Error("the labelled puzzle is not solvable");
  return { solvedNumbers, board, moves: result.moves.length };
}
const expected = computeLabelledPuzzle();

async function openApp(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("body[data-ready=true]").waitFor();
  await page.waitForLoadState("networkidle");
  await page.context().setOffline(true);
  return errors;
}

async function uploadOwnerPhoto(page: Page): Promise<void> {
  await page.locator("#photo-file").setInputFiles(OWNER_PHOTO);
  await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
}

/** Tile numbers of a grid, row-major; the empty slot is 0. */
async function readNumbers(page: Page, selector: string): Promise<number[]> {
  const texts = await page.locator(`${selector} .cell`).allTextContents();
  return texts.map((text) => (text === "" ? 0 : Number(text)));
}

/** What is drawn in each cell of a grid ("" for the empty slot): the numbers are fixed labels, the images move. */
async function readImages(page: Page, selector: string): Promise<string[]> {
  return page
    .locator(`${selector} .cell`)
    .evaluateAll((cells) => cells.map((cell) => cell.querySelector("canvas")?.toDataURL() ?? ""));
}

async function takeScreenshot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true });
}

const solvedGrid = "#review .grid[aria-label='Solved picture']";
const photoGrid = "#review .grid[aria-label^='Your puzzle as photographed']";

test("the owner's picture photo is shown in picture mode, solved and stepped to the end", async ({
  page,
}) => {
  const errors = await openApp(page);
  await uploadOwnerPhoto(page);

  const solvedNumbers = await readNumbers(page, solvedGrid);
  expect(solvedNumbers.filter((n) => n === 0)).toHaveLength(1);
  await expect(page.locator("#check")).toContainText("can be solved");
  await expect(page.locator(`${solvedGrid} .tile-image`)).toHaveCount(15);
  await expect(page.locator(`${photoGrid} .tile-image`)).toHaveCount(15);
  // Tiles are numbered by solved position, so the photo's board holds each number once.
  expect([...(await readNumbers(page, photoGrid))].sort((a, b) => a - b)).toEqual(
    [...solvedNumbers].sort((a, b) => a - b),
  );

  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toContainText("moves");
  const moves = await page.locator("#solution .move-list li").count();
  expect(moves).toBeGreaterThan(0);
  await expect(page.locator("#solution-heading")).toHaveText(`Solution: ${String(moves)} moves`);
  await expect(page.locator("#solution .cell.moving")).toHaveCount(1);
  await expect(page.locator("#solution .tile-image")).toHaveCount(15);
  // Moves name tiles by their number in the solved picture.
  await expect(page.locator("#move-label")).toContainText(/Move \d+ (up|down|left|right)/);

  // Step through every move, as a person would.
  for (let i = 0; i < moves; i++) await page.locator("#next").click();
  await expect(page.locator("#move-label")).toContainText("Solved.");
  expect(await readNumbers(page, "#solution")).toEqual(solvedNumbers);
  await takeScreenshot(page, "picture-solved");
  // Nothing overflows the phone-sized viewport sideways.
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  expect(errors).toEqual([]);
});

test("the owner's photo is assembled as labelled and solved in exactly 24 moves", async ({
  page,
}) => {
  await openApp(page);
  await uploadOwnerPhoto(page);
  expect(await readNumbers(page, solvedGrid)).toEqual(expected.solvedNumbers);
  expect(await readNumbers(page, photoGrid)).toEqual(expected.board);
  // Only the near-tied pair the assembler is least sure of (solved positions 7 and 11) is flagged.
  await expect(page.locator(`${solvedGrid} .flagged`)).toHaveCount(2);
  await expect(page.locator("#selection")).toContainText("2 tiles are outlined");
  await takeScreenshot(page, "picture-review");
  expect(expected.moves).toBe(24);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toHaveText("Solution: 24 moves");
});

test("swapping two tiles updates the arrangement and withdraws the solution", async ({ page }) => {
  await openApp(page);
  await uploadOwnerPhoto(page);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toBeVisible();

  const before = await readImages(page, solvedGrid);
  const boardBefore = await readNumbers(page, photoGrid);
  // Solved positions 5 and 6 are both tiles whatever the empty slot's home: select one, then the other.
  await page.locator("#solved-cell-5").click();
  await expect(page.locator("#selection")).toContainText("is selected");
  await page.locator("#solved-cell-6").click();

  const after = await readImages(page, solvedGrid);
  expect(after).toEqual(before.map((n, i) => (i === 5 ? before[6] : i === 6 ? before[5] : n)));
  expect(await readNumbers(page, photoGrid)).not.toEqual(boardBefore);
  await expect(page.locator("#solution")).toBeHidden();
  // One swap changes the parity, and on the owner's photo the pair that could be blamed is the
  // flagged one, so this swap is always reported as unsolvable.
  await expect(page.locator("#check")).toContainText("Cannot solve yet");

  await page.locator("#solved-cell-5").click();
  await page.locator("#solved-cell-6").click();
  expect(await readImages(page, solvedGrid)).toEqual(before);
  expect(await readNumbers(page, photoGrid)).toEqual(boardBefore);
  await expect(page.locator("#check")).toContainText("can be solved");
});

test("the empty slot's home can be moved by tapping it and a tile", async ({ page }) => {
  await openApp(page);
  await uploadOwnerPhoto(page);
  const before = await readNumbers(page, solvedGrid);
  const home = before.indexOf(0);
  const other = home === 15 ? 0 : 15;
  await page.locator(`#solved-cell-${String(home)}`).click();
  await expect(page.locator("#selection")).toContainText("The empty square is selected");
  await page.locator(`#solved-cell-${String(other)}`).click();
  const numbers = await readNumbers(page, solvedGrid);
  expect(numbers.indexOf(0)).toBe(other);
  expect(numbers[home]).not.toBe(0);
});

test("a second photo of a solved picture puzzle is recognised from the saved puzzle", async ({
  page,
}) => {
  await openApp(page);
  await uploadOwnerPhoto(page);
  await expect(page.locator("#recognised")).toHaveCount(0);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toBeVisible();
  await expect(page.locator("#saved-note")).toContainText("Saved this puzzle");
  await expect(page.locator("#saved")).toContainText("1 picture puzzle is saved");
  const solvedImages = await readImages(page, solvedGrid);

  await page.locator("#photo-file").setInputFiles(OWNER_PHOTO);
  await expect(page.locator("#recognised")).toContainText("Recognised your saved puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
  expect(await readImages(page, solvedGrid)).toEqual(solvedImages);

  // "Not this puzzle" assembles the tiles again instead.
  await page.locator("#not-this-puzzle").click();
  await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
  await expect(page.locator("#recognised")).toHaveCount(0);

  // Forgetting empties the store, and the next photo is assembled, not recognised.
  await page.locator("#forget-saved").click();
  await expect(page.locator("#saved")).toBeHidden();
  await page.locator("#photo-file").setInputFiles(OWNER_PHOTO);
  await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
  await expect(page.locator("#recognised")).toHaveCount(0);
});

test("a numbered photo can be re-read as a picture puzzle and back", async ({ page }) => {
  await openApp(page);
  await page.locator("#photo-file").setInputFiles(`${FIXTURES}/photo-0008.jpg`);
  await expect(page.locator("#review-heading")).toHaveText("Detected board", {
    timeout: READ_TIMEOUT_MS,
  });
  await page.locator("#use-picture").click();
  await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
  await page.locator("#use-numbers").click();
  await expect(page.locator("#review-heading")).toHaveText("Detected board", {
    timeout: READ_TIMEOUT_MS,
  });
});

test("corrections made after Solve are what a later photo of the puzzle comes back as", async ({
  page,
}) => {
  await openApp(page);
  await uploadOwnerPhoto(page);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toBeVisible();
  await expect(page.locator("#saved")).toContainText("1 picture puzzle is saved");

  // Move the empty slot's home onto tile 1, after the solve.
  const original = await readNumbers(page, solvedGrid);
  await page.locator(`#solved-cell-${String(original.indexOf(0))}`).click();
  await page.locator(`#solved-cell-${String(original.indexOf(1))}`).click();
  await expect(page.locator("#check")).toContainText(/can be solved|look the same/);
  const corrected = await readImages(page, solvedGrid);

  // Reload keeps localStorage; a fresh page is offline again after it has loaded.
  await page.context().setOffline(false);
  await page.reload();
  await page.locator("body[data-ready=true]").waitFor();
  await page.waitForLoadState("networkidle");
  await page.context().setOffline(true);
  await page.locator("#photo-file").setInputFiles(OWNER_PHOTO);
  await expect(page.locator("#recognised")).toContainText("Recognised your saved puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
  expect(await readImages(page, solvedGrid)).toEqual(corrected);
  // The edit replaced the saved puzzle rather than adding a second one.
  await expect(page.locator("#saved")).toContainText("1 picture puzzle is saved");
});

test("an unsolvable swap points at the outlined tiles and hides the empty board; Solve shows its heading", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openApp(page);
  await uploadOwnerPhoto(page);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toBeVisible();
  await expect(page.locator("#solution-heading")).toBeInViewport({ ratio: 1 });

  // Swapping two tiles flips the parity and the pair that could be blamed is the flagged one, so the
  // board is unsolvable while tiles 7 and 11 stay outlined.
  await page.locator("#solved-cell-5").click();
  await page.locator("#solved-cell-6").click();
  const check = page.locator("#check");
  await expect(check).toContainText("Cannot solve yet");
  await expect(check).toContainText("outlined ones");
  await expect(check).not.toContainText("no two tiles look alike");
  await takeScreenshot(page, "picture-unsolvable");
  await expect(page.getByRole("heading", { name: "Your puzzle now" })).toHaveCount(0);
});
