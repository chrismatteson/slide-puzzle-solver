// Page helpers shared by the e2e files that read photos in the app.
import { type Page, expect } from "@playwright/test";
import { type PatternTables } from "../src/puzzle/pattern-db.ts";
import { solve } from "../src/puzzle/solver.ts";
import { type PictureLabel } from "../scripts/labels.ts";

const SHOTS = `${import.meta.dirname}/../e2e-artifacts`;
const OWNER_PHOTO = `${import.meta.dirname}/../test/fixtures/owner-picture/picture-4x4-01.jpg`;

// A read takes 1-4 s, a picture photo 2-4 s, more on a loaded machine. Same margin as app.e2e.ts.
export const READ_TIMEOUT_MS = 30_000;

/** Opens the app, waits until it is ready and then goes offline; returns the page's console and page errors. */
export async function openApp(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("body[data-ready=true]").waitFor();
  await page.waitForLoadState("networkidle");
  // Offline from here on: a request would fail the read or solve it belongs to.
  await page.context().setOffline(true);
  return errors;
}

export async function takeScreenshot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true });
}

/** The numbers shown in the cells of the grid at `selector`; an empty cell is 0. */
export async function readNumbers(page: Page, selector: string): Promise<number[]> {
  const texts = await page.locator(`${selector} .cell`).allTextContents();
  return texts.map((text) => (text === "" ? 0 : Number(text)));
}

/** What is drawn in each cell of a grid ("" for the empty slot): the numbers are fixed labels, the images move. */
export async function readImages(page: Page, selector: string): Promise<string[]> {
  return page
    .locator(`${selector} .cell`)
    .evaluateAll((cells) => cells.map((cell) => cell.querySelector("canvas")?.toDataURL() ?? ""));
}

/** Uploads the owner's picture photo and waits for it to be read as a picture puzzle. */
export async function uploadOwnerPhoto(page: Page): Promise<void> {
  await page.locator("#photo-file").setInputFiles(OWNER_PHOTO);
  await expect(page.locator("#review-heading")).toHaveText("Detected picture puzzle", {
    timeout: READ_TIMEOUT_MS,
  });
}

/**
 * What the app should show for a labelled picture photo: the tile number at each solved position (0 at
 * the empty slot's home), the numbers on the photo's board, and the length of the shortest solution.
 */
export function solveLabelledPuzzle(
  label: PictureLabel,
  tables: PatternTables,
): { solvedNumbers: number[]; board: number[]; moves: number } {
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
