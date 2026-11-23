import { readFileSync } from "node:fs";
import { type Page, expect, test } from "@playwright/test";
import { type Board } from "../src/puzzle/board.ts";
import { loadPatternTables } from "../src/puzzle/pattern-db-node.ts";
import { solve } from "../src/puzzle/solver.ts";

const FIXTURES = `${import.meta.dirname}/../test/fixtures`;
const SHOTS = `${import.meta.dirname}/../e2e-artifacts`;
const tables = await loadPatternTables();

// Reading a photo (a vision worker run) takes 1-3 s for an accepted read and 2-3x that for a rejected one
// (measured here, 4-core container, whole tests 2.5-4.3 s including page load; 5.6-7.6 s when the machine
// was loaded). Timeout is 10x the 3 s: Playwright's default 5 s expect timeout failed 3 of 9 tests.
const READ_TIMEOUT_MS = 30_000;

interface Label {
  size: 3 | 4;
  state: number[];
}

function readLabel(name: string): Label {
  return JSON.parse(readFileSync(`${FIXTURES}/photo-${name}.json`, "utf8")) as Label;
}

function computeExpectedSolution(label: Label): { count: number; first: string } {
  const board: Board = { size: label.size, cells: label.state };
  const result = solve(board, { tables });
  if (result.status !== "solved" || !result.moves[0]) throw new Error("fixture is not solvable");
  const { tile, direction } = result.moves[0];
  return { count: result.moves.length, first: `Move ${String(tile)} ${direction}` };
}

/** Opens the app and waits until the workers are loaded; from then on nothing may touch the network. */
async function openApp(page: Page): Promise<{ errors: string[]; requests: string[] }> {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("body[data-ready=true]").waitFor();
  await page.waitForLoadState("networkidle");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  // Belt and braces: even a request the listener missed would fail now.
  await page.context().setOffline(true);
  return { errors, requests };
}

async function uploadFixture(page: Page, name: string): Promise<void> {
  await page.locator("#photo-file").setInputFiles(`${FIXTURES}/photo-${name}.jpg`);
}

async function readGrid(page: Page): Promise<number[]> {
  const texts = await page.locator("#review .grid .cell").allTextContents();
  return texts.map((text) => (text === "" ? 0 : Number(text)));
}

async function takeScreenshot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true });
}

for (const [name, fixture] of [
  ["3x3", "0008"],
  ["4x4", "0001"],
] as const) {
  test(`reads a ${name} photo, solves it and steps through the moves`, async ({ page }) => {
    const label = readLabel(fixture);
    const expected = computeExpectedSolution(label);
    const { errors, requests } = await openApp(page);

    await uploadFixture(page, fixture);
    await expect(page.locator("#review .grid")).toBeVisible({ timeout: READ_TIMEOUT_MS });
    expect(await readGrid(page)).toEqual(label.state);
    await expect(page.locator("#check")).toContainText("consistent");
    await takeScreenshot(page, `${name}-read`);

    await page.getByRole("button", { name: "Solve", exact: true }).click();
    await expect(page.locator("#solution-heading")).toHaveText(
      `Solution: ${String(expected.count)} moves`,
    );
    await expect(page.locator("#move-label")).toContainText(expected.first);
    await expect(page.locator("#move-label")).toContainText(`step 0 of ${String(expected.count)}`);
    await expect(page.locator("#solution .cell.moving")).toHaveCount(1);
    await expect(page.locator("#solution .move-list li")).toHaveCount(expected.count);
    await takeScreenshot(page, `${name}-solved`);

    await page.keyboard.press("ArrowRight");
    await expect(page.locator("#move-label")).toContainText(`step 1 of ${String(expected.count)}`);
    await page.keyboard.press("ArrowLeft");
    await expect(page.locator("#move-label")).toContainText("step 0 of");

    // Stepping to the end shows the solved board.
    await page.locator(`#move-${String(expected.count - 1)}`).click();
    await page.locator("#next").click();
    await expect(page.locator("#move-label")).toContainText("Solved.");

    expect(errors).toEqual([]);
    expect(requests).toEqual([]);
  });
}

test("rejects an unsolvable photo with the reason, and highlights the cells to check", async ({
  page,
}) => {
  const { errors, requests } = await openApp(page);
  await uploadFixture(page, "0004");
  await expect(page.locator("#check")).toContainText("cannot be solved", {
    timeout: READ_TIMEOUT_MS,
  });
  await expect(page.locator("#solve")).toBeDisabled();
  await expect(page.locator("#review .cell.flagged")).toHaveCount(2);
  await takeScreenshot(page, "unsolvable");
  expect(errors).toEqual([]);
  expect(requests).toEqual([]);
});

test("a manual correction re-runs the check and enables Solve", async ({ page }) => {
  const label = readLabel("0008");
  await openApp(page);
  await uploadFixture(page, "0008");
  await expect(page.locator("#solve")).toBeEnabled({ timeout: READ_TIMEOUT_MS });

  // Make cell 0 a copy of cell 1's tile: a duplicate, so the board is rejected live.
  await page.locator("#cell-0").click();
  await page.getByRole("button", { name: `Set to ${String(label.state[1])}`, exact: true }).click();
  await expect(page.locator("#check")).toContainText("duplicate");
  await expect(page.locator("#solve")).toBeDisabled();
  await expect(page.locator("#review .cell.flagged")).toHaveCount(2);

  await page.locator("#cell-0").click();
  await page.getByRole("button", { name: `Set to ${String(label.state[0])}`, exact: true }).click();
  await expect(page.locator("#check")).toContainText("consistent");
  await expect(page.locator("#solve")).toBeEnabled();
  expect(await readGrid(page)).toEqual(label.state);
});

// photo-lowconf is a synthetic photo (seed 5151, `gen_photos.py --split eval`, image 44) whose 3x3 board
// the pipeline reads correctly but with one cell well below the confidence threshold (0.53). If the weights, the
// reading pipeline or MIN_CELL_CONFIDENCE change and it is read confidently, this test fails at the first
// assertion: pick another rejected-only-for-low-confidence photo (README, "Confirming uncertain cells")
// and replace the fixture.
test("a photo rejected only for low confidence is solved after the user confirms each guess", async ({
  page,
}) => {
  const label = readLabel("lowconf");
  const { errors, requests } = await openApp(page);
  await uploadFixture(page, "lowconf");

  await expect(page.locator("#check")).toContainText("Not sure about", {
    timeout: READ_TIMEOUT_MS,
  });
  await expect(page.locator("#solve")).toBeDisabled();
  const flagged = page.locator("#review .cell.flagged");
  expect(await flagged.count()).toBeGreaterThan(0);
  // The best guess is on the board and in the list, with the "check this cell" prompt.
  const items = page.locator("#review .confirm-item");
  await expect(items).toHaveCount(await flagged.count());
  await expect(items.first()).toContainText("Check this cell");
  expect(await readGrid(page)).toEqual(label.state);
  await takeScreenshot(page, "low-confidence");

  // Nothing is accepted by itself: Solve stays disabled until every flagged cell is tapped.
  while ((await page.locator("#review .confirm-item").count()) > 0) {
    await expect(page.locator("#solve")).toBeDisabled();
    await page.locator("#review .confirm-item button:not(.secondary)").first().click();
  }
  await expect(page.locator("#check")).toContainText("consistent");
  await expect(page.locator("#review .cell.flagged")).toHaveCount(0);
  await expect(page.locator("#solve")).toBeEnabled();
  expect(await readGrid(page)).toEqual(label.state);

  const expected = computeExpectedSolution(label);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toHaveText(
    `Solution: ${String(expected.count)} moves`,
  );
  expect(errors).toEqual([]);
  expect(requests).toEqual([]);
});

test("a flagged guess can be changed with the number pad instead of confirmed", async ({
  page,
}) => {
  const label = readLabel("lowconf");
  await openApp(page);
  await uploadFixture(page, "lowconf");
  await expect(page.locator("#check")).toContainText("Not sure about", {
    timeout: READ_TIMEOUT_MS,
  });

  // Change the first flagged cell to a wrong tile: the board is then inconsistent and Solve stays off.
  const first = page.locator("#review .confirm-item .secondary").first();
  await first.click();
  const wrong = label.state.find((v) => v !== 0) ?? 1;
  await page.locator(`#pad-${String(wrong)}`).click();
  await expect(page.locator("#solve")).toBeDisabled();
  await expect(page.locator("#check")).toContainText(/duplicate|cannot be solved|Not sure/);
});

test("the camera preview captures a frame and reads it", async ({ page }) => {
  const label = readLabel("0008");
  await page.addInitScript(() => {
    // Record what the app asks for, then hand over to the fake camera.
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    const asked: MediaStreamConstraints[] = [];
    (window as unknown as { cameraRequests: MediaStreamConstraints[] }).cameraRequests = asked;
    navigator.mediaDevices.getUserMedia = (constraints) => {
      asked.push(constraints ?? {});
      return real(constraints);
    };
  });
  const { errors, requests } = await openApp(page);
  await page.getByRole("button", { name: "Use camera" }).click();
  await expect(page.locator("#preview")).toBeVisible();
  await expect(page.locator(".guide")).toBeVisible();
  await expect
    .poll(() => page.locator("#preview").evaluate((v: HTMLVideoElement) => v.videoWidth))
    .toBeGreaterThan(0);
  const asked = await page.evaluate(
    () => (window as unknown as { cameraRequests: MediaStreamConstraints[] }).cameraRequests,
  );
  expect(asked).toEqual([{ video: { facingMode: "environment" } }]);
  await takeScreenshot(page, "camera");
  await page.getByRole("button", { name: "Capture" }).click();
  await expect(page.locator("#review .grid")).toBeVisible({ timeout: READ_TIMEOUT_MS });
  expect(await readGrid(page)).toEqual(label.state);
  await expect(page.locator("#camera")).toBeHidden();
  expect(errors).toEqual([]);
  expect(requests).toEqual([]);
});

test("a denied camera shows a message and upload still works", async ({ page }) => {
  await page.addInitScript(() => {
    // The one thing a test cannot do for real is click "Block" in the permission prompt.
    navigator.mediaDevices.getUserMedia = () =>
      Promise.reject(new DOMException("denied", "NotAllowedError"));
  });
  await openApp(page);
  await page.getByRole("button", { name: "Use camera" }).click();
  await expect(page.locator("#notice")).toContainText("Camera access was denied");
  await takeScreenshot(page, "camera-denied");
  await uploadFixture(page, "0008");
  await expect(page.locator("#review .grid")).toBeVisible({ timeout: READ_TIMEOUT_MS });
  await expect(page.locator("#notice")).toBeHidden();
});

test("a long solve can be cancelled and the next solve still works", async ({ page }) => {
  const label = readLabel("0001");
  await openApp(page);
  await uploadFixture(page, "0001");
  await expect(page.locator("#solve")).toBeEnabled({ timeout: READ_TIMEOUT_MS });

  // One of the hardest 4x4 states known (Korf #88, turned half a turn to this project's goal):
  // optimal search runs far past the test, so there is time to press Cancel.
  const hard = [15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0]
    .map((v) => (v === 0 ? 0 : 16 - v))
    .reverse();
  for (const [index, value] of hard.entries()) {
    await page.locator(`#cell-${String(index)}`).click();
    await page.locator(`#pad-${String(value)}`).click();
  }
  await expect(page.locator("#check")).toContainText("consistent");
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solving-status")).toContainText("Solving");
  await page.waitForTimeout(500);
  await expect(page.locator("#solving-status")).not.toContainText("Solving… 0.0 s");
  await takeScreenshot(page, "solving");
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.locator("#notice")).toContainText("Solve cancelled");
  await expect(page.locator("#solve")).toBeEnabled();

  await uploadFixture(page, "0001");
  await expect(page.locator("#review .grid")).toBeVisible({ timeout: READ_TIMEOUT_MS });
  expect(await readGrid(page)).toEqual(label.state);
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toBeVisible();
});

test("uploading a photo during a solve drops the old solve", async ({ page }) => {
  // The stale solve is cancelled, not awaited: the 25 s check below only proves anything if it would
  // otherwise have finished inside it (the unfixed app showed "88 moves" about 20 s in).
  test.setTimeout(90_000);
  const label = readLabel("0008");
  const expected = computeExpectedSolution(label);
  const { errors, requests } = await openApp(page);
  await uploadFixture(page, "0001");
  await expect(page.locator("#solve")).toBeEnabled({ timeout: READ_TIMEOUT_MS });

  // Korf #88, as in the cancel test.
  const hard = [15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0]
    .map((v) => (v === 0 ? 0 : 16 - v))
    .reverse();
  for (const [index, value] of hard.entries()) {
    await page.locator(`#cell-${String(index)}`).click();
    await page.locator(`#pad-${String(value)}`).click();
  }
  await expect(page.locator("#check")).toContainText("consistent");
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solving-status")).toContainText("Solving");

  await uploadFixture(page, "0008");
  await expect(page.locator("#review .grid .cell")).toHaveCount(9, { timeout: READ_TIMEOUT_MS });
  await expect(page.locator("#solve")).toBeEnabled();
  await page.waitForTimeout(2000);
  await expect(page.locator("#solution")).toBeHidden();
  await expect(page.locator("#cancel-solve")).toHaveCount(0);
  await expect(page.locator("#notice")).toBeHidden();

  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toHaveText(
    `Solution: ${String(expected.count)} moves`,
  );

  // The old solve would have finished (at its 20 s budget) by now.
  await page.waitForTimeout(25_000);
  await expect(page.locator("#solution-heading")).toHaveText(
    `Solution: ${String(expected.count)} moves`,
  );
  expect(await readGrid(page)).toEqual(label.state);
  expect(errors).toEqual([]);
  // Cancelling restarts the solver from an inline worker: a blob: URL made in the page, not a fetch.
  expect(requests.filter((url) => !url.startsWith("blob:"))).toEqual([]);
});

test("correcting a cell during a solve drops the old solve", async ({ page }) => {
  // As in the upload test above: the 25 s check only proves anything because the unfixed app
  // showed "Solution: 88 moves" about 20 s in.
  test.setTimeout(90_000);
  const label = readLabel("0001");
  const expected = computeExpectedSolution(label);
  const { errors, requests } = await openApp(page);
  await uploadFixture(page, "0001");
  await expect(page.locator("#solve")).toBeEnabled({ timeout: READ_TIMEOUT_MS });
  // Korf #88, as in the cancel test.
  const hard = [15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0]
    .map((v) => (v === 0 ? 0 : 16 - v))
    .reverse();
  for (const [index, value] of hard.entries()) {
    await page.locator(`#cell-${String(index)}`).click();
    await page.locator(`#pad-${String(value)}`).click();
  }
  await expect(page.locator("#check")).toContainText("consistent");
  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solving-status")).toContainText("Solving");

  // Correct the board back to what the photo shows while the solve runs.
  for (const [index, value] of label.state.entries()) {
    await page.locator(`#cell-${String(index)}`).click();
    await page.locator(`#pad-${String(value)}`).click();
  }
  await expect(page.locator("#check")).toContainText("consistent");
  await expect(page.locator("#cancel-solve")).toHaveCount(0);
  await expect(page.locator("#solve")).toBeEnabled();
  await expect(page.locator("#solution")).toBeHidden();
  await expect(page.locator("#notice")).toBeHidden();
  await page.waitForTimeout(25_000); // the old solve would have settled at its 20 s budget
  await expect(page.locator("#solution")).toBeHidden();
  expect(await readGrid(page)).toEqual(label.state);

  await page.getByRole("button", { name: "Solve", exact: true }).click();
  await expect(page.locator("#solution-heading")).toHaveText(
    `Solution: ${String(expected.count)} moves`,
  );
  expect(errors).toEqual([]);
  // Cancelling restarts the solver from an inline worker: a blob: URL made in the page, not a fetch.
  expect(requests.filter((url) => !url.startsWith("blob:"))).toEqual([]);
});

test("Use camera twice keeps one stream", async ({ page }) => {
  await page.addInitScript(() => {
    // Keep every stream the app is handed, so the test can check each one's tracks after Close.
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    const streams: MediaStream[] = [];
    (window as unknown as { cameraStreams: MediaStream[] }).cameraStreams = streams;
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await real(constraints);
      streams.push(stream);
      return stream;
    };
  });
  const { errors, requests } = await openApp(page);
  const waitForPreview = async (): Promise<void> => {
    await expect(page.locator("#preview")).toBeVisible();
    await expect
      .poll(() => page.locator("#preview").evaluate((v: HTMLVideoElement) => v.videoWidth))
      .toBeGreaterThan(0);
  };
  await page.getByRole("button", { name: "Use camera" }).click();
  await waitForPreview();
  await page.getByRole("button", { name: "Use camera" }).click();
  await waitForPreview();
  await page.getByRole("button", { name: "Close camera" }).click();
  await expect(page.locator("#camera")).toBeHidden();
  const states = await page.evaluate(() =>
    (window as unknown as { cameraStreams: MediaStream[] }).cameraStreams.flatMap((stream) =>
      stream.getTracks().map((track) => track.readyState),
    ),
  );
  expect(states.length).toBeGreaterThan(0);
  expect(states.every((state) => state === "ended")).toBe(true);
  expect(errors).toEqual([]);
  expect(requests).toEqual([]);
});
