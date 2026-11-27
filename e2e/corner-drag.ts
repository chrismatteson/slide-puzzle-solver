// Pointer helpers shared by the corner-editor e2e tests: drags that start on a dot and end on a screen point,
// and where a point of the photo is on screen.
import { type Page } from "@playwright/test";

export async function findHandleCentre(
  page: Page,
  index: number,
): Promise<{ x: number; y: number }> {
  const box = await page.locator(`#corner-${String(index)}`).boundingBox();
  if (!box) throw new Error(`no handle ${String(index)}`);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

/** Drags handle `index` onto `target` with a mouse. */
export async function dragWithMouse(
  page: Page,
  index: number,
  target: { x: number; y: number },
): Promise<void> {
  const from = await findHandleCentre(page, index);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move((from.x + target.x) / 2, (from.y + target.y) / 2, { steps: 4 });
  await page.mouse.move(target.x, target.y, { steps: 4 });
  await page.mouse.up();
}

/** Drags handle `index` onto `target` with touch events from the DevTools protocol. */
export async function dragWithTouch(
  page: Page,
  index: number,
  target: { x: number; y: number },
): Promise<void> {
  const session = await page.context().newCDPSession(page);
  const from = await findHandleCentre(page, index);
  const touch = async (
    type: "touchStart" | "touchMove" | "touchEnd",
    at?: { x: number; y: number },
  ): Promise<void> => {
    await session.send("Input.dispatchTouchEvent", {
      type,
      touchPoints: at ? [{ x: at.x, y: at.y }] : [],
    });
  };
  await touch("touchStart", from);
  for (let i = 1; i <= 6; i++) {
    await touch("touchMove", {
      x: from.x + ((target.x - from.x) * i) / 6,
      y: from.y + ((target.y - from.y) * i) / 6,
    });
  }
  await touch("touchEnd");
  await session.detach();
}

/** Where a point of the photo (in the file's pixels) is on screen; the photo may be cropped by the zoom, never moved. */
export async function mapPhotoToScreen(
  page: Page,
  photo: { width: number; height: number },
  [x, y]: [number, number],
): Promise<{ x: number; y: number }> {
  const box = await page.locator(".corner-wrap img").boundingBox();
  if (!box) throw new Error("the editor's photo is not shown");
  return {
    x: box.x + (x / photo.width) * box.width,
    y: box.y + (y / photo.height) * box.height,
  };
}
