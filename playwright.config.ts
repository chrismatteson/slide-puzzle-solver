import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { defineConfig } from "@playwright/test";

const PORT = 4173;
const ARTIFACTS = "e2e-artifacts";

// Chromium's fake camera plays a file of concatenated JPEGs (MJPEG), so the camera test captures a
// real fixture photo instead of Chromium's test pattern.
mkdirSync(ARTIFACTS, { recursive: true });
const fakeCameraFile = `${import.meta.dirname}/${ARTIFACTS}/fake-camera.mjpeg`;
const frame = readFileSync(`${import.meta.dirname}/test/fixtures/photo-0008.jpg`);
writeFileSync(fakeCameraFile, Buffer.concat(Array.from({ length: 30 }, () => frame)));

export default defineConfig({
  testDir: "e2e",
  testMatch: "**/*.e2e.ts",
  outputDir: `${ARTIFACTS}/test-results`,
  reporter: "list",
  // Reading takes a few seconds on a slow CI core; the cancel test waits on a deliberately long solve.
  timeout: 60_000,
  use: {
    baseURL: `http://localhost:${String(PORT)}`,
    viewport: { width: 390, height: 844 },
    permissions: ["camera"],
    launchOptions: {
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-video-capture=${fakeCameraFile}`,
      ],
    },
  },
  webServer: {
    command: `npm run build && npm run preview -- --port ${String(PORT)} --strictPort`,
    url: `http://localhost:${String(PORT)}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
