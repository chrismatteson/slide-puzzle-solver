import { type Board } from "../puzzle/board.ts";
// Inline, so restarting the solver (cancel) builds the worker from a blob and never fetches a script:
// a cancel must work offline, and a failed fetch left a dead worker that never answered.
import SolveWorker from "./solve.worker.ts?worker&inline";
import { type RgbaImage } from "../vision/image.ts";
import { type ReadResult } from "../vision/read-board.ts";
import {
  type SolveOutcome,
  type SolveRequest,
  type SolveResponse,
  type VisionRequest,
  type VisionResponse,
} from "./messages.ts";

export class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}

interface Pending<T> {
  resolve(value: T): void;
  reject(error: Error): void;
}

function takePending<T>(pending: Map<number, Pending<T>>, id: number): Pending<T> | undefined {
  const entry = pending.get(id);
  pending.delete(id);
  return entry;
}

export interface VisionClient {
  /** Resolves once the worker has loaded the classifier. */
  ready: Promise<void>;
  read(image: RgbaImage): Promise<ReadResult>;
}

export interface SolverClient {
  /** Resolves once 4x4 tables are decoded; 3x3 boards can be solved before that. */
  ready: Promise<void>;
  solve(board: Board): Promise<SolveOutcome>;
  /** Stops the running solve (its promise rejects with CancelledError) and starts a fresh worker. */
  cancel(): void;
}

export function createVisionClient(): VisionClient {
  const worker = new Worker(new URL("./vision.worker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, Pending<ReadResult>>();
  let nextId = 0;
  const ready = new Promise<void>((resolve, reject) => {
    worker.addEventListener("message", (event: MessageEvent<VisionResponse>) => {
      const message = event.data;
      if (message.type === "ready") resolve();
      else if (message.type === "read") takePending(pending, message.id)?.resolve(message.result);
      else takePending(pending, message.id)?.reject(new Error(message.message));
    });
    worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "the vision worker failed to start");
      reject(error);
      for (const entry of pending.values()) entry.reject(error);
      pending.clear();
    });
  });
  return {
    ready,
    read(image) {
      const id = nextId++;
      return new Promise<ReadResult>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        const request: VisionRequest = { type: "read", id, image };
        // The pixel buffer is not needed here afterwards, so hand it over instead of copying it.
        worker.postMessage(request, [image.data.buffer]);
      });
    },
  };
}

export function createSolverClient(patternDb: Promise<ArrayBuffer>): SolverClient {
  let worker: Worker;
  /** Set when the current worker died; solve() then rejects instead of posting to it and never settling. */
  let failed: Error | undefined;
  let pending = new Map<number, Pending<SolveOutcome>>();
  let nextId = 0;
  let resolveReady: () => void;
  let rejectReady: (error: Error) => void;
  const ready = new Promise<void>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });

  function startWorker(): void {
    const started = new SolveWorker();
    const mine = pending;
    started.addEventListener("message", (event: MessageEvent<SolveResponse>) => {
      const message = event.data;
      if (message.type === "ready") resolveReady();
      else if (message.type === "solve") takePending(mine, message.id)?.resolve(message.outcome);
      else if (message.id === undefined) rejectReady(new Error(message.message));
      else takePending(mine, message.id)?.reject(new Error(message.message));
    });
    started.addEventListener("error", (event) => {
      const error = new Error(event.message || "the solver worker failed to start");
      if (started === worker) failed = error; // a replaced worker's death is not the current one's
      rejectReady(error);
      for (const entry of mine.values()) entry.reject(error);
      mine.clear();
    });
    worker = started;
    // Cloned, not transferred: the page keeps its copy for the next worker.
    patternDb.then(
      (bytes) => {
        const request: SolveRequest = { type: "init", patternDb: bytes };
        started.postMessage(request);
      },
      (error: unknown) => {
        rejectReady(error instanceof Error ? error : new Error(String(error)));
      },
    );
  }
  startWorker();

  return {
    ready,
    solve(board) {
      if (failed) return Promise.reject(failed);
      const id = nextId++;
      return new Promise<SolveOutcome>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        const request: SolveRequest = { type: "solve", id, board };
        worker.postMessage(request);
      });
    },
    cancel() {
      worker.terminate();
      for (const entry of pending.values()) entry.reject(new CancelledError());
      pending = new Map();
      failed = undefined;
      startWorker();
    },
  };
}
