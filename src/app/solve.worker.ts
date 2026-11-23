import { decodePatternTables, type PatternTables } from "../puzzle/pattern-db.ts";
import { solve } from "../puzzle/solver.ts";
import { type SolveRequest, type SolveResponse } from "./messages.ts";

// Settled by the `init` message; 4x4 solves wait on it so they never fall back to the slow search by accident.
let provideTables: (tables: Promise<PatternTables>) => void;
const tables = new Promise<PatternTables>((resolve) => {
  provideTables = resolve;
});

function reply(message: SolveResponse): void {
  postMessage(message);
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function handleSolve(
  id: number,
  request: Extract<SolveRequest, { type: "solve" }>,
): Promise<void> {
  const { board } = request;
  // 3x3 needs no tables, so it never waits for them.
  const options = board.size === 4 ? { tables: await tables } : {};
  const start = performance.now();
  const result = solve(board, options);
  const millis = performance.now() - start;
  reply({
    type: "solve",
    id,
    outcome:
      result.status === "solved"
        ? {
            status: "solved",
            moves: result.moves,
            notGuaranteedShortest: result.notGuaranteedShortest === true,
            millis,
          }
        : { status: "unsolvable" },
  });
}

addEventListener("message", (event: MessageEvent<SolveRequest>) => {
  const request = event.data;
  if (request.type === "init") {
    const decoded = decodePatternTables(new Uint8Array(request.patternDb));
    provideTables(decoded);
    decoded.then(
      () => {
        reply({ type: "ready" });
      },
      (error: unknown) => {
        reply({
          type: "error",
          message: `could not load the 4x4 solver tables: ${describeError(error)}`,
        });
      },
    );
    return;
  }
  // The page must hear about a failure, or it would wait on this board forever.
  handleSolve(request.id, request).catch((error: unknown) => {
    reply({ type: "error", id: request.id, message: describeError(error) });
  });
});
