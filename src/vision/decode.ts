// Joint decoding of a whole board. The cells hold each of 0..N²-1 exactly once, so instead of
// taking each cell's best class on its own, pick the assignment of values to cells with the highest
// total log-probability (an assignment problem). A cell the classifier hesitates over is then
// settled by the values the other cells already use.
//
// Solvability is deliberately NOT a constraint here: forcing it would "repair" a genuinely
// swapped board into a confident wrong answer. checkReading rejects unsolvable boards afterwards.

/** Probabilities below this are treated as this, so one impossible class cannot make a cost infinite. */
const MIN_PROBABILITY = 1e-9;
/** Stands in for "not allowed" in a cost matrix; far above any real total (16 * -ln(1e-9) = 332). */
const FORBIDDEN = 1e6;

export interface DecodedBoard {
  /** Value assigned to each cell, row-major. */
  values: number[];
  /**
   * Per cell: how much more likely the best assignment is than the best one that puts a different
   * value in this cell, as a probability in (0.5, 1] (a logistic of the log-likelihood gap).
   * 0.5 means a tie, so a tie never passes the confidence threshold.
   */
  confidences: number[];
  /** Log-likelihood of the chosen assignment, per cell. */
  meanLogLikelihood: number;
}

/**
 * Minimum-cost perfect assignment of rows to columns of a square matrix (Hungarian algorithm with
 * potentials, O(n³)). Returns the column chosen for each row. Not taken from a library: the common
 * npm packages are unmaintained or untyped, and this is the textbook algorithm. Ties go to the
 * lowest-numbered column, so results are deterministic.
 */
export function solveAssignment(cost: readonly (readonly number[])[]): number[] {
  const n = cost.length;
  // 1-based arrays, with index 0 as the virtual row/column of the textbook formulation.
  const rowPotential = new Array<number>(n + 1).fill(0);
  const colPotential = new Array<number>(n + 1).fill(0);
  const rowOfCol = new Array<number>(n + 1).fill(0);
  const previousCol = new Array<number>(n + 1).fill(0);
  for (let row = 1; row <= n; row++) {
    rowOfCol[0] = row;
    let col = 0;
    const minSlack = new Array<number>(n + 1).fill(Infinity);
    const visited = new Array<boolean>(n + 1).fill(false);
    do {
      visited[col] = true;
      const currentRow = rowOfCol[col] ?? 0;
      let delta = Infinity;
      let next = 0;
      for (let c = 1; c <= n; c++) {
        if (visited[c]) continue;
        const slack =
          (cost[currentRow - 1]?.[c - 1] ?? FORBIDDEN) -
          (rowPotential[currentRow] ?? 0) -
          (colPotential[c] ?? 0);
        if (slack < (minSlack[c] ?? Infinity)) {
          minSlack[c] = slack;
          previousCol[c] = col;
        }
        if ((minSlack[c] ?? Infinity) < delta) {
          delta = minSlack[c] ?? Infinity;
          next = c;
        }
      }
      for (let c = 0; c <= n; c++) {
        if (visited[c]) {
          rowPotential[rowOfCol[c] ?? 0] = (rowPotential[rowOfCol[c] ?? 0] ?? 0) + delta;
          colPotential[c] = (colPotential[c] ?? 0) - delta;
        } else {
          minSlack[c] = (minSlack[c] ?? 0) - delta;
        }
      }
      col = next;
    } while ((rowOfCol[col] ?? 0) !== 0);
    do {
      const back = previousCol[col] ?? 0;
      rowOfCol[col] = rowOfCol[back] ?? 0;
      col = back;
    } while (col !== 0);
  }
  const columnOfRow = new Array<number>(n).fill(0);
  for (let c = 1; c <= n; c++) columnOfRow[(rowOfCol[c] ?? 1) - 1] = c - 1;
  return columnOfRow;
}

function sumAssignmentCost(
  cost: readonly (readonly number[])[],
  columnOfRow: readonly number[],
): number {
  return columnOfRow.reduce((sum, c, r) => sum + (cost[r]?.[c] ?? FORBIDDEN), 0);
}

/**
 * Decodes `probabilities[cell][value]` (one row per cell, one column per value, N² of each) into
 * the most likely permutation, with a confidence per cell.
 */
export function decodePermutation(probabilities: readonly (readonly number[])[]): DecodedBoard {
  const n = probabilities.length;
  const cost = probabilities.map((row) => {
    if (row.length !== n) throw new Error(`expected ${String(n)} probabilities per cell`);
    return row.map((p) => -Math.log(Math.max(p, MIN_PROBABILITY)));
  });
  const best = solveAssignment(cost);
  const bestCost = sumAssignmentCost(cost, best);
  const confidences = best.map((value, cell) => {
    // Best assignment that gives this cell anything else; the gap to `bestCost` is the evidence for `value`.
    const alternative = cost.map((row, r) =>
      r === cell ? row.map((c, v) => (v === value ? FORBIDDEN : c)) : row,
    );
    const gap = sumAssignmentCost(alternative, solveAssignment(alternative)) - bestCost;
    return 1 / (1 + Math.exp(-gap));
  });
  return { values: best, confidences, meanLogLikelihood: -bestCost / n };
}
