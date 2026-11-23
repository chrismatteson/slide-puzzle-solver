// Runtime checks for board labels: eval label files and the real-photo manifest are hand-written or
// generated outside the type system, so nothing about them is trusted until checked here.

export interface BoardLabel {
  size: 3 | 4;
  state: number[];
  solvable?: boolean;
  corners?: [number, number][];
  font?: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function readCorners(value: unknown, where: string): [number, number][] {
  if (!Array.isArray(value) || value.length !== 4)
    throw new Error(`${where}: corners must be 4 [x, y] pairs`);
  return value.map((pair: unknown): [number, number] => {
    if (!Array.isArray(pair) || pair.length !== 2)
      throw new Error(`${where}: corners must be 4 [x, y] pairs`);
    const x: unknown = pair[0];
    const y: unknown = pair[1];
    if (
      typeof x !== "number" ||
      typeof y !== "number" ||
      !Number.isFinite(x) ||
      !Number.isFinite(y)
    )
      throw new Error(`${where}: corners must be finite numbers`);
    return [x, y];
  });
}

/** Checks a parsed label and returns it typed; throws `<where>: <what is wrong>` otherwise. */
export function parseBoardLabel(raw: unknown, where: string): BoardLabel {
  if (!isRecord(raw)) throw new Error(`${where}: label must be a JSON object`);
  const { size, state, solvable, corners, font } = raw;
  if (size !== 3 && size !== 4) throw new Error(`${where}: size must be 3 or 4`);
  if (!Array.isArray(state)) throw new Error(`${where}: state must be an array`);
  const values: number[] = [];
  const items: unknown[] = state;
  for (const value of items) {
    if (typeof value !== "number" || !Number.isInteger(value))
      throw new Error(`${where}: state must hold integers only`);
    values.push(value);
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length !== size * size || sorted.some((v, i) => v !== i))
    throw new Error(`${where}: state is not a permutation of 0..${String(size * size - 1)}`);
  const label: BoardLabel = { size, state: values };
  if (solvable !== undefined) {
    if (typeof solvable !== "boolean") throw new Error(`${where}: solvable must be true or false`);
    label.solvable = solvable;
  }
  if (corners !== undefined) label.corners = readCorners(corners, where);
  if (font !== undefined) {
    if (typeof font !== "string") throw new Error(`${where}: font must be a string`);
    label.font = font;
  }
  return label;
}
