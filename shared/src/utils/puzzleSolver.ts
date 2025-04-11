import { PuzzleState, Move, Solution } from '../types/puzzle';

interface Node {
  state: PuzzleState;
  g: number; // Cost from start to current node
  h: number; // Heuristic (estimated cost to goal)
  f: number; // Total cost (g + h)
  parent: Node | null;
  move: Move | null;
}

export class PuzzleSolver {
  private goalState: PuzzleState;

  constructor(goalState: PuzzleState) {
    this.goalState = goalState;
  }

  private calculateManhattanDistance(state: PuzzleState): number {
    let distance = 0;
    const { rows, cols } = state.size;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const currentValue = state.grid[row][col];
        if (currentValue === 0) continue; // Skip empty space

        // Find current value's position in goal state
        let goalRow = 0, goalCol = 0;
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            if (this.goalState.grid[i][j] === currentValue) {
              goalRow = i;
              goalCol = j;
              break;
            }
          }
        }

        distance += Math.abs(row - goalRow) + Math.abs(col - goalCol);
      }
    }

    return distance;
  }

  private getPossibleMoves(state: PuzzleState): Move[] {
    const moves: Move[] = [];
    const { row, col } = state.emptyPosition;
    const { rows, cols } = state.size;

    // Check all possible moves (up, right, down, left)
    if (row > 0) moves.push({ from: { row: row - 1, col }, to: { row, col } });
    if (col < cols - 1) moves.push({ from: { row, col: col + 1 }, to: { row, col } });
    if (row < rows - 1) moves.push({ from: { row: row + 1, col }, to: { row, col } });
    if (col > 0) moves.push({ from: { row, col: col - 1 }, to: { row, col } });

    return moves;
  }

  private applyMove(state: PuzzleState, move: Move): PuzzleState {
    const newGrid = state.grid.map(row => [...row]);
    const { from, to } = move;
    
    // Swap the values
    newGrid[to.row][to.col] = newGrid[from.row][from.col];
    newGrid[from.row][from.col] = 0;

    return {
      ...state,
      grid: newGrid,
      emptyPosition: from
    };
  }

  private isGoalState(state: PuzzleState): boolean {
    return JSON.stringify(state.grid) === JSON.stringify(this.goalState.grid);
  }

  public solve(initialState: PuzzleState): Solution {
    const openSet: Node[] = [{
      state: initialState,
      g: 0,
      h: this.calculateManhattanDistance(initialState),
      f: this.calculateManhattanDistance(initialState),
      parent: null,
      move: null
    }];

    const closedSet = new Set<string>();
    const moves: Move[] = [];

    while (openSet.length > 0) {
      // Find node with lowest f value in open set
      let currentIndex = 0;
      for (let i = 1; i < openSet.length; i++) {
        if (openSet[i].f < openSet[currentIndex].f) {
          currentIndex = i;
        }
      }

      const current = openSet[currentIndex];

      if (this.isGoalState(current.state)) {
        // Reconstruct path
        let node: Node | null = current;
        while (node && node.move) {
          moves.unshift(node.move);
          node = node.parent;
        }
        return { moves };
      }

      // Move current node from open to closed set
      openSet.splice(currentIndex, 1);
      closedSet.add(JSON.stringify(current.state.grid));

      // Generate successors
      const possibleMoves = this.getPossibleMoves(current.state);
      for (const move of possibleMoves) {
        const successorState = this.applyMove(current.state, move);
        const successorKey = JSON.stringify(successorState.grid);

        if (closedSet.has(successorKey)) continue;

        const g = current.g + 1;
        const h = this.calculateManhattanDistance(successorState);
        const f = g + h;

        const successor: Node = {
          state: successorState,
          g,
          h,
          f,
          parent: current,
          move
        };

        openSet.push(successor);
      }
    }

    throw new Error('No solution found');
  }
} 