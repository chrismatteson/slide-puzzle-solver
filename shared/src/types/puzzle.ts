export interface Position {
  row: number;
  col: number;
}

export interface Size {
  rows: number;
  cols: number;
}

export interface PuzzleState {
  grid: number[][];
  emptyPosition: Position;
  size: Size;
  pieces?: PuzzlePiece[];
}

export interface Move {
  from: Position;
  to: Position;
}

export interface Solution {
  moves: Move[];
}

export interface PuzzlePiece {
  id: number;
  currentPosition: {
    row: number;
    col: number;
  };
  correctPosition: {
    row: number;
    col: number;
  };
  image: string;
}

export interface PuzzleConfig {
  rows: number;
  cols: number;
  imageUrl: string;
  difficulty: 'easy' | 'medium' | 'hard';
} 