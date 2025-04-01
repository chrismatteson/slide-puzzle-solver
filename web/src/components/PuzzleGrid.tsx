import { PuzzleState } from '@shared/types/puzzle';

interface PuzzleGridProps {
  state: PuzzleState;
}

export function PuzzleGrid({ state }: PuzzleGridProps) {
  const { grid, size } = state;

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-semibold mb-4">Current Puzzle State</h2>
      <div
        className="grid gap-1 bg-gray-200 p-2 rounded-lg"
        style={{
          gridTemplateColumns: `repeat(${size.cols}, minmax(0, 1fr))`,
          width: '100%',
          maxWidth: '500px'
        }}
      >
        {grid.map((row: number[], rowIndex: number) =>
          row.map((piece: number, colIndex: number) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`aspect-square flex items-center justify-center text-lg font-bold rounded-md ${
                piece === 0
                  ? 'bg-gray-300'
                  : 'bg-white shadow-md hover:shadow-lg transition-shadow'
              }`}
            >
              {piece !== 0 && piece}
            </div>
          ))
        )}
      </div>
    </div>
  );
} 