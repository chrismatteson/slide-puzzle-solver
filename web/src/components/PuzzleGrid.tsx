import { PuzzleState } from '@shared/types/puzzle';

interface PuzzleGridProps {
  state: PuzzleState;
}

export function PuzzleGrid({ state }: PuzzleGridProps) {
  const { grid, size, pieces } = state;

  // Find piece by ID
  const findPiece = (id: number) => {
    return pieces?.find(piece => piece.id === id);
  };

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-semibold mb-4">Detected Puzzle</h2>
      <div
        className="grid gap-1 bg-gray-800 p-2 rounded-lg"
        style={{
          gridTemplateColumns: `repeat(${size.cols}, minmax(0, 1fr))`,
          width: '100%',
          maxWidth: '500px',
          aspectRatio: `${size.cols}/${size.rows}`
        }}
      >
        {grid.map((row: number[], rowIndex: number) =>
          row.map((piece: number, colIndex: number) => {
            // Get the piece image if available
            const pieceData = piece !== 0 ? findPiece(piece) : undefined;
            
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`relative aspect-square flex items-center justify-center text-lg font-bold rounded-md overflow-hidden ${
                  piece === 0
                    ? 'bg-gray-600'
                    : 'bg-white shadow-md hover:shadow-lg transition-shadow'
                }`}
              >
                {piece !== 0 && (
                  pieceData?.image 
                    ? <div className="w-full h-full relative">
                        <img 
                          src={pieceData.image} 
                          alt={`Piece ${piece}`} 
                          className="absolute inset-0 w-full h-full object-cover"
                        />
                        <span className="absolute top-1 left-1 bg-black bg-opacity-50 text-white text-xs px-1.5 py-0.5 rounded">
                          {piece}
                        </span>
                      </div>
                    : piece
                )}
              </div>
            );
          })
        )}
      </div>
      <div className="mt-4 text-sm text-gray-500">
        {pieces && pieces.length > 0 
          ? `${size.rows}x${size.cols} puzzle detected with ${pieces.length} pieces` 
          : 'Puzzle structure detected, but no piece images available'}
      </div>
    </div>
  );
} 