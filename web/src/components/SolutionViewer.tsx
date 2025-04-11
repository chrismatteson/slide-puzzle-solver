import { useState } from 'react';
import { Solution } from '@shared/types/puzzle';

interface SolutionViewerProps {
  solution: Solution | null;
}

export function SolutionViewer({ solution }: SolutionViewerProps) {
  const [currentStep, setCurrentStep] = useState(0);

  if (!solution || solution.moves.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <p className="text-gray-500">No solution available yet</p>
      </div>
    );
  }

  const handlePreviousStep = () => {
    setCurrentStep(prev => Math.max(0, prev - 1));
  };

  const handleNextStep = () => {
    setCurrentStep(prev => Math.min(solution.moves.length - 1, prev + 1));
  };

  const currentMove = solution.moves[currentStep];
  if (!currentMove) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <p className="text-gray-500">Invalid solution step</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-semibold mb-4">Solution Steps</h2>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          Step {currentStep + 1} of {solution.moves.length}
        </p>
      </div>

      <div className="flex items-center space-x-4 mb-4">
        <button
          onClick={handlePreviousStep}
          disabled={currentStep === 0}
          className="px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={handleNextStep}
          disabled={currentStep === solution.moves.length - 1}
          className="px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>

      <div className="text-center">
        <p className="text-lg font-medium">
          Move {currentMove.from.row + 1}, {currentMove.from.col + 1} to {currentMove.to.row + 1}, {currentMove.to.col + 1}
        </p>
      </div>
    </div>
  );
} 