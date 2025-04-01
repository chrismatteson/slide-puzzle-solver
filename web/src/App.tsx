import React, { useState } from 'react'
import { Camera } from './components/Camera'
import { PuzzleGrid } from './components/PuzzleGrid'
import { SolutionViewer } from './components/SolutionViewer'
import { PuzzleState, Solution } from '@shared/types/puzzle'
import { PuzzleSolver } from '@shared/utils/puzzleSolver'
import './App.css'

export function App() {
  const [puzzleState, setPuzzleState] = useState<PuzzleState | null>(null)
  const [solution, setSolution] = useState<Solution | null>(null)
  const [isARMode, setIsARMode] = useState(false)

  const handlePuzzleDetected = (state: PuzzleState) => {
    setPuzzleState(state)
    if (!solution) {
      const solver = new PuzzleSolver(state)
      const newSolution = solver.solve(state)
      setSolution(newSolution)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-3xl font-bold text-gray-900">Slide Puzzle Solver</h1>
            <button
              onClick={() => setIsARMode(!isARMode)}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                isARMode ? 'bg-green-600 hover:bg-green-700' : 'bg-indigo-600 hover:bg-indigo-700'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
            >
              {isARMode ? 'Exit AR Mode' : 'Enter AR Mode'}
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="h-96">
                <Camera onPuzzleDetected={handlePuzzleDetected} isARMode={isARMode} />
              </div>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg p-4">
              {puzzleState ? (
                <>
                  <PuzzleGrid state={puzzleState} />
                  <SolutionViewer solution={solution} />
                </>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  Point your camera at a slide puzzle to begin
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
