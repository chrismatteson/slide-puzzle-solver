# Slide Puzzle Solver

A web application that helps solve slide puzzles using computer vision and AI. The application can detect a slide puzzle from a camera feed, analyze its current state, and provide step-by-step instructions to solve it.

## Features

- Real-time puzzle detection using computer vision
- Automatic puzzle state analysis
- A* algorithm for finding optimal solutions
- Step-by-step solution visualization
- Mobile-friendly interface
- AR mode for overlay instructions

## Project Structure

```
slide-puzzle-solver/
├── shared/           # Shared types and utilities
│   ├── src/
│   │   ├── types/   # TypeScript type definitions
│   │   └── utils/   # Shared utility functions
│   └── package.json
└── web/             # Web application
    ├── src/
    │   ├── components/  # React components
    │   └── App.tsx      # Main application
    └── package.json
```

## Prerequisites

- Node.js (v16 or later)
- npm (v7 or later)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/slide-puzzle-solver.git
   cd slide-puzzle-solver
   ```

2. Install dependencies:
   ```bash
   # Install shared package dependencies
   cd shared
   npm install
   npm run build

   # Install web application dependencies
   cd ../web
   npm install
   ```

3. Start the development server:
   ```bash
   cd web
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Allow camera access when prompted
2. Point your camera at a slide puzzle
3. The application will automatically detect and analyze the puzzle
4. Once detected, you'll see the current state and solution steps
5. Follow the step-by-step instructions to solve the puzzle

## Development

- The shared package contains reusable types and utilities
- The web application is built with React and Vite
- TailwindCSS is used for styling
- TypeScript is used throughout the project

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
