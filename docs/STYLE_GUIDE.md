# Style Guide

This document outlines the coding standards and best practices for the Slide Puzzle Solver project.

## TypeScript

### General Guidelines

- Use TypeScript for all new code
- Enable strict mode in `tsconfig.json`
- Use explicit type annotations for function parameters and return types
- Avoid using `any` type unless absolutely necessary
- Use interfaces for object shapes
- Use type aliases for complex types

### Naming Conventions

```typescript
// Interfaces
interface PuzzleState {
  grid: number[][];
  dimensions: Dimensions;
}

// Types
type Direction = 'up' | 'down' | 'left' | 'right';

// Enums
enum PuzzleDifficulty {
  Easy = 'easy',
  Medium = 'medium',
  Hard = 'hard'
}

// Functions
function calculateManhattanDistance(state: PuzzleState): number {
  // Implementation
}

// Variables
const MAX_PUZZLE_SIZE = 5;
let currentMove: Move | null = null;
```

### File Organization

- One class/interface per file
- Group related interfaces in a single file
- Use index.ts files for exports
- Keep files under 300 lines when possible

## React/React Native

### Component Structure

```typescript
// Functional Components
const PuzzleGrid: React.FC<PuzzleGridProps> = ({ state, onMove }) => {
  // Implementation
};

// Hooks
function usePuzzleState(initialState: PuzzleState) {
  // Implementation
}

// Context
const PuzzleContext = React.createContext<PuzzleContextType | null>(null);
```

### Styling

- Use TailwindCSS for web components
- Use React Native StyleSheet for mobile
- Follow mobile-first responsive design
- Use CSS variables for theming

### State Management

- Use React Context for global state
- Use local state for component-specific state
- Implement proper loading and error states
- Use proper TypeScript types for state

## Testing

### Jest/React Testing Library

```typescript
// Test file structure
describe('PuzzleGrid', () => {
  it('should render puzzle pieces correctly', () => {
    // Test implementation
  });

  it('should handle move events', () => {
    // Test implementation
  });
});
```

### Test Coverage

- Aim for 80% code coverage
- Test edge cases and error conditions
- Mock external dependencies
- Use meaningful test descriptions

## Git Workflow

### Commit Messages

Use conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Test changes
- chore: Maintenance tasks

### Branch Naming

- feature/feature-name
- bugfix/bug-description
- hotfix/issue-description
- release/version-number

## Documentation

### Code Comments

- Use JSDoc for function documentation
- Document complex algorithms
- Explain non-obvious code
- Keep comments up to date

```typescript
/**
 * Calculates the Manhattan distance heuristic for the A* algorithm
 * @param currentState - The current puzzle state
 * @param goalState - The target puzzle state
 * @returns The Manhattan distance between current and goal states
 */
function calculateManhattanDistance(
  currentState: PuzzleState,
  goalState: PuzzleState
): number {
  // Implementation
}
```

### API Documentation

- Document all API endpoints
- Include request/response examples
- Document error cases
- Keep documentation up to date

## Performance

### Optimization Guidelines

- Use React.memo for expensive components
- Implement proper loading states
- Use lazy loading for routes
- Optimize images and assets
- Implement proper caching strategies

### Code Splitting

- Split code by routes
- Lazy load components
- Use dynamic imports
- Implement proper chunking

## Accessibility

### Guidelines

- Use semantic HTML elements
- Implement proper ARIA attributes
- Ensure keyboard navigation
- Maintain proper color contrast
- Provide alt text for images

## Error Handling

### Best Practices

- Use try/catch blocks
- Implement proper error boundaries
- Log errors appropriately
- Provide user-friendly error messages
- Handle edge cases gracefully

## Security

### Guidelines

- Validate all user input
- Sanitize data before display
- Use proper authentication
- Implement rate limiting
- Follow security best practices 