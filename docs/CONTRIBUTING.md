# Contributing to Slide Puzzle Solver

Thank you for your interest in contributing to the Slide Puzzle Solver project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the issues
2. If not, create a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots or videos if applicable
   - Environment details (OS, browser, device)

### Suggesting Enhancements

1. Check if the enhancement has been suggested
2. Create a new issue with:
   - A clear title and description
   - Use case and motivation
   - Proposed solution
   - Alternatives considered
   - Screenshots or mockups if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`npm test`)
5. Commit your changes using conventional commits
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Create a Pull Request

### Development Setup

1. Clone your fork:
```bash
git clone https://github.com/your-username/slide-puzzle-solver.git
cd slide-puzzle-solver
```

2. Install dependencies:
```bash
# Web
cd web
npm install

# Mobile
cd ../mobile
npm install
```

3. Set up environment variables:
```bash
# Copy example env files
cp .env.example .env
```

4. Start the development servers:
```bash
# Web
cd web
npm run dev

# Mobile
cd mobile
npm start
```

### Testing

- Write tests for new features
- Update tests for modified features
- Ensure all tests pass before submitting PR
- Include test coverage information
- Test offline functionality
- Test performance on different devices

### Documentation

- Update README.md if needed
- Add/update component documentation
- Include JSDoc comments for new functions
- Update CHANGELOG.md
- Document offline capabilities

### Code Style

- Follow the style guide in `docs/STYLE_GUIDE.md`
- Use TypeScript for all new code
- Follow React/React Native best practices
- Use proper naming conventions
- Optimize for client-side performance

### Commit Messages

Follow conventional commits format:
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

### Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Keep PRs focused and manageable
4. Update PR based on feedback
5. Squash commits when requested

### Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Deploy to production

## Project Structure

```
slide-puzzle-solver/
├── web/                 # Web application
├── mobile/             # React Native Expo app
├── shared/             # Shared types and utilities
└── docs/              # Project documentation
```

## Getting Help

- Check the documentation
- Search existing issues
- Ask in discussions
- Contact maintainers

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- README.md
- Release notes

Thank you for contributing to Slide Puzzle Solver! 