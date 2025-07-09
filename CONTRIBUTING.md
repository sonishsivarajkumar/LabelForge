# Contributing to LabelForge

We welcome contributions to LabelForge! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/LabelForge.git
   cd LabelForge
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```
4. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Use descriptive variable and function names

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a PR:
  ```bash
  pytest tests/
  ```
- Aim for good test coverage of new code

### Documentation

- Update documentation for any new features
- Include examples in docstrings where helpful
- Update the README if necessary

## Types of Contributions

### Labeling Functions

We welcome contributions of reusable labeling functions for common domains:

- Create LFs in `src/labelforge/library/` 
- Include clear documentation and examples
- Add tests demonstrating the LF behavior

### Core Features

For major features:

1. Open an issue first to discuss the proposed change
2. Follow the existing code patterns and architecture
3. Include comprehensive tests
4. Update documentation

### Bug Fixes

1. Create an issue describing the bug (if one doesn't exist)
2. Fix the bug with minimal changes
3. Add tests to prevent regression
4. Reference the issue in your commit message

## Submitting Changes

1. Ensure your code follows the style guidelines
2. Add or update tests as needed
3. Update documentation if necessary
4. Commit your changes with a clear commit message:
   ```bash
   git commit -m "Add feature X that does Y"
   ```
5. Push to your fork and submit a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include screenshots or examples if relevant
- Be responsive to code review feedback

## Community Guidelines

- Be respectful and constructive in discussions
- Help newcomers get started
- Share knowledge and best practices
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

If you have questions about contributing:

- Open an issue for discussion
- Join our community discussions
- Check existing documentation and issues

Thank you for contributing to LabelForge!
