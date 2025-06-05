# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Quick Reference

### Common Commands
```bash
# Package Management
uv add package                  # Install package
uv run tool                     # Run tool
uv add --dev package --upgrade-package package  # Upgrade

# Code Quality
uv run ruff format .           # Format code
uv run ruff check . --fix      # Fix linting issues
uv run pyright                 # Type checking
uv run pytest                  # Run tests

# Git Operations
git status                     # Check changes
git diff                       # Review changes
git commit --trailer "Reported-by:<name>"     # For user-reported issues
git commit --trailer "Github-Issue:#<number>"  # For GitHub issues
```

## Project Structure

### Documentation
- `docs/design.md` - Product requirements, user stories, acceptance criteria
- `docs/tasks/` - Task-master generated tasks (if using claude-task-master)
- `.mdc/` - Cursor-specific rules and framework patterns

### Core Components
- `config.py`: Configuration management
- `daemon.py`: Main daemon
[etc... fill in here]

## Core Development Rules

1. Package Management
   - Use UV for most operations, pip only for tools that might have uv compatibility 
     issues, or where documentation directs pip usage (PocketFlow requires `pip install pocketflow` per its documentation).
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## System Architecture

[fill in here]

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `ArthurClune` as reviewer.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Python Tools

## Code Formatting

1. Ruff
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

2. Type Checking
   - Tool: `uv run pyright`
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

3. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Prettier (YAML/JSON), Ruff (Python)
   - Ruff updates:
     - Check PyPI versions
     - Update config rev
     - Commit config first

## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly

## Documentation Structure

### For AI/LLM Development
- LLM Instructions: CLAUDE.md (this file), .cursorrules
- Framework Patterns: `.mdc/` directory
- Design Specifications: `docs/design.md`, `docs/flow-design.md`
- Task Decomposition: `docs/tasks/` (if using task-master)

### For Human Developers
- Getting Started: `README.md`
- Developer Guide: `docs/developer-guide.md`
- Architecture Overview: `docs/architecture.md`
- API Documentation: `docs/api-reference.md`

## External Documentation

For detailed information, refer to:
- Product Requirements: `docs/design.md`
- Flow Design Patterns: `docs/flow-design.md`
- Framework Patterns: `.mdc/pocketflow-rules.md`
- Current Tasks: `docs/tasks/current.md` (if using task-master)