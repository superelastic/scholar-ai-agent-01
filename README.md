# claude-pocketflow

A template project for building PocketFlow agentic applications using Cursor, Claude-Code, and Task-Master.

## 🚀 Use This Template

Click the "Use this template" button above to create a new repository based on this template.

## Quick Start

### Prerequisites
- Python 3.8+
- UV package manager (`pip install uv`)
- API keys for any external services you plan to use

### Setup Your New Project
```bash
# After creating from template, clone your new repository
git clone <your-new-repository-url>
cd your-project-name

# Install dependencies
uv sync

# Set up your environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the example
python main.py

# Run tests to verify setup
uv run pytest
```

### Running the Application
```bash
python main.py
```

### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Type checking
uv run pyright
```

## Project Structure
```
claude-pocketflow/
├── main.py              # Application entry point
├── nodes.py             # Node definitions
├── flow.py              # Flow orchestration logic
├── utils/               # External service integrations
├── tests/               # Test suite
├── docs/                # Documentation
├── .mdc/                # Cursor-specific rules
├── CLAUDE.md            # AI assistant instructions
└── .cursorrules         # Integrated cursor rules
```

## Documentation

- **Developer Guide**: [docs/developer-guide.md](docs/developer-guide.md) - How to build with this template
- **Architecture**: [docs/architecture.md](docs/architecture.md) - System design and patterns
- **API Reference**: [docs/api-reference.md](docs/api-reference.md) - Node and utility documentation

## Development Tools

This project is optimized for AI-assisted development using:
- **Cursor AI** - AI-powered IDE with project-specific rules
- **Claude-Code** - Command-line interface for Claude
- **Task-Master** - AI task decomposition and management

## License

[Your License Here]