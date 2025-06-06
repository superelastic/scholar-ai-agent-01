# Scholar AI Agent 🎓

An intelligent academic research assistant that analyzes seminal papers, finds citations, and suggests future research directions using a coordinated multi-agent system built on PocketFlow.

## 🌟 Features

- **📄 Academic Paper Analysis**: Extract metadata, key concepts, methodology, and findings
- **🔍 Citation Discovery**: Find recent papers that cite your work using Google Scholar
- **🔬 Research Direction Generation**: AI-powered suggestions for future research
- **📊 Multi-Format Output**: Export results in JSON, Markdown, HTML, and Plain Text
- **⚡ Real-time Progress**: Track analysis progress with live updates
- **🤖 Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **🛡️ Robust Error Handling**: Graceful degradation when services are unavailable

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for OpenAI or Anthropic (for LLM analysis)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd scholar-ai-agent-01
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_openai_key_here
   # or
   # ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

### Basic Usage

**Analyze a single paper**:
```bash
python run_scholar_ai.py paper.pdf
```

**Export results in specific formats**:
```bash
python run_scholar_ai.py paper.pdf --formats json markdown html
```

**Specify output directory**:
```bash
python run_scholar_ai.py paper.pdf --output-dir ./my_results
```

**Export all formats with verbose output**:
```bash
python run_scholar_ai.py paper.pdf --export-all --verbose
```

## 📖 Detailed Usage

### Command Line Interface

The `run_scholar_ai.py` script provides a simple command-line interface:

```bash
python run_scholar_ai.py <pdf_path> [options]
```

**Options**:
- `--output-dir, -o`: Directory for output files (default: ./exports)
- `--formats, -f`: Output formats (choices: json, markdown, html, txt)
- `--export-all`: Export in all available formats
- `--verbose, -v`: Enable verbose logging
- `--no-export`: Skip file export, just show results

**Examples**:
```bash
# Basic analysis with default exports
python run_scholar_ai.py research_paper.pdf

# Analysis with custom output directory
python run_scholar_ai.py research_paper.pdf -o ./results

# Export only JSON and Markdown
python run_scholar_ai.py research_paper.pdf -f json markdown

# Verbose mode with all formats
python run_scholar_ai.py research_paper.pdf --export-all --verbose

# Quick analysis without file export
python run_scholar_ai.py research_paper.pdf --no-export
```

### Programmatic Usage

You can also use the Scholar AI system programmatically:

```python
import asyncio
from coordinator import ScholarAICoordinator
from utils import export_analysis_results

async def analyze_paper():
    # Initialize coordinator
    coordinator = ScholarAICoordinator(config={
        'coordinator': {'timeout': 300},
        'web_search': {'max_results': 20},
        'research_synthesis': {'min_confidence': 0.7}
    })
    
    # Analyze paper
    results = await coordinator.process_paper('paper.pdf')
    
    if results['success']:
        # Export results
        export_info = export_analysis_results(
            results['results'],
            formats=['json', 'markdown', 'html'],
            output_dir='./exports'
        )
        print(f"Analysis complete! Results exported to {export_info['files']}")
    else:
        print(f"Analysis failed: {results['error']}")

# Run the analysis
asyncio.run(analyze_paper())
```

## 🏗️ Architecture

The Scholar AI Agent uses a multi-agent architecture with three specialized agents:

### 🎯 Academic Coordinator Agent
- **Purpose**: Main orchestrator managing the research workflow
- **Responsibilities**: PDF processing, content extraction, LLM analysis
- **Nodes**: UserInputNode, PaperAnalysisNode, CitationSearchNode, ResearchSynthesisNode, PresentationNode

### 🔍 Academic Web Search Agent  
- **Purpose**: Specialized for finding citing papers via Google Scholar
- **Responsibilities**: Query generation, Scholar search, result filtering, citation formatting
- **Nodes**: SearchQueryNode, GoogleScholarNode, CitationFilterNode, CitationFormatterNode

### 🔬 Academic Research Synthesis Agent
- **Purpose**: Synthesizes information to suggest research directions
- **Responsibilities**: Trend analysis, gap identification, direction generation, suggestion formatting
- **Nodes**: PaperSynthesisNode, TrendAnalysisNode, DirectionGeneratorNode, SuggestionFormatterNode

## 📊 Output Formats

The system generates professional output in multiple formats:

### 📋 JSON Format
- Structured data perfect for further processing
- Complete metadata and analysis results
- Machine-readable format for integration

### 📝 Markdown Format  
- Professional documentation style
- Great for GitHub, documentation sites
- Human-readable with proper formatting

### 🌐 HTML Format
- Styled reports with CSS
- Interactive and visually appealing
- Suitable for web presentation

### 📄 Plain Text Format
- Clean, simple text output
- Great for email or plain text systems
- Proper text wrapping and formatting

## ⚙️ Configuration

You can customize the behavior through configuration:

```python
config = {
    'coordinator': {
        'timeout': 300,        # Overall timeout in seconds
        'upload_dir': './uploads'
    },
    'web_search': {
        'max_results': 20,     # Maximum citations to find
        'year_filter': 2020,   # Only papers from this year onwards
        'relevance_threshold': 0.3  # Minimum relevance score
    },
    'research_synthesis': {
        'min_confidence': 0.7, # Minimum confidence for suggestions
        'max_suggestions': 5   # Maximum research directions
    }
}
```

## 📂 Output Structure

When you run an analysis, the system creates:

```
exports/
├── paper_name_20231201_143022/          # Session directory
│   ├── paper_name.json                  # JSON format
│   ├── paper_name.md                    # Markdown format  
│   ├── paper_name.html                  # HTML format
│   ├── paper_name.txt                   # Plain text format
│   ├── export_metadata.json             # Export metadata
│   └── paper_name_export_bundle.zip     # ZIP bundle of all files
```

## 🔧 Advanced Usage

### Running Specific Components

You can run individual components for testing:

```bash
# Test PDF extraction
python examples/test_pdf_extraction.py

# Test LLM analysis
python examples/test_llm_analysis.py

# Test Google Scholar search
python examples/test_scholar_search.py

# Test full coordinator workflow
python examples/test_coordinator_workflow.py

# Test presentation and UI features
python examples/test_presentation_ui.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_formatters.py -v
python -m pytest tests/test_progress_tracker.py -v
python -m pytest tests/test_export_manager.py -v

# Run with coverage
python -m pytest tests/ --cov=utils --cov=nodes --cov=agents
```

### Code Quality

```bash
# Format code
python -m ruff format .

# Check linting
python -m ruff check . --fix

# Type checking (if pyright is installed)
pyright
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'pocketflow'"**
   ```bash
   pip install pocketflow
   ```

2. **"API key not found"**
   - Make sure you've set up your `.env` file with valid API keys
   - Check that the environment variables are loaded correctly

3. **"PDF file not found"**
   - Ensure the PDF path is correct and the file exists
   - Use absolute paths if relative paths don't work

4. **"Google Scholar rate limiting"**
   - The system includes rate limiting and retry logic
   - Wait a few minutes and try again
   - Consider reducing `max_results` in configuration

5. **"Analysis takes too long"**
   - Increase the timeout in configuration
   - Check your internet connection
   - Verify API keys are working

### Debug Mode

Enable verbose logging to see detailed information:

```bash
python run_scholar_ai.py paper.pdf --verbose
```

## 📝 Sample Output

Here's what you can expect from an analysis:

```
🎓 Scholar AI Agent - Academic Paper Analysis System
============================================================
📄 Analyzing: attention_is_all_you_need.pdf
⏳ This may take 30-60 seconds...

✅ Analysis completed successfully!
==================================================
📄 Paper: Attention Is All You Need
👥 Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar
📅 Year: 2017

🔑 Key Concepts (8):
   • transformer architecture
   • self-attention mechanism
   • multi-head attention
   • positional encoding
   • sequence-to-sequence models

📚 Citations Found: 15
🔬 Research Directions: 4
⏱️  Processing Time: 45.3 seconds
==================================================

📁 Exporting results to: ./exports
✅ Export successful!
   JSON: ./exports/attention_is_all_you_need_20231201_143022/attention_is_all_you_need.json
   MARKDOWN: ./exports/attention_is_all_you_need_20231201_143022/attention_is_all_you_need.md
   HTML: ./exports/attention_is_all_you_need_20231201_143022/attention_is_all_you_need.html
   ZIP Bundle: ./exports/attention_is_all_you_need_20231201_143022/attention_is_all_you_need_export_bundle.zip

🎉 Analysis complete! Check the exported files for detailed results.
```

## 🤝 Contributing

This project uses AI-assisted development tools:
- **Cursor AI** - AI-powered IDE
- **Claude-Code** - Command-line interface for Claude
- **Task-Master** - AI task decomposition

## 📄 License

[Your License Here]

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Enable verbose mode for debugging
3. Check the logs in the console output
4. Verify your API keys and internet connection

---

**Happy researching! 🎓✨**