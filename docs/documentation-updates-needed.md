# Documentation Updates Needed for Citation-Based API Design

## Overview
This document identifies specific updates needed across all documentation files to align with the new citation-based API design approach outlined in design.md.

## 1. architecture.md Updates

### Component Architecture Section (Lines 28-68)
- Add new "Citation Service Layer" between Node Layer and Utility Layer
- Include citation processing components: LLM interface, citation extraction, citation validation
- Add diagram showing data flow through citation pipeline

### Data Flow Section (Lines 69-99)
- Update store structure example to include citation-specific keys:
  ```python
  store = {
      # Citation inputs
      "research_focus": "climate impacts of urban tree loss",
      "retrieved_citations": [...],
      
      # Citation processing
      "llm_response": {...},
      "extracted_citations": [...],
      "validation_results": {...},
      
      # Citation outputs
      "formatted_citations": {...},
      "citation_markdown": "..."
  }
  ```

### New Section: Citation Processing Architecture
Add after line 99:
- Citation retrieval strategies
- LLM prompt engineering for citations
- Citation validation pipeline
- Output formatting system

### Scalability Considerations (Lines 121-133)
- Add considerations for LLM rate limiting
- Citation caching strategies
- Batch processing of citations

## 2. flow-design.md Updates

### Example Flows Section (Lines 53-116)
Replace the generic API Integration Flow with a "Citation Processing Flow":
```python
citation_flow = {
    "start": {
        "node": "ResearchInputNode",
        "transitions": {
            "valid": "retrieve_citations",
            "invalid": "handle_input_error"
        }
    },
    "retrieve_citations": {
        "node": "CitationRetrievalNode",
        "transitions": {
            "found": "process_with_llm",
            "not_found": "handle_no_citations",
            "error": "handle_retrieval_error"
        }
    },
    "process_with_llm": {
        "node": "LLMCitationNode",
        "transitions": {
            "success": "validate_citations",
            "rate_limited": "wait_and_retry",
            "error": "handle_llm_error"
        }
    },
    "validate_citations": {
        "node": "CitationValidationNode",
        "transitions": {
            "valid": "format_output",
            "invalid": "handle_validation_error"
        }
    },
    "format_output": {
        "node": "CitationFormatterNode",
        "transitions": {
            "complete": "end",
            "error": "handle_format_error"
        }
    }
}
```

### Store Design Patterns Section (Lines 134-168)
Add citation-specific store patterns:
- Citation metadata structure
- LLM response caching
- Validation state tracking

## 3. developer-guide.md Updates

### Core Concepts Section (Lines 9-25)
- Add "Citation Processing" as a fourth core concept
- Explain how citations flow through the system

### Creating Your First Node Section (Lines 27-88)
Replace GreetingNode example with a CitationExtractionNode example that demonstrates:
- Parsing research focus
- Handling citation data
- Error handling for missing citations

### Working with Utilities Section (Lines 121-164)
Replace WeatherAPI example with:
- ScholarAPI utility for citation retrieval
- LLMService utility for OpenAI integration
- CitationValidator utility

### Best Practices Section (Lines 166-187)
Add citation-specific best practices:
- Citation data validation
- LLM prompt optimization
- Rate limiting strategies
- Citation formatting standards

### AI-Assisted Development Workflow (Lines 189-210)
Update examples to show:
- Creating citation processing nodes
- Testing with mock citation data
- Debugging LLM responses

## 4. api-reference.md Updates

### Built-in Nodes Section (Lines 29-72)
Add new citation-specific nodes:

#### CitationRetrievalNode
```
**Store Keys:**
- Input:
  - `research_focus`: str - Research topic/question
  - `max_citations`: int - Maximum citations to retrieve
- Output:
  - `citations`: List[Dict] - Retrieved citation data
  - `retrieval_metadata`: Dict - API response metadata
```

#### LLMCitationNode
```
**Store Keys:**
- Input:
  - `research_focus`: str - Research context
  - `citations`: List[Dict] - Citations to process
  - `llm_model`: str - Model to use (default: gpt-4)
- Output:
  - `llm_response`: str - Raw LLM response
  - `extracted_citations`: List[Dict] - Parsed citations
```

#### CitationValidationNode
```
**Store Keys:**
- Input:
  - `extracted_citations`: List[Dict] - Citations to validate
- Output:
  - `valid_citations`: List[Dict] - Validated citations
  - `validation_errors`: List[Dict] - Validation issues
```

#### CitationFormatterNode
```
**Store Keys:**
- Input:
  - `valid_citations`: List[Dict] - Citations to format
  - `format_style`: str - APA, MLA, Chicago, etc.
- Output:
  - `formatted_citations`: str - Markdown formatted output
  - `citation_summary`: Dict - Summary statistics
```

### Utility Classes Section (Lines 105-136)
Add citation-specific utilities:
- ScholarAPIClient
- OpenAIClient
- CitationCache
- CitationFormatter

### Common Store Keys Section (Lines 185-204)
Add citation-specific keys:
- `research_focus`: Research question/topic
- `citations`: Raw citation data
- `llm_prompt`: Generated prompt
- `llm_response`: LLM output
- `citation_format`: Output format preference

### Error Codes Section (Lines 206-221)
Add citation-specific errors:
- `CITATION_NOT_FOUND`: No citations match criteria
- `CITATION_INVALID_FORMAT`: Citation data malformed
- `LLM_RATE_LIMIT`: OpenAI rate limit hit
- `LLM_INVALID_RESPONSE`: LLM response unparseable

### Configuration Section (Lines 223-248)
Add citation-specific config:
```bash
# Citation APIs
SCHOLAR_API_KEY=...
OPENAI_API_KEY=sk-...

# Citation Processing
MAX_CITATIONS_PER_REQUEST=20
CITATION_CACHE_TTL=3600
LLM_MODEL=gpt-4
LLM_MAX_RETRIES=3
```

## 5. New Documentation Needed

### docs/citation-integration-guide.md
Create a comprehensive guide covering:
- Citation data sources and APIs
- LLM prompt engineering for citations
- Citation validation rules
- Output formatting options
- Performance optimization
- Common issues and solutions

### docs/implementation-log.md
Document the pivot from traditional agent to citation-focused tool:
- Original vision vs new approach
- Technical decisions made
- Architecture changes
- Lessons learned

## Implementation Priority

1. **High Priority** (Core functionality):
   - api-reference.md: Add new node documentation
   - flow-design.md: Add citation flow example
   - Create citation-integration-guide.md

2. **Medium Priority** (Architecture clarity):
   - architecture.md: Add citation service layer
   - developer-guide.md: Update examples to citation focus

3. **Low Priority** (Context/history):
   - Create implementation-log.md

## Next Steps

1. Update each documentation file according to the changes outlined above
2. Ensure all examples use citation-based scenarios instead of generic ones
3. Validate that documentation aligns with actual implementation
4. Add diagrams where helpful (especially in architecture.md)
5. Review for consistency across all documentation