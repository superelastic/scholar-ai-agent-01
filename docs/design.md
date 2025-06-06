# Product Requirements Document - Citation-Based API Design

## Project Overview
**Project Name:** Scholar AI Agent  
**Type:** Academic Research Tool  
**Purpose:** AI-powered academic research assistant that analyzes papers through citation metadata, discovers citing works with abstracts, and suggests future research directions using academic APIs

## Product Vision
Create an intelligent academic research tool using PocketFlow's agentic architecture that helps researchers understand papers, discover citations with full abstracts, and identify promising research directions through API-based analysis and a coordinated multi-agent system.

## 🔄 Major Design Change
**From**: PDF upload and parsing approach  
**To**: Citation-based API approach with abstract analysis  
**Rationale**: Improved reliability, structured data access, and richer citation context through abstracts

## User Stories

### As a Researcher
1. I want to provide citation information (title, authors, year) and receive comprehensive analysis
2. I want to discover papers that cite my work with full abstract analysis
3. I want to see what references my paper cites (bibliography analysis)
4. I want AI-generated research directions based on abstract-level understanding
5. I want the analysis presented in a clear, structured format with export options

### As a Developer
1. I want to implement the three-agent system using reliable APIs
2. I want structured JSON responses instead of HTML parsing
3. I want robust error handling for API failures
4. I want comprehensive testing without PDF parsing complexity

### As an AI Assistant
1. I need clear patterns for API integration and data aggregation
2. I need to understand multi-API coordination strategies
3. I need abstract analysis patterns for research synthesis
4. I need testing approaches for API-based workflows

## Functional Requirements

### Academic Research System Architecture

1. **Agent System Design**
   - **Academic Coordinator Agent**: Main orchestrator managing the research workflow
   - **Academic Web Search Agent**: Specialized for finding citing papers via APIs with abstract retrieval
   - **Academic New Research Agent**: Synthesizes abstracts and metadata to suggest research directions

2. **Node Architecture**
   
   **Coordinator Flow Nodes:**
   - `UserInputNode`: Receives citation info (title, authors, year, DOI optional)
   - `PaperIdentificationNode`: Uses APIs to find and verify the paper
   - `MetadataEnrichmentNode`: Gathers comprehensive metadata from multiple APIs
   - `CitationSearchNode`: Delegates to Web Search Agent for citations
   - `ReferenceSearchNode`: Retrieves papers referenced by the target paper
   - `ResearchSynthesisNode`: Delegates to Research Agent for future directions
   - `PresentationNode`: Formats and presents all results to user
   
   **Web Search Flow Nodes:**
   - `APIQueryNode`: Formulates API queries for multiple services
   - `SerpAPINode`: Interfaces with SerpAPI for Google Scholar data
   - `SemanticScholarNode`: Gets additional metadata and references
   - `AbstractAnalysisNode`: Extracts key information from abstracts using LLM
   - `CitationEnrichmentNode`: Combines data from multiple sources
   
   **Research Synthesis Flow Nodes:**
   - `AbstractSynthesisNode`: Analyzes abstracts of citing papers
   - `TrendAnalysisNode`: Identifies research trends from abstract content
   - `GapIdentificationNode`: Finds gaps mentioned in abstracts
   - `DirectionGeneratorNode`: Proposes research based on abstract analysis
   - `SuggestionFormatterNode`: Formats suggestions with evidence

3. **API Integration Strategy**
   
   **Primary APIs:**
   - **SerpAPI**: Google Scholar search, citations, abstracts
   - **Semantic Scholar API**: References, detailed metadata, citation contexts
   - **CrossRef API**: DOI resolution, additional metadata
   - **OpenAlex API**: Comprehensive academic database (backup)
   
   **Data Flow:**
   ```
   User Input → Paper Identification → Multi-API Enrichment → Analysis
       ↓              ↓                      ↓                  ↓
   Title/Author    SerpAPI            Semantic Scholar      LLM Analysis
   Year/DOI        Find paper         Get references        of abstracts
                   Get citations      Get contexts
   ```

4. **State Management**
   - **Paper Store**: Verified metadata, abstract, key concepts
   - **Citation Store**: Citing papers with full abstracts, relevance scores
   - **Reference Store**: Papers cited by target work with abstracts
   - **Analysis Store**: Extracted concepts, trends, gaps from abstracts
   - **Research Store**: Suggested directions with abstract-based evidence

### Technical Components

1. **External Integrations**
   - **SerpAPI**: Primary citation search with abstracts (requires API key)
   - **Semantic Scholar API**: References and detailed metadata (free tier)
   - **CrossRef API**: DOI-based metadata (free)
   - **OpenAlex API**: Backup data source (free)
   - **LLM Integration**: Abstract analysis and synthesis

2. **Utility Services**
   - `CitationResolverUtility`: Multi-API paper identification
   - `AbstractAnalysisUtility`: LLM-based abstract information extraction
   - `APIAggregatorUtility`: Combines data from multiple sources
   - `ResearchSynthesisUtility`: Analyzes patterns across abstracts
   - `FormatterUtility`: Consistent output formatting

3. **Abstract Analysis Prompts**
   - **Concept Extraction**: Extract key concepts, methods, findings from abstract
   - **Gap Analysis**: Identify limitations and future work mentioned
   - **Methodology Extraction**: Understand approaches used
   - **Trend Synthesis**: Find patterns across multiple abstracts
   - **Direction Generation**: Propose research based on abstract evidence

### Data Enrichment Strategy

1. **From User Input to Rich Data**
   ```
   Input: "Attention is All You Need, Vaswani et al., 2017"
   ↓
   Step 1: SerpAPI finds paper + cluster ID
   Step 2: Get 20+ citing papers with abstracts
   Step 3: Semantic Scholar gets references + contexts  
   Step 4: LLM analyzes all abstracts for patterns
   Step 5: Generate research directions with evidence
   ```

2. **Abstract-Based Analysis Benefits**
   - Understand WHY papers cited the work
   - Identify specific techniques and improvements
   - Find mentioned limitations and challenges
   - Track methodology evolution
   - Generate evidence-based suggestions

## Non-Functional Requirements

### Performance
- API response aggregation within 5 seconds per source
- LLM abstract analysis: 2-3 seconds per abstract
- Total workflow completion: under 45 seconds
- Parallel API calls where possible

### Reliability
- Fallback to alternate APIs if primary fails
- Cache API responses for 7 days
- Graceful degradation if some APIs unavailable
- Retry logic with exponential backoff

### Cost Management
- Use free tiers where available
- Cache aggressively to minimize API calls
- Batch abstract analysis for efficiency
- Monitor API usage limits

### User Experience
- Real-time progress updates as APIs respond
- Clear indication of data sources
- Export results with full abstract context
- Highlight insights derived from abstracts

## Acceptance Criteria

### Core Functionality
- [ ] User can input citation info and get analysis within 45 seconds
- [ ] System retrieves 10+ citing papers with abstracts
- [ ] System retrieves referenced papers when available
- [ ] Abstract analysis provides specific insights
- [ ] Research suggestions include evidence from abstracts

### API Integration
- [ ] Multi-API coordination works seamlessly
- [ ] Fallback mechanisms handle API failures
- [ ] Data deduplication across sources works correctly
- [ ] Abstract extraction is reliable
- [ ] Rate limiting prevents API blocks

### Abstract Analysis Quality
- [ ] Key concepts extracted accurately from abstracts
- [ ] Limitations and gaps identified correctly
- [ ] Methodology patterns recognized
- [ ] Trends synthesized across multiple abstracts
- [ ] Research directions cite specific abstract evidence

## Success Metrics
1. Paper identification accuracy: >95%
2. Abstract retrieval rate: >80% of citations
3. Research suggestion quality: Evidence-based with abstract citations
4. API reliability: <5% failure rate with fallbacks
5. User satisfaction: >4.5/5 rating

## Data Structure Updates

### Enhanced Citation Structure
```python
{
    "citing_papers": List[{
        "title": str,
        "authors": List[str],
        "year": int,
        "venue": str,
        "url": str,
        "abstract": str,  # NEW: Full abstract
        "abstract_analysis": {  # NEW: LLM analysis
            "key_concepts": List[str],
            "methodology": str,
            "findings": List[str],
            "limitations": List[str],
            "future_work": List[str]
        },
        "relevance_score": float,
        "citation_context": str  # NEW: How it cites
    }],
    "total_found": int,
    "sources": List[str]  # APIs used
}
```

### Research Suggestion Structure
```python
{
    "suggestions": List[{
        "title": str,
        "description": str,
        "rationale": str,
        "confidence": float,
        "evidence_from_abstracts": List[{  # NEW
            "paper_title": str,
            "relevant_quote": str,
            "insight_type": str  # gap/method/trend
        }],
        "methodology_hints": List[str],  # NEW
        "potential_impact": str
    }],
    "trend_summary": Dict,  # NEW: Aggregated trends
    "gap_analysis": Dict    # NEW: Common gaps found
}
```

## Implementation Changes

### Removed Components
- PDF upload and parsing functionality
- PDFExtractorUtility
- File upload handling
- PDF validation logic

### New Components
- Citation input form/API endpoint
- Multi-API orchestration logic
- Abstract analysis pipeline
- API response caching layer
- Enhanced research synthesis with abstract evidence

### Modified Components
- UserInputNode → accepts citation info instead of PDF
- PaperAnalysisNode → uses API metadata instead of extracted text
- CitationSearchNode → enhanced with abstract retrieval
- ResearchSynthesisNode → analyzes abstracts instead of just metadata

## Migration Strategy

1. **Phase 1**: Add API integration alongside PDF support
2. **Phase 2**: Implement abstract analysis features
3. **Phase 3**: Migrate UI to citation-based input
4. **Phase 4**: Deprecate PDF parsing (keep as optional)

## Future Enhancements
1. Full-text analysis when available via APIs
2. Citation network visualization with abstract snippets
3. Trend tracking over time windows
4. Collaborative annotation of abstracts
5. API for programmatic access
6. Multi-language abstract support