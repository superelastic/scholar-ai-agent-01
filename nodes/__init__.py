"""Node implementations for Scholar AI Agent flows."""

from nodes.coordinator_nodes import (
    UserInputNode,
    PaperAnalysisNode,
    CitationSearchNode,
    ResearchSynthesisNode,
    PresentationNode,
    CoordinatorNodeError
)

from nodes.web_search_nodes import (
    SearchQueryNode,
    GoogleScholarNode,
    CitationFilterNode,
    CitationFormatterNode,
    WebSearchNodeError
)

from nodes.research_synthesis_nodes import (
    PaperSynthesisNode,
    TrendAnalysisNode,
    DirectionGeneratorNode,
    SuggestionFormatterNode,
    ResearchSynthesisNodeError
)

__all__ = [
    "UserInputNode",
    "PaperAnalysisNode", 
    "CitationSearchNode",
    "ResearchSynthesisNode",
    "PresentationNode",
    "CoordinatorNodeError",
    "SearchQueryNode",
    "GoogleScholarNode",
    "CitationFilterNode",
    "CitationFormatterNode",
    "WebSearchNodeError",
    "PaperSynthesisNode",
    "TrendAnalysisNode",
    "DirectionGeneratorNode",
    "SuggestionFormatterNode",
    "ResearchSynthesisNodeError"
]