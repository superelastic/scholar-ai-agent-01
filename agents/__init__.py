"""Academic research agents module."""

from agents.coordinator_agent import AcademicCoordinatorAgent
from agents.web_search_agent import AcademicWebSearchAgent
from agents.research_synthesis_agent import AcademicNewResearchAgent

__all__ = [
    "AcademicCoordinatorAgent",
    "AcademicWebSearchAgent",
    "AcademicNewResearchAgent",
]