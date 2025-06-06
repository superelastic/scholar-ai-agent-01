"""Test project setup and basic agent initialization."""

import pytest

from agents import (
    AcademicCoordinatorAgent,
    AcademicNewResearchAgent,
    AcademicWebSearchAgent,
)
from config import Settings


def test_settings_initialization():
    """Test that settings can be initialized."""
    settings = Settings()
    assert settings.upload_dir == "uploads"
    assert settings.max_file_size_mb == 50
    assert settings.llm_model == "gpt-4"


def test_coordinator_agent_initialization():
    """Test coordinator agent can be initialized."""
    store = {}
    agent = AcademicCoordinatorAgent(store)
    assert agent.name == "AcademicCoordinatorAgent"
    assert agent.store == store


def test_web_search_agent_initialization():
    """Test web search agent can be initialized."""
    store = {}
    config = {"max_results": 10, "year_filter": 3}
    agent = AcademicWebSearchAgent(store, config)
    assert agent.name == "AcademicWebSearchAgent"
    assert agent.max_results == 10
    assert agent.year_filter == 3


def test_research_synthesis_agent_initialization():
    """Test research synthesis agent can be initialized."""
    store = {}
    config = {"min_confidence": 0.8}
    agent = AcademicNewResearchAgent(store, config)
    assert agent.name == "AcademicNewResearchAgent"
    assert agent.min_confidence == 0.8


@pytest.mark.asyncio
async def test_coordinator_agent_run():
    """Test coordinator agent run method."""
    store = {}
    agent = AcademicCoordinatorAgent(store)
    result = await agent.run({"pdf_path": "test.pdf"})
    # Agent now actually processes, so non-existent file should cause error
    assert result["success"] is False
    assert "error" in result