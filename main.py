"""Main entry point for Scholar AI Agent."""

import asyncio
import logging
from pathlib import Path

# Store will be a simple dict for state management

from agents import (
    AcademicCoordinatorAgent,
    AcademicNewResearchAgent,
    AcademicWebSearchAgent,
)
from config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure application logging.
    
    Args:
        settings: Application settings
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format
    )


async def main():
    """Main application entry point."""
    # Initialize settings
    settings = Settings()
    setup_logging(settings)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Scholar AI Agent")
    
    # Initialize shared store as a dict
    store = {}
    
    # Initialize agents
    coordinator = AcademicCoordinatorAgent(store, config={
        "upload_dir": settings.upload_dir,
        "timeout": settings.total_workflow_timeout
    })
    
    web_search = AcademicWebSearchAgent(store, config={
        "max_results": settings.scholar_max_results,
        "year_filter": settings.scholar_year_filter
    })
    
    research_synthesis = AcademicNewResearchAgent(store, config={
        "min_confidence": 0.7
    })
    
    logger.info("All agents initialized successfully")
    
    # Example usage (will be replaced with actual implementation)
    result = await coordinator.run({
        "pdf_path": "example.pdf",
        "user_preferences": {}
    })
    
    logger.info(f"Workflow completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())