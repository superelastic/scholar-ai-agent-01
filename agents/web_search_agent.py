"""Academic Web Search Agent implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pocketflow import AsyncFlow

from nodes import (
    SearchQueryNode,
    GoogleScholarNode,
    CitationFilterNode,
    CitationFormatterNode,
    WebSearchNodeError
)
from utils import ScholarSearchUtility, LLMAnalysisUtility

logger = logging.getLogger(__name__)


class AcademicWebSearchAgent:
    """Specialized agent for academic citation search and filtering.
    
    Handles search query formulation, Google Scholar interactions,
    result filtering, and citation formatting.
    """

    def __init__(self, store: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize the Academic Web Search Agent.
        
        Args:
            store: Shared state store for agent communication
            config: Optional configuration parameters
        """
        self.store = store
        self.config = config or {}
        self.name = "AcademicWebSearchAgent"
        
        # Agent-specific configuration
        self.max_results = self.config.get('max_results', 20)
        self.year_filter = self.config.get('year_filter', 3)  # Last 3 years
        self.relevance_threshold = self.config.get('relevance_threshold', 0.5)
        
        # Initialize utilities
        self.scholar_utility = ScholarSearchUtility(
            cache_ttl_hours=self.config.get('scholar_cache_ttl', 24),
            min_request_interval=self.config.get('scholar_interval', 2.0),
            max_retries=self.config.get('scholar_retries', 3)
        )
        
        self.llm_utility = LLMAnalysisUtility(
            openai_api_key=self.config.get('openai_api_key'),
            anthropic_api_key=self.config.get('anthropic_api_key'),
            model=self.config.get('llm_model', 'gpt-4'),
            timeout=self.config.get('llm_timeout', 30)
        )
        
        # Initialize nodes
        self.search_query_node = SearchQueryNode(self.llm_utility)
        self.google_scholar_node = GoogleScholarNode(self.scholar_utility)
        self.citation_filter_node = CitationFilterNode()
        self.citation_formatter_node = CitationFormatterNode()
        
        logger.info(f"Initialized {self.name} with all nodes and utilities")

    def create_flow(self) -> AsyncFlow:
        """Create the web search flow.
        
        Returns:
            Flow object defining the web search workflow
        """
        # For now, create a simple flow structure
        # The actual implementation will depend on PocketFlow's specific API
        flow = AsyncFlow()
        logger.info(f"Created flow for {self.name}")
        return flow

    def initialize_store(self) -> Dict[str, Any]:
        """Initialize the agent-specific store with default values.
        
        Returns:
            Initialized store dictionary
        """
        return {
            'paper_metadata': {},
            'search_queries': [],
            'raw_results': [],
            'filtered_results': [],
            'formatted_citations': {},
            'status': 'initialized',
            'errors': [],
            'retry_count': 0,
            'search_stats': {
                'total_queries': 0,
                'total_results': 0,
                'filtered_count': 0,
                'processing_time': 0
            },
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the web search workflow.
        
        Args:
            input_data: Input containing paper metadata and search preferences
            
        Returns:
            Search results including filtered citations and summary statistics
        """
        logger.info(f"{self.name} starting citation search workflow")
        
        # Initialize agent store if needed
        if not self.store:
            self.store = self.initialize_store()
        
        # Create execution context
        context = {
            'store': self.store,
            'input_data': input_data,
            'config': self.config
        }
        
        try:
            # Execute workflow step by step
            logger.info("Starting Search Query Node...")
            result = await self.search_query_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Query generation failed'))
            
            logger.info("Starting Google Scholar Node...")
            result = await self.google_scholar_node.process(context)
            if not result.get('success'):
                # Retry logic is handled within the node
                if self.store.get('retry_count', 0) >= 3:
                    return self._create_error_response(result.get('error', 'Scholar search failed after retries'))
                # Otherwise, continue with empty results
                logger.warning(f"Scholar search failed: {result.get('error')}")
                self.store['scholar_search_error'] = result.get('error')
            
            logger.info("Starting Citation Filter Node...")
            result = await self.citation_filter_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Citation filtering failed'))
            
            logger.info("Starting Citation Formatter Node...")
            result = await self.citation_formatter_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Citation formatting failed'))
            
            # Return successful results
            return {
                'success': True,
                'status': self.store.get('status', 'completed'),
                'citations': self.store.get('formatted_citations', {}),
                'stats': self.store.get('search_stats', {}),
                'processing_time': self._get_processing_time(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Web search workflow failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        if self.store:
            self.store.update({
                'status': 'error',
                'error': error_message,
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'success': False,
            'status': 'error',
            'error': error_message,
            'citations': {},
            'stats': self.store.get('search_stats', {}) if self.store else {},
            'timestamp': datetime.now().isoformat()
        }

    def _get_processing_time(self) -> Optional[float]:
        """Get total processing time.
        
        Returns:
            Processing time in seconds or None
        """
        if not self.store:
            return None
            
        try:
            start_time = self.store.get('started_at')
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                return (datetime.now() - start_dt).total_seconds()
        except Exception:
            pass
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current search status.
        
        Returns:
            Status dictionary
        """
        if not self.store:
            return {'status': 'not_started', 'stats': {}}
        
        return {
            'status': self.store.get('status', 'unknown'),
            'stats': self.store.get('search_stats', {}),
            'last_updated': self.store.get('last_updated'),
            'error': self.store.get('error'),
            'processing_time': self._get_processing_time()
        }