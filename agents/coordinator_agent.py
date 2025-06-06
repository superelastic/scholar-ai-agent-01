"""Academic Coordinator Agent implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pocketflow import AsyncFlow

from nodes import (
    UserInputNode,
    PaperAnalysisNode,
    CitationSearchNode,
    ResearchSynthesisNode,
    PresentationNode,
    CoordinatorNodeError
)
from utils import (
    PDFExtractorUtility,
    LLMAnalysisUtility,
    ScholarSearchUtility
)

logger = logging.getLogger(__name__)


class AcademicCoordinatorAgent:
    """Main orchestrator managing the academic research workflow.
    
    Coordinates the analysis of seminal papers, citation searches,
    and research direction synthesis through specialized nodes.
    """

    def __init__(self, store: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize the Academic Coordinator Agent.
        
        Args:
            store: Shared state store for agent communication
            config: Optional configuration parameters
        """
        self.store = store
        self.config = config or {}
        self.name = "AcademicCoordinatorAgent"
        
        # Initialize utilities
        self.pdf_extractor = PDFExtractorUtility(
            cache_dir=self.config.get('cache_dir', './cache'),
            timeout_seconds=self.config.get('pdf_timeout', 5)
        )
        
        self.llm_utility = LLMAnalysisUtility(
            openai_api_key=self.config.get('openai_api_key'),
            anthropic_api_key=self.config.get('anthropic_api_key'),
            model=self.config.get('llm_model', 'gpt-4'),
            timeout=self.config.get('llm_timeout', 30)
        )
        
        self.scholar_utility = ScholarSearchUtility(
            cache_ttl_hours=self.config.get('scholar_cache_ttl', 24),
            min_request_interval=self.config.get('scholar_interval', 2.0),
            max_retries=self.config.get('scholar_retries', 3)
        )
        
        # Initialize nodes
        self.user_input_node = UserInputNode(self.pdf_extractor)
        self.paper_analysis_node = PaperAnalysisNode(self.pdf_extractor, self.llm_utility)
        self.citation_search_node = CitationSearchNode(self.scholar_utility, self.llm_utility)
        self.research_synthesis_node = ResearchSynthesisNode(self.llm_utility)
        self.presentation_node = PresentationNode(
            supported_formats=self.config.get('output_formats', ['json', 'markdown', 'html', 'txt'])
        )
        
        logger.info(f"Initialized {self.name} with all nodes and utilities")

    def create_flow(self) -> AsyncFlow:
        """Create the main coordination flow.
        
        Returns:
            Flow object defining the coordination workflow
        """
        # For now, create a simple flow structure
        # The actual implementation will depend on PocketFlow's specific API
        flow = AsyncFlow()
        logger.info(f"Created flow for {self.name}")
        return flow

    def initialize_store(self) -> Dict[str, Any]:
        """Initialize the shared store with default values.
        
        Returns:
            Initialized store dictionary
        """
        return {
            'paper_metadata': {},
            'paper_content': '',
            'paper_sections': {},
            'paper_references': [],
            'analysis_results': {},
            'search_queries': [],
            'citation_results': {},
            'filtered_citations': [],
            'formatted_citations': {},
            'research_suggestions': {},
            'final_results': {},
            'presentation': {},
            'status': 'initialized',
            'progress': 0,
            'errors': [],
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the coordination workflow.
        
        Args:
            input_data: Input containing PDF path and user preferences
            
        Returns:
            Analysis results including paper analysis, citations, and research directions
        """
        logger.info(f"{self.name} starting workflow")
        
        # Initialize store with input data
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
            logger.info("Starting User Input Node...")
            result = await self.user_input_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'User input failed'))
            
            logger.info("Starting Paper Analysis Node...")
            result = await self.paper_analysis_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Paper analysis failed'))
            
            logger.info("Starting Citation Search Node...")
            result = await self.citation_search_node.process(context)
            if not result.get('success'):
                # Citation search failure is not critical, continue with limited data
                logger.warning(f"Citation search failed: {result.get('error')}")
                self.store['citation_search_error'] = result.get('error')
            
            logger.info("Starting Research Synthesis Node...")
            result = await self.research_synthesis_node.process(context)
            if not result.get('success'):
                # Research synthesis failure is not critical, continue
                logger.warning(f"Research synthesis failed: {result.get('error')}")
                self.store['research_synthesis_error'] = result.get('error')
            
            logger.info("Starting Presentation Node...")
            result = await self.presentation_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Presentation failed'))
            
            # Return successful results
            return {
                'success': True,
                'status': self.store.get('status', 'completed'),
                'results': self.store.get('final_results', {}),
                'presentation': self.store.get('presentation', {}),
                'progress': 100,
                'processing_time': self._get_processing_time(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
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
            'progress': self.store.get('progress', 0) if self.store else 0,
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
        """Get current workflow status.
        
        Returns:
            Status dictionary
        """
        if not self.store:
            return {'status': 'not_started', 'progress': 0}
        
        return {
            'status': self.store.get('status', 'unknown'),
            'progress': self.store.get('progress', 0),
            'last_updated': self.store.get('last_updated'),
            'error': self.store.get('error'),
            'processing_time': self._get_processing_time()
        }