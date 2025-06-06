"""Academic New Research Agent implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pocketflow import AsyncFlow

from nodes import (
    PaperSynthesisNode,
    TrendAnalysisNode,
    DirectionGeneratorNode,
    SuggestionFormatterNode,
    ResearchSynthesisNodeError
)
from utils import LLMAnalysisUtility

logger = logging.getLogger(__name__)


class AcademicNewResearchAgent:
    """Agent for synthesizing information to suggest research directions.
    
    Analyzes seminal papers and citations to identify trends,
    gaps, and propose novel research directions.
    """

    def __init__(self, store: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize the Academic New Research Agent.
        
        Args:
            store: Shared state store for agent communication
            config: Optional configuration parameters
        """
        self.store = store
        self.config = config or {}
        self.name = "AcademicNewResearchAgent"
        
        # Agent-specific configuration
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.max_suggestions = self.config.get("max_suggestions", 5)
        self.trend_analysis_depth = self.config.get("trend_analysis_depth", "comprehensive")
        
        # Initialize utilities
        self.llm_utility = LLMAnalysisUtility(
            openai_api_key=self.config.get('openai_api_key'),
            anthropic_api_key=self.config.get('anthropic_api_key'),
            model=self.config.get('llm_model', 'gpt-4'),
            timeout=self.config.get('llm_timeout', 30)
        )
        
        # Initialize nodes
        self.paper_synthesis_node = PaperSynthesisNode(self.llm_utility)
        self.trend_analysis_node = TrendAnalysisNode(self.llm_utility)
        self.direction_generator_node = DirectionGeneratorNode(self.llm_utility)
        self.suggestion_formatter_node = SuggestionFormatterNode()
        
        logger.info(f"Initialized {self.name} with all nodes and utilities")

    def create_flow(self) -> AsyncFlow:
        """Create the research synthesis flow.
        
        Returns:
            Flow object defining the synthesis workflow
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
            'paper_analysis': {},
            'citation_data': [],
            'identified_trends': [],
            'research_gaps': [],
            'suggested_directions': [],
            'formatted_suggestions': {},
            'status': 'initialized',
            'errors': [],
            'synthesis_stats': {
                'trends_identified': 0,
                'gaps_found': 0,
                'suggestions_generated': 0,
                'high_confidence_suggestions': 0,
                'processing_time': 0
            },
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research synthesis workflow.
        
        Args:
            input_data: Paper analysis and citation data
            
        Returns:
            Research suggestions with rationale and confidence
        """
        logger.info(f"{self.name} starting research synthesis workflow")
        
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
            logger.info("Starting Paper Synthesis Node...")
            result = await self.paper_synthesis_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Paper synthesis failed'))
            
            logger.info("Starting Trend Analysis Node...")
            result = await self.trend_analysis_node.process(context)
            if not result.get('success'):
                # Trend analysis failure is not critical, continue with basic analysis
                logger.warning(f"Trend analysis failed: {result.get('error')}")
                self.store['trend_analysis_error'] = result.get('error')
            
            logger.info("Starting Direction Generator Node...")
            result = await self.direction_generator_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Direction generation failed'))
            
            logger.info("Starting Suggestion Formatter Node...")
            result = await self.suggestion_formatter_node.process(context)
            if not result.get('success'):
                return self._create_error_response(result.get('error', 'Suggestion formatting failed'))
            
            # Return successful results
            return {
                'success': True,
                'status': self.store.get('status', 'completed'),
                'suggestions': self.store.get('formatted_suggestions', {}),
                'stats': self.store.get('synthesis_stats', {}),
                'processing_time': self._get_processing_time(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Research synthesis workflow failed: {str(e)}"
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
            'suggestions': {},
            'stats': self.store.get('synthesis_stats', {}) if self.store else {},
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
        """Get current synthesis status.
        
        Returns:
            Status dictionary
        """
        if not self.store:
            return {'status': 'not_started', 'stats': {}}
        
        return {
            'status': self.store.get('status', 'unknown'),
            'stats': self.store.get('synthesis_stats', {}),
            'last_updated': self.store.get('last_updated'),
            'error': self.store.get('error'),
            'processing_time': self._get_processing_time()
        }