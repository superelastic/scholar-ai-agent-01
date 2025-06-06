"""Main coordination module for Scholar AI Agent system.

This module provides the main orchestration logic that coordinates the three agents
through the shared state management system.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents import (
    AcademicCoordinatorAgent,
    AcademicWebSearchAgent,
    AcademicNewResearchAgent
)
from utils import SharedStore, AgentCommunicator

logger = logging.getLogger(__name__)


class ScholarAICoordinator:
    """Main coordinator that orchestrates the Scholar AI workflow.
    
    Manages the shared state and coordinates communication between the three agents:
    - Academic Coordinator Agent
    - Academic Web Search Agent  
    - Academic New Research Agent
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, persistence_dir: Optional[str] = None):
        """Initialize the Scholar AI Coordinator.
        
        Args:
            config: Configuration dictionary for agents and utilities
            persistence_dir: Directory for persisting session data
        """
        self.config = config or {}
        self.persistence_dir = persistence_dir
        
        # Initialize shared store
        self.shared_store = SharedStore(persistence_dir=persistence_dir)
        
        # Initialize agents with shared store
        self._initialize_agents()
        
        # Initialize communicators for each agent
        self.coordinator_comm = AgentCommunicator(self.shared_store, 'coordinator')
        self.web_search_comm = AgentCommunicator(self.shared_store, 'web_search')
        self.research_synthesis_comm = AgentCommunicator(self.shared_store, 'research_synthesis')
        
        logger.info("Initialized Scholar AI Coordinator with all agents")
    
    def _initialize_agents(self) -> None:
        """Initialize all three agents with proper configuration."""
        # Academic Coordinator Agent
        self.coordinator_agent = AcademicCoordinatorAgent(
            store={},  # Will use shared store through communicator
            config=self.config.get('coordinator', {})
        )
        
        # Academic Web Search Agent
        self.web_search_agent = AcademicWebSearchAgent(
            store={},  # Will use shared store through communicator
            config=self.config.get('web_search', {})
        )
        
        # Academic New Research Agent
        self.research_synthesis_agent = AcademicNewResearchAgent(
            store={},  # Will use shared store through communicator
            config=self.config.get('research_synthesis', {})
        )
        
        logger.info("Initialized all three agents")
    
    async def process_paper(self, pdf_path: str, preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process an academic paper through the complete workflow.
        
        Args:
            pdf_path: Path to the PDF file
            preferences: User preferences for processing
            
        Returns:
            Complete analysis results including citations and research directions
        """
        logger.info(f"Starting paper processing for: {pdf_path}")
        start_time = datetime.now()
        
        try:
            # Update session preferences
            if preferences:
                self.shared_store.update_store('session_store', {
                    'preferences': preferences
                })
            
            # Step 1: Coordinator Agent - PDF processing and initial analysis
            self.coordinator_comm.update_workflow_stage('pdf_processing')
            self.coordinator_comm.update_agent_status('active')
            self.coordinator_comm.add_status_update("Starting PDF processing and analysis")
            
            coordinator_result = await self._run_coordinator_agent(pdf_path)
            
            if not coordinator_result['success']:
                raise Exception(f"Coordinator agent failed: {coordinator_result.get('error')}")
            
            # Extract paper data from coordinator results
            paper_store = self.shared_store.get_store('paper_store')
            
            # Step 2: Web Search Agent - Citation search
            self.coordinator_comm.update_workflow_stage('citation_search')
            self.web_search_comm.update_agent_status('active')
            self.coordinator_comm.add_status_update("Searching for citations")
            
            web_search_result = await self._run_web_search_agent(paper_store)
            
            # Continue even if citation search fails (graceful degradation)
            if not web_search_result['success']:
                logger.warning(f"Citation search failed: {web_search_result.get('error')}")
                self.coordinator_comm.add_status_update(
                    "Citation search encountered issues, continuing with available data", 
                    level='warning'
                )
            
            # Step 3: Research Synthesis Agent - Generate research directions
            self.coordinator_comm.update_workflow_stage('synthesis')
            self.research_synthesis_comm.update_agent_status('active')
            self.coordinator_comm.add_status_update("Synthesizing research directions")
            
            synthesis_result = await self._run_research_synthesis_agent()
            
            if not synthesis_result['success']:
                logger.warning(f"Research synthesis failed: {synthesis_result.get('error')}")
                self.coordinator_comm.add_status_update(
                    "Research synthesis encountered issues", 
                    level='warning'
                )
            
            # Step 4: Final formatting and presentation
            self.coordinator_comm.update_workflow_stage('formatting')
            self.coordinator_comm.add_status_update("Formatting final results")
            
            final_results = self._compile_final_results()
            
            # Update performance metrics
            total_time = (datetime.now() - start_time).total_seconds()
            self.coordinator_comm.update_performance_metric('total_time', total_time)
            
            # Mark workflow as completed
            self.coordinator_comm.update_workflow_stage('completed')
            self.coordinator_comm.update_agent_status('completed')
            self.web_search_comm.update_agent_status('completed')
            self.research_synthesis_comm.update_agent_status('completed')
            
            self.coordinator_comm.add_status_update(f"Analysis completed in {total_time:.1f} seconds")
            
            return {
                'success': True,
                'results': final_results,
                'performance': self.shared_store.get_store('session_store')['performance_metrics'],
                'session_id': self.shared_store.get_store('session_store')['session_id']
            }
            
        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(error_msg)
            
            self.coordinator_comm.log_error(error_msg, 'workflow_error')
            self.coordinator_comm.update_workflow_stage('error')
            
            return {
                'success': False,
                'error': error_msg,
                'partial_results': self._compile_partial_results()
            }
    
    async def _run_coordinator_agent(self, pdf_path: str) -> Dict[str, Any]:
        """Run the Academic Coordinator Agent.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Coordinator agent results
        """
        try:
            # Run coordinator agent
            result = await self.coordinator_agent.run({'pdf_path': pdf_path})
            
            if result.get('success'):
                # Update shared stores with coordinator results
                paper_data = result.get('results', {})
                
                # Update paper store
                self.shared_store.update_store('paper_store', {
                    'title': paper_data.get('paper_metadata', {}).get('title', ''),
                    'authors': paper_data.get('paper_metadata', {}).get('authors', []),
                    'year': paper_data.get('paper_metadata', {}).get('year'),
                    'abstract': paper_data.get('paper_metadata', {}).get('abstract', ''),
                    'key_concepts': paper_data.get('paper_analysis', {}).get('key_concepts', []),
                    'methodology': paper_data.get('paper_analysis', {}).get('methodology', ''),
                    'findings': paper_data.get('paper_analysis', {}).get('findings', []),
                    'pdf_path': pdf_path,
                    'pdf_content': paper_data.get('paper_content', ''),
                    'extracted_at': datetime.now().isoformat(),
                    'sections': paper_data.get('paper_sections', {}),
                    'references': paper_data.get('paper_references', []),
                    'analysis_complete': True
                })
                
                # Update performance metric
                if 'processing_time' in result:
                    self.coordinator_comm.update_performance_metric(
                        'pdf_processing_time', 
                        result['processing_time']
                    )
            
            return result
            
        except Exception as e:
            error_msg = f"Coordinator agent error: {str(e)}"
            logger.error(error_msg)
            self.coordinator_comm.log_error(error_msg, 'coordinator_error')
            return {'success': False, 'error': error_msg}
    
    async def _run_web_search_agent(self, paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Academic Web Search Agent.
        
        Args:
            paper_metadata: Paper metadata for citation search
            
        Returns:
            Web search agent results
        """
        try:
            # Request citation search
            self.coordinator_comm.request_citation_search(paper_metadata)
            
            # Run web search agent
            result = await self.web_search_agent.run({'paper_metadata': paper_metadata})
            
            if result.get('success'):
                # Update citation store with results
                citations_data = result.get('citations', {})
                
                self.shared_store.update_store('citation_store', {
                    'citing_papers': citations_data.get('citations', []),
                    'total_found': citations_data.get('summary', {}).get('total_count', 0),
                    'filtered_count': len(citations_data.get('citations', [])),
                    'search_queries': citations_data.get('metadata', {}).get('queries_used', []),
                    'search_metadata': {
                        'sources': ['Google Scholar'],
                        'search_date': datetime.now().isoformat(),
                        'filter_criteria': citations_data.get('metadata', {}).get('filter_settings', {})
                    },
                    'search_status': 'completed'
                })
                
                # Update performance metric
                if 'processing_time' in result:
                    self.web_search_comm.update_performance_metric(
                        'citation_search_time', 
                        result['processing_time']
                    )
            
            return result
            
        except Exception as e:
            error_msg = f"Web search agent error: {str(e)}"
            logger.error(error_msg)
            self.web_search_comm.log_error(error_msg, 'search_error')
            return {'success': False, 'error': error_msg}
    
    async def _run_research_synthesis_agent(self) -> Dict[str, Any]:
        """Run the Academic New Research Agent.
        
        Returns:
            Research synthesis agent results
        """
        try:
            # Prepare synthesis data from stores
            paper_store = self.shared_store.get_store('paper_store')
            citation_store = self.shared_store.get_store('citation_store')
            
            synthesis_input = {
                'paper_metadata': {
                    'title': paper_store['title'],
                    'authors': paper_store['authors'],
                    'year': paper_store['year'],
                    'key_concepts': paper_store['key_concepts']
                },
                'paper_analysis': {
                    'key_concepts': paper_store['key_concepts'],
                    'methodology': paper_store['methodology'],
                    'findings': paper_store['findings']
                },
                'citation_data': citation_store['citing_papers']
            }
            
            # Request research synthesis
            self.coordinator_comm.request_research_synthesis(synthesis_input)
            
            # Run synthesis agent
            result = await self.research_synthesis_agent.run(synthesis_input)
            
            if result.get('success'):
                # Update research store with results
                suggestions_data = result.get('suggestions', {})
                
                self.shared_store.update_store('research_store', {
                    'suggestions': suggestions_data.get('research_suggestions', []),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'synthesis_data': {
                        'trends': suggestions_data.get('synthesis_summary', {}).get('trend_overview', {}),
                        'gaps': suggestions_data.get('synthesis_summary', {}).get('gap_analysis', {}),
                        'opportunities': suggestions_data.get('impact_assessment', {})
                    },
                    'confidence_scores': {
                        f"suggestion_{i}": s.get('confidence', 0) 
                        for i, s in enumerate(suggestions_data.get('research_suggestions', []))
                    },
                    'implementation_roadmap': suggestions_data.get('implementation_roadmap', {}),
                    'synthesis_status': 'completed'
                })
                
                # Update performance metric
                if 'processing_time' in result:
                    self.research_synthesis_comm.update_performance_metric(
                        'synthesis_time', 
                        result['processing_time']
                    )
            
            return result
            
        except Exception as e:
            error_msg = f"Research synthesis agent error: {str(e)}"
            logger.error(error_msg)
            self.research_synthesis_comm.log_error(error_msg, 'synthesis_error')
            return {'success': False, 'error': error_msg}
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile final results from all stores.
        
        Returns:
            Complete analysis results
        """
        paper_store = self.shared_store.get_store('paper_store')
        citation_store = self.shared_store.get_store('citation_store')
        research_store = self.shared_store.get_store('research_store')
        session_store = self.shared_store.get_store('session_store')
        
        return {
            'paper_analysis': {
                'metadata': {
                    'title': paper_store['title'],
                    'authors': paper_store['authors'],
                    'year': paper_store['year']
                },
                'analysis': {
                    'key_concepts': paper_store['key_concepts'],
                    'methodology': paper_store['methodology'],
                    'findings': paper_store['findings']
                }
            },
            'citations': {
                'summary': {
                    'total_found': citation_store['total_found'],
                    'filtered_count': citation_store['filtered_count']
                },
                'papers': citation_store['citing_papers']
            },
            'research_directions': {
                'suggestions': research_store['suggestions'],
                'synthesis_summary': research_store['synthesis_data'],
                'implementation_roadmap': research_store['implementation_roadmap']
            },
            'session_info': {
                'session_id': session_store['session_id'],
                'preferences': session_store['preferences'],
                'status_updates': session_store['status_updates'][-10:]  # Last 10 updates
            }
        }
    
    def _compile_partial_results(self) -> Dict[str, Any]:
        """Compile partial results in case of failure.
        
        Returns:
            Available partial results
        """
        try:
            return self._compile_final_results()
        except Exception:
            # Return whatever is available
            return {
                'paper_store': self.shared_store.get_store('paper_store'),
                'citation_store': self.shared_store.get_store('citation_store'),
                'research_store': self.shared_store.get_store('research_store')
            }
    
    def save_session(self, file_path: Optional[str] = None) -> str:
        """Save the current session.
        
        Args:
            file_path: Optional path for session file
            
        Returns:
            Path to saved session file
        """
        return self.shared_store.save_session(file_path)
    
    def load_session(self, file_path: str) -> None:
        """Load a saved session.
        
        Args:
            file_path: Path to session file
        """
        self.shared_store.load_session(file_path)
        logger.info(f"Loaded session from {file_path}")
    
    def export_results(self, format: str = 'json', file_path: Optional[str] = None) -> str:
        """Export results in specified format.
        
        Args:
            format: Export format (json, markdown)
            file_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        if format == 'json':
            return self.shared_store.export_as_json(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status.
        
        Returns:
            Current status information
        """
        workflow_store = self.shared_store.get_store('workflow_store')
        session_store = self.shared_store.get_store('session_store')
        
        return {
            'current_stage': workflow_store['current_stage'],
            'completed_stages': workflow_store['completed_stages'],
            'agent_status': workflow_store['agent_status'],
            'performance_metrics': session_store['performance_metrics'],
            'last_activity': session_store['last_activity'],
            'errors': session_store['error_log']
        }