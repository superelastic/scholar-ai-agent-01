"""Node implementations for the Academic Coordinator Agent flow."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pocketflow import AsyncNode

from utils import (
    PDFExtractorUtility, 
    PDFExtractionError,
    LLMAnalysisUtility,
    LLMAnalysisError,
    ScholarSearchUtility,
    ScholarSearchError,
    FormatterFactory,
    FormatterError
)

logger = logging.getLogger(__name__)


class CoordinatorNodeError(Exception):
    """Base exception for coordinator node errors."""
    pass


class UserInputNode(AsyncNode):
    """Node for handling PDF upload and validation."""
    
    def __init__(self, pdf_extractor: PDFExtractorUtility):
        """Initialize the user input node.
        
        Args:
            pdf_extractor: PDF extraction utility instance
        """
        super().__init__()
        self.pdf_extractor = pdf_extractor
        self.name = "UserInputNode"
    
    async def prepare(self, context: Dict[str, Any]) -> None:
        """Prepare node for execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Preparing {self.name}")
        context.setdefault('node_start_time', datetime.now())
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and validate PDF.
        
        Args:
            context: Execution context containing input data
            
        Returns:
            Processing result
        """
        logger.info(f"Processing {self.name}")
        
        try:
            # Get input data
            input_data = context.get('input_data', {})
            pdf_path = input_data.get('pdf_path')
            
            if not pdf_path:
                raise CoordinatorNodeError("No PDF path provided")
            
            # Validate PDF exists
            if not Path(pdf_path).exists():
                raise CoordinatorNodeError(f"PDF file not found: {pdf_path}")
            
            # Validate PDF format and content
            is_valid, error_message = self.pdf_extractor.validate_pdf(pdf_path)
            if not is_valid:
                raise CoordinatorNodeError(f"Invalid PDF: {error_message}")
            
            # Update store
            store = context.get('store', {})
            store.update({
                'pdf_path': pdf_path,
                'status': 'pdf_validated',
                'progress': 10,
                'last_updated': datetime.now().isoformat()
            })
            
            logger.info(f"PDF validation successful: {pdf_path}")
            return {'success': True, 'message': 'PDF validated successfully'}
            
        except Exception as e:
            error_msg = f"User input processing failed: {str(e)}"
            logger.error(error_msg)
            
            store = context.get('store', {})
            store.update({
                'status': 'error',
                'error': error_msg,
                'last_updated': datetime.now().isoformat()
            })
            
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up after node execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Cleaning up {self.name}")


class PaperAnalysisNode(AsyncNode):
    """Node for extracting and analyzing paper content using LLM."""
    
    def __init__(self, pdf_extractor: PDFExtractorUtility, llm_utility: LLMAnalysisUtility):
        """Initialize the paper analysis node.
        
        Args:
            pdf_extractor: PDF extraction utility instance
            llm_utility: LLM analysis utility instance
        """
        super().__init__()
        self.pdf_extractor = pdf_extractor
        self.llm_utility = llm_utility
        self.name = "PaperAnalysisNode"
    
    async def prepare(self, context: Dict[str, Any]) -> None:
        """Prepare node for execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Preparing {self.name}")
        context.setdefault('node_start_time', datetime.now())
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze paper content.
        
        Args:
            context: Execution context
            
        Returns:
            Processing result
        """
        logger.info(f"Processing {self.name}")
        
        try:
            store = context.get('store', {})
            pdf_path = store.get('pdf_path')
            
            if not pdf_path:
                raise CoordinatorNodeError("No PDF path in store")
            
            # Update progress
            store.update({
                'status': 'extracting_content',
                'progress': 20,
                'last_updated': datetime.now().isoformat()
            })
            
            # Extract text and metadata
            logger.info("Extracting text from PDF...")
            text_result = self.pdf_extractor.extract_text(pdf_path)
            metadata_result = self.pdf_extractor.extract_metadata(pdf_path)
            sections_result = self.pdf_extractor.extract_sections(pdf_path)
            references_result = self.pdf_extractor.extract_references(pdf_path)
            
            paper_text = text_result.get('text', '')
            paper_metadata = metadata_result.get('metadata', {})
            
            # Store extracted content
            store.update({
                'paper_content': paper_text,
                'paper_metadata': paper_metadata,
                'paper_sections': sections_result.get('sections', {}),
                'paper_references': references_result.get('references', []),
                'status': 'analyzing_content',
                'progress': 40,
                'last_updated': datetime.now().isoformat()
            })
            
            # Analyze paper with LLM
            logger.info("Analyzing paper content with LLM...")
            analysis_result = await self.llm_utility.analyze_paper(paper_text, paper_metadata)
            
            # Store analysis results
            store.update({
                'analysis_results': analysis_result,
                'status': 'analysis_complete',
                'progress': 60,
                'last_updated': datetime.now().isoformat()
            })
            
            logger.info("Paper analysis completed successfully")
            return {'success': True, 'message': 'Paper analysis completed'}
            
        except Exception as e:
            error_msg = f"Paper analysis failed: {str(e)}"
            logger.error(error_msg)
            
            store = context.get('store', {})
            store.update({
                'status': 'error',
                'error': error_msg,
                'last_updated': datetime.now().isoformat()
            })
            
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up after node execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Cleaning up {self.name}")


class CitationSearchNode(AsyncNode):
    """Node for delegating citation search to Web Search Agent."""
    
    def __init__(self, scholar_utility: ScholarSearchUtility, llm_utility: LLMAnalysisUtility):
        """Initialize the citation search node.
        
        Args:
            scholar_utility: Scholar search utility instance
            llm_utility: LLM utility for query generation
        """
        super().__init__()
        self.scholar_utility = scholar_utility
        self.llm_utility = llm_utility
        self.name = "CitationSearchNode"
    
    async def prepare(self, context: Dict[str, Any]) -> None:
        """Prepare node for execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Preparing {self.name}")
        context.setdefault('node_start_time', datetime.now())
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Search for citing papers.
        
        Args:
            context: Execution context
            
        Returns:
            Processing result
        """
        logger.info(f"Processing {self.name}")
        
        try:
            store = context.get('store', {})
            paper_metadata = store.get('paper_metadata', {})
            analysis_results = store.get('analysis_results', {})
            
            if not paper_metadata:
                raise CoordinatorNodeError("No paper metadata available")
            
            # Update progress
            store.update({
                'status': 'searching_citations',
                'progress': 70,
                'last_updated': datetime.now().isoformat()
            })
            
            # Generate search queries using LLM
            logger.info("Generating search queries...")
            enhanced_metadata = {
                **paper_metadata,
                'key_concepts': analysis_results.get('key_concepts', [])
            }
            
            query_result = await self.llm_utility.generate_search_queries(enhanced_metadata)
            
            # Use primary search approach
            title = paper_metadata.get('title', '')
            authors = paper_metadata.get('authors', [])
            year = paper_metadata.get('year', 2020)
            
            if not title:
                raise CoordinatorNodeError("No paper title available for search")
            
            # Search for citations
            logger.info(f"Searching for citations of: {title}")
            search_result = self.scholar_utility.search_citations(
                title, authors, year, max_results=20
            )
            
            # Filter results for relevance and recency
            if search_result.get('success') and search_result.get('papers'):
                filtered_papers = self.scholar_utility.filter_results(
                    search_result['papers'],
                    min_year=year - 5,  # Last 5 years
                    relevance_threshold=0.3
                )
                
                # Format citations
                formatted_citations = self.scholar_utility.format_citations(filtered_papers)
            else:
                filtered_papers = []
                formatted_citations = {'citations': [], 'summary': {'total_count': 0}}
            
            # Store search results
            store.update({
                'search_queries': query_result.get('queries', []),
                'citation_results': search_result,
                'filtered_citations': filtered_papers,
                'formatted_citations': formatted_citations,
                'status': 'citations_found',
                'progress': 80,
                'last_updated': datetime.now().isoformat()
            })
            
            logger.info(f"Citation search completed. Found {len(filtered_papers)} relevant papers")
            return {'success': True, 'message': f'Found {len(filtered_papers)} citing papers'}
            
        except Exception as e:
            error_msg = f"Citation search failed: {str(e)}"
            logger.error(error_msg)
            
            store = context.get('store', {})
            store.update({
                'status': 'error',
                'error': error_msg,
                'last_updated': datetime.now().isoformat()
            })
            
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up after node execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Cleaning up {self.name}")


class ResearchSynthesisNode(AsyncNode):
    """Node for delegating research synthesis to Research Agent."""
    
    def __init__(self, llm_utility: LLMAnalysisUtility):
        """Initialize the research synthesis node.
        
        Args:
            llm_utility: LLM analysis utility instance
        """
        super().__init__()
        self.llm_utility = llm_utility
        self.name = "ResearchSynthesisNode"
    
    async def prepare(self, context: Dict[str, Any]) -> None:
        """Prepare node for execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Preparing {self.name}")
        context.setdefault('node_start_time', datetime.now())
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize research directions.
        
        Args:
            context: Execution context
            
        Returns:
            Processing result
        """
        logger.info(f"Processing {self.name}")
        
        try:
            store = context.get('store', {})
            analysis_results = store.get('analysis_results', {})
            filtered_citations = store.get('filtered_citations', [])
            
            if not analysis_results:
                raise CoordinatorNodeError("No analysis results available")
            
            # Update progress
            store.update({
                'status': 'synthesizing_research',
                'progress': 90,
                'last_updated': datetime.now().isoformat()
            })
            
            # Synthesize research directions
            logger.info("Synthesizing research directions...")
            synthesis_result = await self.llm_utility.synthesize_research_directions(
                analysis_results, filtered_citations
            )
            
            # Store synthesis results
            store.update({
                'research_suggestions': synthesis_result,
                'status': 'synthesis_complete',
                'progress': 95,
                'last_updated': datetime.now().isoformat()
            })
            
            suggestions_count = len(synthesis_result.get('suggestions', []))
            logger.info(f"Research synthesis completed. Generated {suggestions_count} suggestions")
            return {'success': True, 'message': f'Generated {suggestions_count} research directions'}
            
        except Exception as e:
            error_msg = f"Research synthesis failed: {str(e)}"
            logger.error(error_msg)
            
            store = context.get('store', {})
            store.update({
                'status': 'error',
                'error': error_msg,
                'last_updated': datetime.now().isoformat()
            })
            
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up after node execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Cleaning up {self.name}")


class PresentationNode(AsyncNode):
    """Node for formatting and presenting final results in multiple formats."""
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """Initialize the presentation node.
        
        Args:
            supported_formats: List of supported output formats. Defaults to all formats.
        """
        super().__init__()
        self.name = "PresentationNode"
        self.supported_formats = supported_formats or ['json', 'markdown', 'html', 'txt']
    
    async def prepare(self, context: Dict[str, Any]) -> None:
        """Prepare node for execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Preparing {self.name}")
        context.setdefault('node_start_time', datetime.now())
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format and present final results in multiple formats.
        
        Args:
            context: Execution context
            
        Returns:
            Processing result with formatted presentations
        """
        logger.info(f"Processing {self.name}")
        
        try:
            store = context.get('store', {})
            
            # Update progress
            store.update({
                'status': 'formatting_results',
                'progress': 95,
                'last_updated': datetime.now().isoformat()
            })
            
            # Gather all results into standardized format
            presentation_data = self._compile_presentation_data(store)
            
            # Generate presentations in multiple formats
            formatted_presentations = {}
            export_files = {}
            
            logger.info(f"Generating presentations in {len(self.supported_formats)} formats...")
            
            for format_type in self.supported_formats:
                try:
                    formatter = FormatterFactory.create_formatter(format_type)
                    formatted_content = formatter.format(presentation_data)
                    formatted_presentations[format_type] = formatted_content
                    
                    # Generate file path for export
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    paper_title = presentation_data.get('paper_metadata', {}).get('title', 'analysis')
                    safe_title = self._make_safe_filename(paper_title)
                    file_extension = formatter.get_file_extension()
                    
                    filename = f"scholar_ai_{safe_title}_{timestamp}{file_extension}"
                    export_files[format_type] = filename
                    
                    logger.info(f"Generated {format_type.upper()} presentation")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate {format_type} format: {e}")
                    formatted_presentations[format_type] = None
                    export_files[format_type] = None
            
            # Generate progress summary
            progress_summary = self._generate_progress_summary(store)
            
            # Store final results
            store.update({
                'final_results': presentation_data,
                'formatted_presentations': formatted_presentations,
                'export_files': export_files,
                'progress_summary': progress_summary,
                'status': 'completed',
                'progress': 100,
                'last_updated': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"Presentation formatting completed successfully in {len([f for f in formatted_presentations.values() if f])} formats")
            
            return {
                'success': True, 
                'message': f'Analysis completed with {len([f for f in formatted_presentations.values() if f])} format(s)',
                'results': presentation_data,
                'presentations': formatted_presentations,
                'export_files': export_files,
                'progress_summary': progress_summary
            }
            
        except Exception as e:
            error_msg = f"Presentation formatting failed: {str(e)}"
            logger.error(error_msg)
            
            store = context.get('store', {})
            store.update({
                'status': 'error',
                'error': error_msg,
                'last_updated': datetime.now().isoformat()
            })
            
            return {'success': False, 'error': error_msg}
    
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up after node execution.
        
        Args:
            context: Execution context
        """
        logger.info(f"Cleaning up {self.name}")
    
    def _compile_presentation_data(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all analysis data into standardized presentation format.
        
        Args:
            store: Data store containing analysis results
            
        Returns:
            Standardized presentation data
        """
        # Paper metadata and analysis
        paper_metadata = store.get('paper_metadata', {})
        analysis_results = store.get('analysis_results', {})
        
        # Citations data
        filtered_citations = store.get('filtered_citations', [])
        formatted_citations = store.get('formatted_citations', {})
        
        # Research directions
        research_suggestions = store.get('research_suggestions', {})
        
        # Processing metadata
        processing_metadata = {
            'pdf_path': store.get('pdf_path', ''),
            'processed_at': datetime.now().isoformat(),
            'total_processing_time': self._calculate_processing_time(store),
            'status': store.get('status', 'completed')
        }
        
        return {
            'paper_metadata': paper_metadata,
            'paper_analysis': analysis_results,
            'citations': filtered_citations,
            'research_directions': research_suggestions,
            'formatted_citations': formatted_citations,
            'processing_metadata': processing_metadata
        }
    
    def _generate_progress_summary(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis progress and results.
        
        Args:
            store: Data store
            
        Returns:
            Progress summary
        """
        analysis_results = store.get('analysis_results', {})
        filtered_citations = store.get('filtered_citations', [])
        research_suggestions = store.get('research_suggestions', {})
        
        return {
            'stages_completed': [
                'PDF Processing',
                'Paper Analysis', 
                'Citation Search',
                'Research Synthesis',
                'Result Formatting'
            ],
            'results_summary': {
                'paper_analyzed': bool(analysis_results),
                'key_concepts_found': len(analysis_results.get('key_concepts', [])),
                'citations_found': len(filtered_citations),
                'research_directions_generated': len(research_suggestions.get('suggestions', [])),
                'formats_generated': len(self.supported_formats)
            },
            'total_processing_time': self._calculate_processing_time(store),
            'completion_status': 'success'
        }
    
    def _calculate_processing_time(self, store: Dict[str, Any]) -> Optional[float]:
        """Calculate total processing time.
        
        Args:
            store: Data store
            
        Returns:
            Processing time in seconds or None
        """
        try:
            start_time = store.get('started_at')
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                return (datetime.now() - start_dt).total_seconds()
        except Exception:
            pass
        return None
    
    def _make_safe_filename(self, title: str, max_length: int = 50) -> str:
        """Convert a paper title to a safe filename.
        
        Args:
            title: Paper title
            max_length: Maximum filename length
            
        Returns:
            Safe filename string
        """
        if not title:
            return "untitled"
        
        # Remove/replace problematic characters
        safe_chars = []
        for char in title.lower():
            if char.isalnum():
                safe_chars.append(char)
            elif char in ' -_':
                safe_chars.append('_')
        
        safe_title = ''.join(safe_chars)
        
        # Remove multiple underscores and trim
        safe_title = '_'.join(filter(None, safe_title.split('_')))
        
        # Limit length
        if len(safe_title) > max_length:
            safe_title = safe_title[:max_length].rstrip('_')
        
        return safe_title or "untitled"