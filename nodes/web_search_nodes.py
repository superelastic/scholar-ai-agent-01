"""Web Search Agent nodes for academic citation search."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils import ScholarSearchUtility, LLMAnalysisUtility

logger = logging.getLogger(__name__)


class WebSearchNodeError(Exception):
    """Custom exception for Web Search node errors."""
    pass


class AsyncNode:
    """Base class for async nodes."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the node.
        
        Args:
            context: Execution context containing store and input data
            
        Returns:
            Processing result with success status
        """
        raise NotImplementedError


class SearchQueryNode(AsyncNode):
    """Node for generating effective academic search queries."""

    def __init__(self, llm_utility: LLMAnalysisUtility):
        """Initialize the search query node.
        
        Args:
            llm_utility: LLM utility for query generation
        """
        self.llm_utility = llm_utility

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized search queries.
        
        Args:
            context: Execution context
            
        Returns:
            Generated queries and success status
        """
        store = context['store']
        input_data = context['input_data']
        
        try:
            # Extract paper metadata from input
            paper_metadata = input_data.get('paper_metadata', {})
            if not paper_metadata:
                raise WebSearchNodeError("No paper metadata provided for query generation")
            
            store['paper_metadata'] = paper_metadata
            store['status'] = 'generating_queries'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Generating queries for paper: {paper_metadata.get('title', 'Unknown')}")
            
            # Use LLM to generate optimized search queries
            analysis_context = {
                'title': paper_metadata.get('title', ''),
                'authors': paper_metadata.get('authors', []),
                'year': paper_metadata.get('year'),
                'key_concepts': paper_metadata.get('key_concepts', [])
            }
            
            try:
                result = await self.llm_utility.generate_search_queries(
                    paper_metadata=analysis_context,
                    max_queries=3
                )
                
                if result.get('success'):
                    queries = result['queries']
                    logger.info(f"Generated {len(queries)} LLM-optimized queries")
                else:
                    # Fallback to basic queries
                    queries = self._generate_fallback_queries(paper_metadata)
                    logger.info(f"Used fallback query generation, created {len(queries)} queries")
                
            except Exception as e:
                logger.warning(f"LLM query generation failed: {e}, using fallback")
                queries = self._generate_fallback_queries(paper_metadata)
            
            store['search_queries'] = queries
            store['status'] = 'queries_generated'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'queries': queries
            }
            
        except Exception as e:
            error_msg = f"Query generation failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'query_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_fallback_queries(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """Generate basic queries when LLM fails.
        
        Args:
            paper_metadata: Paper metadata
            
        Returns:
            List of basic queries
        """
        queries = []
        title = paper_metadata.get('title', '')
        authors = paper_metadata.get('authors', [])
        
        if title and authors:
            # Primary query with title and first author
            primary_author = authors[0] if authors else ''
            if primary_author:
                queries.append(f'"{title}" {primary_author}')
            
            # Secondary query with key terms from title
            title_words = title.split()[:3]  # First 3 words
            if len(title_words) >= 2:
                queries.append(' '.join(title_words))
        
        # Ensure we have at least one query
        if not queries and title:
            queries.append(title)
        
        return queries[:3]  # Maximum 3 queries


class GoogleScholarNode(AsyncNode):
    """Node for executing Google Scholar searches with retry logic."""

    def __init__(self, scholar_utility: ScholarSearchUtility):
        """Initialize the Google Scholar node.
        
        Args:
            scholar_utility: Scholar utility for search execution
        """
        self.scholar_utility = scholar_utility

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Scholar searches for all queries.
        
        Args:
            context: Execution context
            
        Returns:
            Search results and success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            queries = store.get('search_queries', [])
            if not queries:
                raise WebSearchNodeError("No search queries available")
            
            store['status'] = 'searching_scholar'
            store['last_updated'] = datetime.now().isoformat()
            
            max_results = config.get('max_results', 20)
            results_per_query = max_results // len(queries) if queries else max_results
            
            all_results = []
            successful_queries = 0
            
            for i, query in enumerate(queries):
                try:
                    logger.info(f"Executing Scholar search {i+1}/{len(queries)}: {query}")
                    
                    result = self.scholar_utility.search_citations(
                        paper_title=query,
                        authors=[],  # Query already includes author info
                        year=None,
                        max_results=results_per_query
                    )
                    
                    if result.get('success') and result.get('papers'):
                        papers = result['papers']
                        all_results.extend(papers)
                        successful_queries += 1
                        logger.info(f"Found {len(papers)} results for query: {query}")
                    else:
                        logger.warning(f"No results for query: {query}")
                        
                except Exception as e:
                    logger.error(f"Search failed for query '{query}': {e}")
                    store['retry_count'] = store.get('retry_count', 0) + 1
                    
                    # Continue with other queries even if one fails
                    if store['retry_count'] >= 3:
                        logger.error("Max retries exceeded for Scholar searches")
                        break
            
            # Remove duplicates based on title similarity
            unique_results = self._deduplicate_results(all_results)
            
            store['raw_results'] = unique_results
            store['search_stats']['total_queries'] = len(queries)
            store['search_stats']['total_results'] = len(unique_results)
            store['status'] = 'search_completed'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Scholar search completed: {len(unique_results)} unique results from {successful_queries}/{len(queries)} successful queries")
            
            return {
                'success': True,
                'results': unique_results,
                'stats': {
                    'successful_queries': successful_queries,
                    'total_queries': len(queries),
                    'unique_results': len(unique_results)
                }
            }
            
        except Exception as e:
            error_msg = f"Scholar search failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'search_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity.
        
        Args:
            results: List of paper results
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return []
        
        unique_results = []
        seen_titles = set()
        
        for paper in results:
            title = paper.get('title', '').lower().strip()
            if not title:
                continue
                
            # Simple deduplication based on title words
            title_words = set(title.split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # Consider duplicate if 80% of words overlap
                if title_words and seen_words:
                    overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_results.append(paper)
                seen_titles.add(title)
        
        logger.info(f"Deduplicated {len(results)} results to {len(unique_results)}")
        return unique_results


class CitationFilterNode(AsyncNode):
    """Node for filtering search results by year and relevance."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Filter results by year and relevance threshold.
        
        Args:
            context: Execution context
            
        Returns:
            Filtered results and success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            raw_results = store.get('raw_results', [])
            
            store['status'] = 'filtering_results'
            store['last_updated'] = datetime.now().isoformat()
            
            if not raw_results:
                logger.info("No raw results to filter")
                store['filtered_results'] = []
                store['search_stats']['filtered_count'] = 0
                store['status'] = 'results_filtered'
                return {
                    'success': True,
                    'filtered': []
                }
            
            # Filter parameters
            year_filter = config.get('year_filter', 3)  # Last 3 years
            relevance_threshold = config.get('relevance_threshold', 0.5)
            max_results = config.get('max_results', 20)
            
            # Apply year filter
            current_year = datetime.now().year
            min_year = current_year - year_filter
            
            # Use utility method for filtering
            filtered = []
            for paper in raw_results:
                # Year filter
                paper_year = paper.get('year', 0)
                if paper_year and paper_year < min_year:
                    continue
                
                # Relevance filter
                relevance = paper.get('relevance_score', 0)
                if relevance < relevance_threshold:
                    continue
                
                filtered.append(paper)
            
            # Sort by relevance and year (most recent first)
            filtered.sort(key=lambda x: (x.get('relevance_score', 0), x.get('year', 0)), reverse=True)
            
            # Limit final results
            filtered = filtered[:max_results]
            
            store['filtered_results'] = filtered
            store['search_stats']['filtered_count'] = len(filtered)
            store['status'] = 'results_filtered'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Filtered {len(raw_results)} results to {len(filtered)} (year >= {min_year}, relevance >= {relevance_threshold})")
            
            return {
                'success': True,
                'filtered': filtered,
                'stats': {
                    'original_count': len(raw_results),
                    'filtered_count': len(filtered),
                    'year_threshold': min_year,
                    'relevance_threshold': relevance_threshold
                }
            }
            
        except Exception as e:
            error_msg = f"Result filtering failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'filter_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }


class CitationFormatterNode(AsyncNode):
    """Node for formatting filtered citations for presentation."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format filtered results for presentation.
        
        Args:
            context: Execution context
            
        Returns:
            Formatted citations and success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            filtered_results = store.get('filtered_results', [])
            
            store['status'] = 'formatting_citations'
            store['last_updated'] = datetime.now().isoformat()
            
            # Create comprehensive citation format
            formatted = {
                'citations': [],
                'summary': {
                    'total_count': len(filtered_results),
                    'recent_count': 0,
                    'high_relevance_count': 0,
                    'avg_relevance': 0.0,
                    'year_range': {'min': None, 'max': None}
                },
                'metadata': {
                    'agent': 'AcademicWebSearchAgent',
                    'queries_used': store.get('search_queries', []),
                    'filter_settings': {
                        'year_filter': config.get('year_filter', 3),
                        'relevance_threshold': config.get('relevance_threshold', 0.5),
                        'max_results': config.get('max_results', 20)
                    },
                    'generated_at': datetime.now().isoformat()
                }
            }
            
            if filtered_results:
                current_year = datetime.now().year
                relevances = []
                years = []
                
                for paper in filtered_results:
                    # Format individual citation
                    citation = {
                        'title': paper.get('title', 'Unknown Title'),
                        'authors': paper.get('authors', []),
                        'year': paper.get('year'),
                        'venue': paper.get('venue', ''),
                        'url': paper.get('url', ''),
                        'relevance_score': paper.get('relevance_score', 0.0),
                        'citation_count': paper.get('citation_count', 0),
                        'abstract': paper.get('abstract', ''),
                        'is_recent': (paper.get('year', 0) >= current_year - 2),
                        'is_high_relevance': (paper.get('relevance_score', 0) >= 0.7)
                    }
                    
                    formatted['citations'].append(citation)
                    
                    # Collect stats
                    relevance = paper.get('relevance_score', 0)
                    year = paper.get('year')
                    
                    relevances.append(relevance)
                    if year:
                        years.append(year)
                
                # Calculate summary statistics
                formatted['summary']['recent_count'] = sum(1 for c in formatted['citations'] if c['is_recent'])
                formatted['summary']['high_relevance_count'] = sum(1 for c in formatted['citations'] if c['is_high_relevance'])
                formatted['summary']['avg_relevance'] = sum(relevances) / len(relevances) if relevances else 0.0
                
                if years:
                    formatted['summary']['year_range']['min'] = min(years)
                    formatted['summary']['year_range']['max'] = max(years)
            
            store['formatted_citations'] = formatted
            store['status'] = 'citations_formatted'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Formatted {len(filtered_results)} citations with comprehensive metadata")
            
            return {
                'success': True,
                'formatted': formatted
            }
            
        except Exception as e:
            error_msg = f"Citation formatting failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'format_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }