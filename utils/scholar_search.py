"""Google Scholar search utility for finding citing papers."""

import hashlib
import json
import logging
import random
import re
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ScholarSearchError(Exception):
    """Base exception for Scholar search errors."""
    pass


class ScholarRateLimitError(ScholarSearchError):
    """Raised when rate limited by Google Scholar."""
    pass


class ScholarParsingError(ScholarSearchError):
    """Raised when parsing Scholar results fails."""
    pass


class ScholarSearchUtility:
    """Utility service for searching Google Scholar to find citing papers."""
    
    def __init__(
        self,
        cache_ttl_hours: int = 24,
        min_request_interval: float = 2.0,
        max_retries: int = 3,
        timeout: int = 10
    ):
        """Initialize Scholar Search Utility.
        
        Args:
            cache_ttl_hours: Cache time-to-live in hours
            min_request_interval: Minimum seconds between requests
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache = {}
        self.last_request_time = 0
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        logger.info("Initialized ScholarSearchUtility")
    
    def _get_cache_key(self, paper_title: str, authors: List[str], year: int) -> str:
        """Generate cache key for search parameters.
        
        Args:
            paper_title: Title of the paper
            authors: List of authors
            year: Publication year
            
        Returns:
            Cache key string
        """
        content = f"{paper_title}_{','.join(authors)}_{year}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached result if still valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if cache_key in self.cache:
            timestamp, result = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                logger.info("Cache hit for Scholar search")
                return result
            else:
                del self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict) -> None:
        """Save result to cache.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        self.cache[cache_key] = (datetime.now(), result)
        logger.info("Cached Scholar search result")
    
    def _respect_rate_limit(self) -> None:
        """Ensure minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers to avoid detection.
        
        Returns:
            Dictionary of HTTP headers
        """
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _build_citation_query(self, paper_title: str, authors: List[str], year: int) -> str:
        """Build Google Scholar search query to find citing papers.
        
        Args:
            paper_title: Title of the paper
            authors: List of authors
            year: Publication year
            
        Returns:
            Search query string
        """
        # Clean and format title
        title_words = re.findall(r'\w+', paper_title.lower())
        title_phrase = ' '.join(title_words[:8])  # Use first 8 words for better matching
        
        # Build query to find the original paper first
        if authors:
            main_author = authors[0].split()[-1]  # Last name of first author
            query = f'"{title_phrase}" {main_author}'
        else:
            query = f'"{title_phrase}"'
        
        return query
    
    def _build_scholar_url(self, query: str, start: int = 0) -> str:
        """Build Google Scholar search URL.
        
        Args:
            query: Search query
            start: Starting result index
            
        Returns:
            Scholar search URL
        """
        base_url = "https://scholar.google.com/scholar"
        params = {
            'q': query,
            'hl': 'en',
            'as_sdt': '0,5',  # Include patents
            'start': start
        }
        
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    
    def _parse_scholar_page(self, html: str) -> List[Dict]:
        """Parse Google Scholar results page.
        
        Args:
            html: HTML content of Scholar page
            
        Returns:
            List of parsed paper results
        """
        soup = BeautifulSoup(html, 'lxml')
        results = []
        
        # Find all result entries
        entries = soup.find_all('div', class_='gs_r gs_or gs_scl')
        
        for entry in entries:
            try:
                result = self._parse_single_result(entry)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse Scholar result: {e}")
                continue
        
        return results
    
    def _parse_single_result(self, entry) -> Optional[Dict]:
        """Parse a single Scholar search result.
        
        Args:
            entry: BeautifulSoup element for the result
            
        Returns:
            Parsed result dictionary or None
        """
        # Extract title and URL
        title_elem = entry.find('h3', class_='gs_rt')
        if not title_elem:
            return None
        
        title_link = title_elem.find('a')
        if title_link:
            title = title_link.get_text().strip()
            url = title_link.get('href', '')
        else:
            title = title_elem.get_text().strip()
            url = ''
        
        # Extract authors and venue
        authors_elem = entry.find('div', class_='gs_a')
        authors = []
        venue = ''
        year = None
        
        if authors_elem:
            authors_text = authors_elem.get_text()
            
            # Try to extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
            if year_match:
                year = int(year_match.group())
            
            # Extract authors (before first '-' or ',' if present)
            author_parts = re.split(r'[-,]', authors_text)
            if author_parts:
                author_names = author_parts[0].strip()
                authors = [name.strip() for name in re.split(r',|\band\b', author_names) if name.strip()]
            
            # Extract venue (after year, if present)
            if year:
                venue_match = re.search(rf'{year}\s*[-,]\s*(.+)', authors_text)
                if venue_match:
                    venue = venue_match.group(1).strip()
        
        # Extract snippet/abstract
        snippet_elem = entry.find('span', class_='gs_rs')
        snippet = snippet_elem.get_text().strip() if snippet_elem else ''
        
        # Extract citation count
        cited_by = 0
        cited_elem = entry.find('a', string=re.compile(r'Cited by'))
        if cited_elem:
            cited_text = cited_elem.get_text()
            cited_match = re.search(r'Cited by (\d+)', cited_text)
            if cited_match:
                cited_by = int(cited_match.group(1))
        
        return {
            'title': title,
            'authors': authors[:3],  # Limit to first 3 authors
            'year': year,
            'venue': venue[:100],  # Limit venue length
            'url': url,
            'snippet': snippet[:300],  # Limit snippet length
            'cited_by': cited_by,
            'relevance_score': 0.0  # Will be calculated later
        }
    
    def _execute_scholar_search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Execute Google Scholar search with retry logic.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of search results
            
        Raises:
            ScholarSearchError: If search fails after retries
        """
        results = []
        start = 0
        results_per_page = 10
        
        while len(results) < max_results and start < 100:  # Scholar limit
            for attempt in range(self.max_retries):
                try:
                    self._respect_rate_limit()
                    
                    url = self._build_scholar_url(query, start)
                    headers = self._get_random_headers()
                    
                    logger.info(f"Searching Scholar: {query} (start={start}, attempt={attempt+1})")
                    
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Check for rate limiting
                    if 'unusual traffic' in response.text.lower():
                        raise ScholarRateLimitError("Rate limited by Google Scholar")
                    
                    page_results = self._parse_scholar_page(response.text)
                    
                    if not page_results:
                        logger.info("No more results found")
                        return results[:max_results]
                    
                    results.extend(page_results)
                    break
                    
                except requests.RequestException as e:
                    logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise ScholarSearchError(f"Failed to search Scholar after {self.max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                except ScholarRateLimitError as e:
                    logger.error(f"Rate limited: {e}")
                    raise e
            
            start += results_per_page
        
        return results[:max_results]
    
    def _calculate_relevance_score(
        self, 
        result: Dict, 
        paper_title: str, 
        authors: List[str], 
        year: int
    ) -> float:
        """Calculate relevance score for a search result.
        
        Args:
            result: Search result
            paper_title: Original paper title
            authors: Original paper authors
            year: Original paper year
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        
        # Title similarity (40% weight)
        title_words = set(re.findall(r'\w+', paper_title.lower()))
        result_words = set(re.findall(r'\w+', result.get('title', '').lower()))
        
        if title_words and result_words:
            title_similarity = len(title_words & result_words) / len(title_words | result_words)
            score += title_similarity * 0.4
        
        # Author overlap (20% weight)
        if authors and result.get('authors'):
            original_authors = set(author.lower() for author in authors)
            result_authors = set(author.lower() for author in result['authors'])
            author_overlap = len(original_authors & result_authors) / max(len(original_authors), 1)
            score += author_overlap * 0.2
        
        # Recency bonus (20% weight)
        if result.get('year') and year:
            year_diff = abs(result['year'] - year)
            recency_score = max(0, 1 - year_diff / 10)  # Decay over 10 years
            score += recency_score * 0.2
        
        # Citation count (20% weight)
        cited_by = result.get('cited_by', 0)
        citation_score = min(1.0, cited_by / 100)  # Normalize to max 100 citations
        score += citation_score * 0.2
        
        return min(1.0, score)
    
    def _find_paper_cluster_id(self, paper_title: str, authors: List[str]) -> Optional[str]:
        """Find a paper and extract its Google Scholar cluster ID.
        
        Args:
            paper_title: Title of the paper
            authors: List of authors
            
        Returns:
            Cluster ID if found, None otherwise
        """
        try:
            query = self._build_citation_query(paper_title, authors, 0)
            url = self._build_scholar_url(query, 0)
            headers = self._get_random_headers()
            
            self._respect_rate_limit()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find first result
            first_result = soup.find('div', class_='gs_r gs_or gs_scl')
            if not first_result:
                logger.warning("No results found for paper")
                return None
            
            # Find "Cited by" link to get cluster ID
            cited_by_link = first_result.find('a', string=re.compile(r'Cited by'))
            if not cited_by_link:
                logger.warning("No 'Cited by' link found for paper")
                return None
            
            href = cited_by_link.get('href', '')
            
            # Extract cluster ID from the cited by link
            cluster_match = re.search(r'cites=(\d+)', href)
            if not cluster_match:
                logger.warning("Could not extract cluster ID from cited by link")
                return None
            
            cluster_id = cluster_match.group(1)
            
            # Log citation count for debugging
            cited_text = cited_by_link.get_text()
            cited_count_match = re.search(r'Cited by (\d+)', cited_text)
            if cited_count_match:
                cited_count = int(cited_count_match.group(1))
                logger.info(f"Found paper with {cited_count:,} citations (cluster ID: {cluster_id})")
            
            return cluster_id
            
        except Exception as e:
            logger.error(f"Failed to find paper cluster ID: {e}")
            return None
    
    def _build_citations_url(self, cluster_id: str, start: int = 0) -> str:
        """Build Google Scholar URL to get papers citing a specific work.
        
        Args:
            cluster_id: Google Scholar cluster ID
            start: Starting result index
            
        Returns:
            Citations URL
        """
        base_url = "https://scholar.google.com/scholar"
        params = {
            'cites': cluster_id,
            'hl': 'en',
            'as_sdt': '0,5',
            'start': start
        }
        
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    
    def search_citations(
        self, 
        paper_title: str, 
        authors: List[str], 
        year: int, 
        max_results: int = 20
    ) -> Dict:
        """Find papers that cite the given work.
        
        Args:
            paper_title: Title of the paper to find citations for
            authors: List of paper authors
            year: Publication year
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        cache_key = self._get_cache_key(paper_title, authors, year)
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            # First, find the paper and get its cluster ID
            cluster_id = self._find_paper_cluster_id(paper_title, authors)
            
            if not cluster_id:
                logger.warning("Could not find paper cluster ID, falling back to title search")
                # Fallback: search for papers mentioning the title
                query = f'"{paper_title}"'
                raw_results = self._execute_scholar_search(query, max_results)
            else:
                # Use cluster ID to get actual citing papers
                raw_results = []
                start = 0
                results_per_page = 10
                
                while len(raw_results) < max_results and start < 100:
                    self._respect_rate_limit()
                    
                    url = self._build_citations_url(cluster_id, start)
                    headers = self._get_random_headers()
                    
                    logger.info(f"Fetching citations using cluster ID: {cluster_id} (start={start})")
                    
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Check for rate limiting
                    if 'unusual traffic' in response.text.lower():
                        raise ScholarRateLimitError("Rate limited by Google Scholar")
                    
                    page_results = self._parse_scholar_page(response.text)
                    
                    if not page_results:
                        logger.info("No more citation results found")
                        break
                    
                    raw_results.extend(page_results)
                    start += results_per_page
                
                raw_results = raw_results[:max_results]
                query = f"Citations for: {paper_title}"
            
            # Calculate relevance scores
            for result in raw_results:
                result['relevance_score'] = self._calculate_relevance_score(
                    result, paper_title, authors, year
                )
            
            # Sort by relevance score
            raw_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            search_result = {
                'query': query,
                'total_found': len(raw_results),
                'papers': raw_results,
                'search_timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            self._save_to_cache(cache_key, search_result)
            return search_result
            
        except Exception as e:
            logger.error(f"Scholar search failed: {e}")
            return {
                'query': '',
                'total_found': 0,
                'papers': [],
                'error': str(e),
                'search_timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def filter_results(
        self, 
        results: List[Dict], 
        min_year: Optional[int] = None, 
        relevance_threshold: float = 0.3
    ) -> List[Dict]:
        """Filter search results by year and relevance.
        
        Args:
            results: List of search results
            min_year: Minimum publication year (None for no filter)
            relevance_threshold: Minimum relevance score
            
        Returns:
            Filtered list of results
        """
        filtered = []
        
        for result in results:
            # Year filter
            if min_year and result.get('year'):
                if result['year'] < min_year:
                    continue
            
            # Relevance filter
            if result.get('relevance_score', 0) < relevance_threshold:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def format_citations(self, results: List[Dict]) -> Dict:
        """Format citations for presentation.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted citations dictionary
        """
        formatted = {
            'citations': [],
            'summary': {
                'total_count': len(results),
                'year_range': None,
                'top_venues': [],
                'avg_relevance': 0.0
            },
            'formatted_at': datetime.now().isoformat()
        }
        
        if not results:
            return formatted
        
        # Format individual citations
        for i, result in enumerate(results, 1):
            citation = {
                'rank': i,
                'title': result.get('title', 'Unknown'),
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'venue': result.get('venue', ''),
                'url': result.get('url', ''),
                'cited_by': result.get('cited_by', 0),
                'relevance_score': result.get('relevance_score', 0.0),
                'snippet': result.get('snippet', '')
            }
            formatted['citations'].append(citation)
        
        # Calculate summary statistics
        years = [r.get('year') for r in results if r.get('year')]
        if years:
            formatted['summary']['year_range'] = {
                'min': min(years),
                'max': max(years)
            }
        
        # Top venues
        venues = [r.get('venue', '') for r in results if r.get('venue')]
        venue_counts = {}
        for venue in venues:
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        top_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        formatted['summary']['top_venues'] = [{'venue': v, 'count': c} for v, c in top_venues]
        
        # Average relevance
        relevance_scores = [r.get('relevance_score', 0) for r in results]
        if relevance_scores:
            formatted['summary']['avg_relevance'] = sum(relevance_scores) / len(relevance_scores)
        
        return formatted