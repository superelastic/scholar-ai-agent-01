"""LLM analysis utility for paper analysis and research synthesis."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMAnalysisError(Exception):
    """Base exception for LLM analysis errors."""
    pass


class LLMTimeoutError(LLMAnalysisError):
    """Raised when LLM calls timeout."""
    pass


class LLMAPIError(LLMAnalysisError):
    """Raised when LLM API calls fail."""
    pass


class LLMAnalysisUtility:
    """Utility service for managing prompts and LLM API calls."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model: str = "gpt-4",
        timeout: int = 30,
        cache_ttl_hours: int = 24
    ):
        """Initialize LLM Analysis Utility.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            model: Model name (gpt-4, claude-3-sonnet, etc.)
            timeout: Maximum timeout in seconds for LLM calls
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.model = model
        self.timeout = timeout
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache = {}
        
        # Initialize clients
        if openai_api_key:
            openai.api_key = openai_api_key
        
        if anthropic_api_key:
            self.anthropic = Anthropic(api_key=anthropic_api_key)
        else:
            self.anthropic = None
            
        logger.info(f"Initialized LLMAnalysisUtility with model: {model}")
    
    def _get_cache_key(self, text: str, operation: str) -> str:
        """Generate cache key for text and operation.
        
        Args:
            text: Input text
            operation: Operation type
            
        Returns:
            Cache key string
        """
        content = f"{operation}_{text}_{self.model}"
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
                logger.info("Cache hit for LLM analysis")
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
        logger.info("Cached LLM analysis result")
    
    async def _call_llm_with_retry(
        self, 
        prompt: str, 
        max_retries: int = 3,
        system_prompt: Optional[str] = None
    ) -> str:
        """Call LLM with retry logic and exponential backoff.
        
        Args:
            prompt: User prompt
            max_retries: Maximum retry attempts
            system_prompt: Optional system prompt
            
        Returns:
            LLM response text
            
        Raises:
            LLMAPIError: If all retries fail
            LLMTimeoutError: If timeout is exceeded
        """
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self.model.startswith("gpt"):
                    return await self._call_openai(prompt, system_prompt)
                elif self.model.startswith("claude"):
                    return await self._call_anthropic(prompt, system_prompt)
                else:
                    raise LLMAPIError(f"Unsupported model: {self.model}")
                    
            except asyncio.TimeoutError:
                raise LLMTimeoutError(f"LLM call timed out after {self.timeout} seconds")
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise LLMAPIError(f"All LLM API attempts failed: {e}")
                
                # Exponential backoff
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    
    async def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await asyncio.wait_for(
                openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                ),
                timeout=self.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMAPIError(f"OpenAI API error: {e}")
    
    async def _call_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Anthropic API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Response text
        """
        if not self.anthropic:
            raise LLMAPIError("Anthropic client not initialized")
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = await asyncio.wait_for(
                self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": full_prompt}]
                ),
                timeout=self.timeout
            )
            return response.content[0].text
        except Exception as e:
            raise LLMAPIError(f"Anthropic API error: {e}")
    
    def _create_analysis_prompt(self, paper_text: str, metadata: Dict) -> str:
        """Create prompt for paper analysis.
        
        Args:
            paper_text: Full paper text
            metadata: Paper metadata
            
        Returns:
            Analysis prompt
        """
        return f"""Analyze the following academic paper and extract key information.

Paper Metadata:
- Title: {metadata.get('title', 'Unknown')}
- Authors: {', '.join(metadata.get('authors', []))}
- Year: {metadata.get('year', 'Unknown')}

Paper Text:
{paper_text[:8000]}  # Limit to 8000 chars

Please extract and provide:
1. Key concepts and technical terms
2. Research methodology used
3. Main findings and contributions
4. Theoretical framework
5. Limitations mentioned
6. Future work suggestions

Format your response as JSON with the following structure:
{{
    "key_concepts": ["concept1", "concept2", ...],
    "methodology": "description of methodology",
    "findings": ["finding1", "finding2", ...],
    "theoretical_framework": "description",
    "limitations": ["limitation1", "limitation2", ...],
    "future_work": ["suggestion1", "suggestion2", ...]
}}"""
    
    def _create_query_generation_prompt(self, paper_metadata: Dict) -> str:
        """Create prompt for search query generation.
        
        Args:
            paper_metadata: Paper metadata
            
        Returns:
            Query generation prompt
        """
        return f"""Generate effective Google Scholar search queries to find papers that cite the following work:

Title: {paper_metadata.get('title', 'Unknown')}
Authors: {', '.join(paper_metadata.get('authors', []))}
Year: {paper_metadata.get('year', 'Unknown')}
Key Concepts: {', '.join(paper_metadata.get('key_concepts', []))}

Create 3-5 search queries that would effectively find citing papers. Queries should:
1. Include the main title words
2. Include author names
3. Use relevant technical terms
4. Vary in specificity to capture different types of citing papers

Format as JSON:
{{
    "queries": ["query1", "query2", "query3", ...]
}}"""
    
    def _create_synthesis_prompt(self, paper_data: Dict, citations: List[Dict]) -> str:
        """Create prompt for research synthesis.
        
        Args:
            paper_data: Original paper analysis
            citations: Citing papers data
            
        Returns:
            Synthesis prompt
        """
        citation_summaries = []
        for i, citation in enumerate(citations[:10], 1):  # Limit to 10 citations
            citation_summaries.append(f"{i}. {citation.get('title', 'Unknown')} ({citation.get('year', 'Unknown')})")
        
        return f"""Based on the original paper and its recent citations, suggest novel research directions.

Original Paper Analysis:
- Key Concepts: {', '.join(paper_data.get('key_concepts', []))}
- Methodology: {paper_data.get('methodology', 'Unknown')}
- Findings: {', '.join(paper_data.get('findings', []))}

Recent Citing Papers:
{chr(10).join(citation_summaries)}

Identify research gaps and suggest 3-5 novel research directions that:
1. Build upon the original work
2. Address current gaps in the field
3. Are informed by recent developments
4. Have practical significance

For each suggestion, provide:
- Title of the research direction
- Brief description
- Rationale for why it's important
- Confidence level (0.0-1.0)

Format as JSON:
{{
    "suggestions": [
        {{
            "title": "Research Direction Title",
            "description": "Detailed description",
            "rationale": "Why this is important",
            "confidence": 0.85
        }},
        ...
    ]
}}"""
    
    def _generate_simplified_analysis(self, paper_text: str, metadata: Dict) -> Dict:
        """Generate simplified analysis when LLM calls fail.
        
        Args:
            paper_text: Paper text
            metadata: Paper metadata
            
        Returns:
            Simplified analysis
        """
        logger.warning("Generating simplified analysis due to LLM failure")
        
        # Extract basic information without LLM
        words = paper_text.lower().split()
        
        # Simple keyword extraction
        academic_keywords = ['machine learning', 'deep learning', 'neural network', 
                           'algorithm', 'model', 'dataset', 'experiment', 'analysis']
        found_concepts = [kw for kw in academic_keywords if kw.replace(' ', '') in ' '.join(words)]
        
        return {
            "key_concepts": found_concepts[:5],
            "methodology": "Unable to analyze - simplified extraction",
            "findings": ["Analysis unavailable due to LLM service issues"],
            "theoretical_framework": "Not available",
            "limitations": ["Analysis incomplete"],
            "future_work": ["Full analysis needed when service is restored"],
            "fallback_used": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_paper(self, paper_text: str, metadata: Dict) -> Dict:
        """Extract key concepts, methodology, and findings from paper.
        
        Args:
            paper_text: Full text of the paper
            metadata: Paper metadata
            
        Returns:
            Analysis results dictionary
        """
        cache_key = self._get_cache_key(paper_text, "analysis")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            prompt = self._create_analysis_prompt(paper_text, metadata)
            response = await self._call_llm_with_retry(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            result["timestamp"] = datetime.now().isoformat()
            result["success"] = True
            
            self._save_to_cache(cache_key, result)
            return result
            
        except (LLMTimeoutError, LLMAPIError) as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._generate_simplified_analysis(paper_text, metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._generate_simplified_analysis(paper_text, metadata)
    
    async def generate_search_queries(self, paper_metadata: Dict) -> Dict:
        """Create effective Google Scholar search queries.
        
        Args:
            paper_metadata: Paper metadata including title, authors, concepts
            
        Returns:
            Dictionary with generated queries
        """
        cache_key = self._get_cache_key(str(paper_metadata), "queries")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            prompt = self._create_query_generation_prompt(paper_metadata)
            response = await self._call_llm_with_retry(prompt)
            
            result = json.loads(response)
            result["timestamp"] = datetime.now().isoformat()
            result["success"] = True
            
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback query generation
            title = paper_metadata.get('title', '')
            authors = paper_metadata.get('authors', [])
            
            fallback_queries = []
            if title:
                fallback_queries.append(f'"{title}"')
                if authors:
                    fallback_queries.append(f'"{title}" {authors[0]}')
            
            return {
                "queries": fallback_queries[:3],
                "fallback_used": True,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def synthesize_research_directions(
        self, 
        paper_data: Dict, 
        citations: List[Dict]
    ) -> Dict:
        """Generate future research directions based on paper and citations.
        
        Args:
            paper_data: Original paper analysis
            citations: List of citing papers
            
        Returns:
            Research directions dictionary
        """
        cache_key = self._get_cache_key(f"{paper_data}_{citations}", "synthesis")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            prompt = self._create_synthesis_prompt(paper_data, citations)
            response = await self._call_llm_with_retry(prompt)
            
            result = json.loads(response)
            result["timestamp"] = datetime.now().isoformat()
            result["success"] = True
            
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {e}")
            return {
                "suggestions": [{
                    "title": "Further Investigation Needed",
                    "description": "Unable to generate research directions due to service issues",
                    "rationale": "Analysis service unavailable",
                    "confidence": 0.0
                }],
                "fallback_used": True,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def format_presentation(self, analysis_results: Dict) -> Dict:
        """Format analysis results for user presentation.
        
        Args:
            analysis_results: Combined analysis results
            
        Returns:
            Formatted presentation data
        """
        try:
            # Create structured presentation format
            presentation = {
                "summary": {
                    "paper_title": analysis_results.get("paper_metadata", {}).get("title", "Unknown"),
                    "analysis_date": datetime.now().isoformat(),
                    "key_findings_count": len(analysis_results.get("paper_analysis", {}).get("findings", [])),
                    "citations_found": len(analysis_results.get("citations", [])),
                    "research_directions": len(analysis_results.get("research_directions", {}).get("suggestions", []))
                },
                "paper_analysis": analysis_results.get("paper_analysis", {}),
                "citations": analysis_results.get("citations", [])[:10],  # Top 10
                "research_directions": analysis_results.get("research_directions", {}),
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "llm_model": self.model,
                    "cache_used": any(r.get("cached", False) for r in [
                        analysis_results.get("paper_analysis", {}),
                        analysis_results.get("research_directions", {})
                    ])
                }
            }
            
            return {
                "presentation": presentation,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Presentation formatting failed: {e}")
            return {
                "presentation": {"error": "Formatting failed"},
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }