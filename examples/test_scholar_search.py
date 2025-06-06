"""Example script demonstrating Google Scholar search functionality."""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import ScholarSearchUtility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_scholar_search():
    """Demonstrate Google Scholar search capabilities."""
    logger.info("Starting Google Scholar Search Demo")
    
    # Initialize Scholar search utility
    scholar_utility = ScholarSearchUtility(
        cache_ttl_hours=24,
        min_request_interval=2.0,  # Be respectful to Google
        max_retries=3,
        timeout=10
    )
    
    # Sample papers to search for citations
    test_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": 2017,
            "description": "Original Transformer paper"
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2018,
            "description": "BERT paper"
        },
        {
            "title": "Deep Learning",
            "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
            "year": 2016,
            "description": "Deep Learning textbook"
        }
    ]
    
    for i, paper in enumerate(test_papers, 1):
        logger.info(f"\n=== Demo {i}: Searching citations for {paper['description']} ===")
        logger.info(f"Paper: {paper['title']}")
        logger.info(f"Authors: {', '.join(paper['authors'])}")
        logger.info(f"Year: {paper['year']}")
        
        try:
            # Demo 1: Basic Citation Search
            logger.info("\n--- Step 1: Searching for citing papers ---")
            
            # Note: In a real demo, this would make actual requests to Google Scholar
            # For demonstration purposes, we'll simulate the results
            
            # Simulate search results
            mock_results = {
                'query': f'"{paper["title"][:30]}" {paper["authors"][0].split()[-1]}',
                'total_found': 15,
                'papers': [
                    {
                        'title': f'Advances in {paper["description"]} Applications',
                        'authors': ['Recent Author', 'Another Author'],
                        'year': 2023,
                        'venue': 'Journal of AI Research',
                        'url': 'https://example.com/paper1',
                        'cited_by': 45,
                        'relevance_score': 0.85,
                        'snippet': f'Building on the work of {paper["authors"][0]}, we propose...'
                    },
                    {
                        'title': f'Improving {paper["description"]} with Novel Methods',
                        'authors': ['Expert Researcher'],
                        'year': 2022,
                        'venue': 'Conference on Machine Learning',
                        'url': 'https://example.com/paper2',
                        'cited_by': 123,
                        'relevance_score': 0.78,
                        'snippet': f'The seminal work by {paper["authors"][0]} et al. showed...'
                    },
                    {
                        'title': f'Survey of {paper["description"]} Techniques',
                        'authors': ['Survey Author', 'Review Author'],
                        'year': 2024,
                        'venue': 'AI Review',
                        'url': 'https://example.com/paper3',
                        'cited_by': 12,
                        'relevance_score': 0.72,
                        'snippet': f'Since the introduction of {paper["title"][:20]}...'
                    }
                ],
                'search_timestamp': '2025-06-05T10:00:00',
                'success': True
            }
            
            logger.info(f"✓ Search completed (simulated)")
            logger.info(f"Query used: {mock_results['query']}")
            logger.info(f"Total papers found: {mock_results['total_found']}")
            
            # Demo 2: Result Filtering
            logger.info("\n--- Step 2: Filtering results ---")
            
            # Filter recent papers with high relevance
            filtered_results = scholar_utility.filter_results(
                mock_results['papers'],
                min_year=2022,
                relevance_threshold=0.7
            )
            
            logger.info(f"✓ Filtered to {len(filtered_results)} high-quality recent papers")
            for j, result in enumerate(filtered_results, 1):
                logger.info(f"  {j}. {result['title']} ({result['year']}) - Relevance: {result['relevance_score']:.2f}")
            
            # Demo 3: Citation Formatting
            logger.info("\n--- Step 3: Formatting citations ---")
            
            formatted = scholar_utility.format_citations(filtered_results)
            
            logger.info("✓ Citations formatted for presentation")
            logger.info(f"Summary:")
            logger.info(f"  - Total citations: {formatted['summary']['total_count']}")
            if formatted['summary']['year_range']:
                logger.info(f"  - Year range: {formatted['summary']['year_range']['min']}-{formatted['summary']['year_range']['max']}")
            logger.info(f"  - Average relevance: {formatted['summary']['avg_relevance']:.2f}")
            
            logger.info("\nTop citations:")
            for citation in formatted['citations'][:3]:  # Show top 3
                logger.info(f"  {citation['rank']}. {citation['title']}")
                logger.info(f"     Authors: {', '.join(citation['authors'])}")
                logger.info(f"     Year: {citation['year']}, Cited by: {citation['cited_by']}")
                logger.info(f"     Relevance: {citation['relevance_score']:.2f}")
                if citation['url']:
                    logger.info(f"     URL: {citation['url']}")
                logger.info("")
            
            # Demo 4: Cache Performance
            logger.info("\n--- Step 4: Demonstrating cache performance ---")
            
            # Simulate cache operations
            cache_key = scholar_utility._get_cache_key(
                paper["title"], 
                paper["authors"], 
                paper["year"]
            )
            
            # Save to cache
            scholar_utility._save_to_cache(cache_key, mock_results)
            
            # Retrieve from cache
            cached_result = scholar_utility._get_cached_result(cache_key)
            
            if cached_result:
                logger.info("✓ Cache mechanism working correctly")
                logger.info(f"  Cached {len(cached_result['papers'])} papers")
            else:
                logger.warning("✗ Cache mechanism not working")
            
        except Exception as e:
            logger.error(f"Demo failed for {paper['description']}: {e}")
        
        if i < len(test_papers):
            logger.info("\n" + "="*60)
    
    # Demo 5: Rate Limiting and Best Practices
    logger.info("\n=== Demo 5: Rate Limiting and Best Practices ===")
    
    logger.info("Rate limiting configuration:")
    logger.info(f"  - Minimum interval: {scholar_utility.min_request_interval} seconds")
    logger.info(f"  - Max retries: {scholar_utility.max_retries}")
    logger.info(f"  - Cache TTL: {scholar_utility.cache_ttl}")
    logger.info(f"  - Request timeout: {scholar_utility.timeout} seconds")
    
    logger.info("\nBest practices demonstrated:")
    logger.info("  ✓ Respectful rate limiting (2+ seconds between requests)")
    logger.info("  ✓ Caching to minimize repeated requests")
    logger.info("  ✓ Random user agent rotation")
    logger.info("  ✓ Robust error handling and retries")
    logger.info("  ✓ Relevance scoring for result ranking")
    
    # Demo 6: Integration Example
    logger.info("\n=== Demo 6: Integration with Other Components ===")
    
    logger.info("This Scholar search utility integrates with:")
    logger.info("  - PDF Extractor: Uses extracted paper metadata for searches")
    logger.info("  - LLM Analysis: Uses LLM-generated queries for better results")
    logger.info("  - Citation formatting: Prepares results for presentation nodes")
    
    example_integration = {
        "pdf_metadata": {
            "title": "Example Academic Paper",
            "authors": ["Research Author"],
            "year": 2020
        },
        "llm_queries": [
            '"Example Academic Paper" "Research Author"',
            'academic research 2020 methodology',
            '"Research Author" citation analysis'
        ],
        "scholar_results": "Would contain actual citing papers...",
        "formatted_output": "Ready for presentation to user"
    }
    
    logger.info(f"Integration workflow:")
    for step, content in example_integration.items():
        logger.info(f"  {step}: {content}")
    
    logger.info("\n=== Demo Complete ===")
    logger.info("Note: This demo uses simulated responses. To use with real Google Scholar:")
    logger.info("1. Ensure you have internet connectivity")
    logger.info("2. Be respectful of Google's terms of service")
    logger.info("3. Implement proper rate limiting (already included)")
    logger.info("4. Consider using proxies for high-volume usage")
    logger.info("5. Handle IP blocking gracefully")
    logger.info("6. Cache results to minimize requests")


if __name__ == "__main__":
    demo_scholar_search()