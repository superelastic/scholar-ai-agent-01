"""Demo script for the Academic Web Search Agent.

This script demonstrates the Web Search Agent's capabilities for:
1. Generating optimized search queries from paper metadata
2. Executing Google Scholar searches with retry logic
3. Filtering results by year and relevance
4. Formatting citations for presentation

Usage:
    python examples/test_web_search_agent.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import AcademicWebSearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_paper_metadata():
    """Create sample paper metadata for testing."""
    return [
        {
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani, A.', 'Shazeer, N.', 'Parmar, N.'],
            'year': 2017,
            'key_concepts': ['transformer', 'attention mechanism', 'neural machine translation'],
            'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'authors': ['Devlin, J.', 'Chang, M.', 'Lee, K.'],
            'year': 2018,
            'key_concepts': ['BERT', 'bidirectional transformers', 'pre-training', 'language understanding'],
            'abstract': 'We introduce a new language representation model called BERT...'
        },
        {
            'title': 'Language Models are Few-Shot Learners',
            'authors': ['Brown, T.', 'Mann, B.', 'Ryder, N.'],
            'year': 2020,
            'key_concepts': ['GPT-3', 'few-shot learning', 'language models', 'in-context learning'],
            'abstract': 'Recent work has demonstrated substantial gains on many NLP tasks...'
        }
    ]


async def test_web_search_agent_basic():
    """Test basic Web Search Agent functionality."""
    logger.info("=== Testing Basic Web Search Agent Functionality ===")
    
    # Configuration for testing (no API keys needed for demo)
    config = {
        'max_results': 5,
        'year_filter': 3,  # Last 3 years
        'relevance_threshold': 0.5,
        'scholar_cache_ttl': 1,  # 1 hour cache
        'scholar_interval': 0.5,  # 0.5 second intervals for demo
        'scholar_retries': 2
    }
    
    # Initialize agent
    store = {}
    agent = AcademicWebSearchAgent(store, config)
    
    logger.info(f"Initialized {agent.name}")
    logger.info(f"Configuration: max_results={agent.max_results}, year_filter={agent.year_filter}")
    
    # Test store initialization
    agent.store = agent.initialize_store()
    logger.info("Initialized agent store")
    
    # Get initial status
    status = agent.get_status()
    logger.info(f"Initial status: {status['status']}")
    
    return agent


async def test_individual_nodes():
    """Test individual Web Search Agent nodes."""
    logger.info("\n=== Testing Individual Web Search Nodes ===")
    
    from nodes import SearchQueryNode, GoogleScholarNode, CitationFilterNode, CitationFormatterNode
    from utils import LLMAnalysisUtility, ScholarSearchUtility
    
    # Mock utilities for testing (no real API calls)
    class MockLLMUtility:
        async def generate_search_queries(self, paper_metadata, max_queries=3):
            title = paper_metadata.get('title', '')
            authors = paper_metadata.get('authors', [])
            
            # Generate realistic queries based on the paper
            queries = []
            if title and authors:
                primary_author = authors[0].split(',')[0] if authors else ''
                queries.append(f'"{title}" {primary_author}')
                
                # Extract key terms
                title_words = title.split()[:3]
                if len(title_words) >= 2:
                    queries.append(' '.join(title_words))
                
                # Add author-focused query
                if primary_author:
                    queries.append(f'{primary_author} {title_words[0] if title_words else ""}')
            
            return {
                'queries': queries[:max_queries],
                'success': True
            }
    
    class MockScholarUtility:
        def search_citations(self, paper_title, authors, year, max_results=10):
            # Generate mock results based on the query
            mock_papers = [
                {
                    'title': f'Recent Advances in {paper_title.split()[0]} Research',
                    'authors': ['Smith, J.', 'Johnson, A.'],
                    'year': 2023,
                    'venue': 'ICML 2023',
                    'url': 'https://example.com/paper1',
                    'relevance_score': 0.85,
                    'citation_count': 125,
                    'abstract': f'This paper builds upon {paper_title} to propose new methods...'
                },
                {
                    'title': f'Applications of {paper_title.split()[0]} in Industry',
                    'authors': ['Brown, K.', 'Davis, L.'],
                    'year': 2022,
                    'venue': 'ICLR 2022',
                    'url': 'https://example.com/paper2',
                    'relevance_score': 0.72,
                    'citation_count': 89,
                    'abstract': f'We explore practical applications of {paper_title}...'
                },
                {
                    'title': f'Theoretical Analysis of {paper_title.split()[0]}',
                    'authors': ['Wilson, M.', 'Taylor, R.'],
                    'year': 2021,
                    'venue': 'NeurIPS 2021',
                    'url': 'https://example.com/paper3',
                    'relevance_score': 0.68,
                    'citation_count': 156,
                    'abstract': f'This work provides theoretical foundations for {paper_title}...'
                },
                {
                    'title': f'Early Work on {paper_title.split()[0]}',
                    'authors': ['Green, P.'],
                    'year': 2019,
                    'venue': 'ACL 2019',
                    'url': 'https://example.com/paper4',
                    'relevance_score': 0.45,
                    'citation_count': 34,
                    'abstract': f'An early exploration of concepts related to {paper_title}...'
                }
            ]
            
            return {
                'papers': mock_papers[:max_results],
                'success': True
            }
    
    # Test nodes individually
    sample_metadata = create_sample_paper_metadata()[0]  # Use Transformer paper
    
    # 1. Test SearchQueryNode
    logger.info("Testing SearchQueryNode...")
    llm_utility = MockLLMUtility()
    query_node = SearchQueryNode(llm_utility)
    
    context = {
        'store': {'errors': []},
        'input_data': {'paper_metadata': sample_metadata}
    }
    
    result = await query_node.process(context)
    if result['success']:
        logger.info(f"Generated queries: {result['queries']}")
    else:
        logger.error(f"Query generation failed: {result['error']}")
    
    # 2. Test GoogleScholarNode
    logger.info("\nTesting GoogleScholarNode...")
    scholar_utility = MockScholarUtility()
    scholar_node = GoogleScholarNode(scholar_utility)
    
    context['store']['search_queries'] = result['queries']
    context['store']['search_stats'] = {}
    context['config'] = {'max_results': 5}
    
    result = await scholar_node.process(context)
    if result['success']:
        logger.info(f"Found {len(result['results'])} papers")
        for i, paper in enumerate(result['results'][:2], 1):
            logger.info(f"  {i}. {paper['title']} ({paper['year']}) - Relevance: {paper['relevance_score']:.2f}")
    else:
        logger.error(f"Scholar search failed: {result['error']}")
    
    # 3. Test CitationFilterNode
    logger.info("\nTesting CitationFilterNode...")
    filter_node = CitationFilterNode()
    
    context['store']['raw_results'] = result['results']
    context['config'].update({
        'year_filter': 3,  # Last 3 years
        'relevance_threshold': 0.6
    })
    
    result = await filter_node.process(context)
    if result['success']:
        logger.info(f"Filtered to {len(result['filtered'])} relevant papers")
        logger.info(f"Filter stats: {result['stats']}")
    else:
        logger.error(f"Filtering failed: {result['error']}")
    
    # 4. Test CitationFormatterNode
    logger.info("\nTesting CitationFormatterNode...")
    formatter_node = CitationFormatterNode()
    
    context['store']['filtered_results'] = result['filtered']
    context['store']['search_queries'] = ['test query']
    
    result = await formatter_node.process(context)
    if result['success']:
        formatted = result['formatted']
        logger.info(f"Formatted {formatted['summary']['total_count']} citations")
        logger.info(f"Summary: {formatted['summary']}")
        
        # Show first citation as example
        if formatted['citations']:
            first_citation = formatted['citations'][0]
            logger.info(f"Example citation:")
            logger.info(f"  Title: {first_citation['title']}")
            logger.info(f"  Authors: {', '.join(first_citation['authors'])}")
            logger.info(f"  Year: {first_citation['year']}")
            logger.info(f"  Relevance: {first_citation['relevance_score']:.2f}")
    else:
        logger.error(f"Formatting failed: {result['error']}")


async def test_full_workflow():
    """Test the complete Web Search Agent workflow."""
    logger.info("\n=== Testing Complete Web Search Agent Workflow ===")
    
    # Create agent with mock utilities
    class MockWebSearchAgent(AcademicWebSearchAgent):
        def __init__(self, store, config):
            super().__init__(store, config)
            
            # Replace utilities with mocks for demo
            self.llm_utility = self._create_mock_llm()
            self.scholar_utility = self._create_mock_scholar()
            
            # Re-initialize nodes with mock utilities
            from nodes import SearchQueryNode, GoogleScholarNode, CitationFilterNode, CitationFormatterNode
            self.search_query_node = SearchQueryNode(self.llm_utility)
            self.google_scholar_node = GoogleScholarNode(self.scholar_utility)
            self.citation_filter_node = CitationFilterNode()
            self.citation_formatter_node = CitationFormatterNode()
        
        def _create_mock_llm(self):
            class MockLLM:
                async def generate_search_queries(self, paper_metadata, max_queries=3):
                    title = paper_metadata.get('title', '')
                    authors = paper_metadata.get('authors', [])
                    key_concepts = paper_metadata.get('key_concepts', [])
                    
                    queries = []
                    if title and authors:
                        primary_author = authors[0].split(',')[0] if authors else ''
                        queries.append(f'"{title}" {primary_author}')
                        
                        # Use key concepts for additional queries
                        if key_concepts:
                            for concept in key_concepts[:2]:
                                queries.append(f'{concept} {primary_author}')
                    
                    return {
                        'queries': queries[:max_queries],
                        'success': True
                    }
            return MockLLM()
        
        def _create_mock_scholar(self):
            class MockScholar:
                def search_citations(self, paper_title, authors, year, max_results=10):
                    # Create realistic mock results
                    import random
                    
                    base_papers = [
                        'Multi-Head Attention Networks for Enhanced Understanding',
                        'Bidirectional Encoder Representations in Modern NLP',
                        'Large-Scale Language Model Pre-training Strategies',
                        'Transformer Architectures for Computer Vision',
                        'Self-Attention Mechanisms in Deep Learning',
                        'Recent Advances in Neural Language Generation',
                        'Contextual Word Embeddings and Their Applications',
                        'End-to-End Neural Machine Translation Systems'
                    ]
                    
                    papers = []
                    for i, title in enumerate(base_papers[:max_results]):
                        papers.append({
                            'title': title,
                            'authors': [f'Author{i+1}, A.', f'Coauthor{i+1}, B.'],
                            'year': random.choice([2021, 2022, 2023, 2024]),
                            'venue': random.choice(['ICML', 'NeurIPS', 'ICLR', 'ACL', 'EMNLP']),
                            'url': f'https://example.com/paper{i+1}',
                            'relevance_score': random.uniform(0.5, 0.95),
                            'citation_count': random.randint(50, 300),
                            'abstract': f'This paper presents {title.lower()} and demonstrates improvements...'
                        })
                    
                    return {
                        'papers': papers,
                        'success': True
                    }
            return MockScholar()
    
    # Test with different papers
    sample_papers = create_sample_paper_metadata()
    
    for i, paper_metadata in enumerate(sample_papers, 1):
        logger.info(f"\n--- Testing with Paper {i}: {paper_metadata['title'][:50]}... ---")
        
        # Create fresh agent for each test
        config = {
            'max_results': 6,
            'year_filter': 3,
            'relevance_threshold': 0.6,
            'scholar_interval': 0.1
        }
        
        store = {}
        agent = MockWebSearchAgent(store, config)
        
        # Run the workflow
        input_data = {'paper_metadata': paper_metadata}
        
        try:
            result = await agent.run(input_data)
            
            if result['success']:
                logger.info(f"✓ Workflow completed successfully")
                logger.info(f"Status: {result['status']}")
                logger.info(f"Processing time: {result['processing_time']:.2f}s")
                
                # Show citation summary
                citations = result['citations']
                summary = citations['summary']
                logger.info(f"Found {summary['total_count']} citations:")
                logger.info(f"  - Recent papers (last 2 years): {summary['recent_count']}")
                logger.info(f"  - High relevance (>0.7): {summary['high_relevance_count']}")
                logger.info(f"  - Average relevance: {summary['avg_relevance']:.2f}")
                logger.info(f"  - Year range: {summary['year_range']['min']}-{summary['year_range']['max']}")
                
                # Show top 2 citations
                if citations['citations']:
                    logger.info("Top citations:")
                    for j, citation in enumerate(citations['citations'][:2], 1):
                        logger.info(f"  {j}. {citation['title']}")
                        logger.info(f"     Authors: {', '.join(citation['authors'])}")
                        logger.info(f"     Year: {citation['year']}, Relevance: {citation['relevance_score']:.2f}")
                
            else:
                logger.error(f"✗ Workflow failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"✗ Workflow exception: {e}")


async def test_error_handling():
    """Test Web Search Agent error handling."""
    logger.info("\n=== Testing Error Handling ===")
    
    from agents import AcademicWebSearchAgent
    
    # Test with minimal config
    config = {'max_results': 5}
    store = {}
    agent = AcademicWebSearchAgent(store, config)
    
    # Test 1: Missing metadata
    logger.info("Testing missing metadata...")
    result = await agent.run({})
    assert result['success'] is False
    logger.info(f"✓ Correctly handled missing metadata: {result['error']}")
    
    # Test 2: Empty metadata
    logger.info("Testing empty metadata...")
    result = await agent.run({'paper_metadata': {}})
    assert result['success'] is False
    logger.info(f"✓ Correctly handled empty metadata: {result['error']}")
    
    logger.info("Error handling tests completed successfully")


def save_results_to_file(results, filename="web_search_results.json"):
    """Save results to a JSON file for inspection."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


async def main():
    """Run all Web Search Agent demos."""
    logger.info("Starting Web Search Agent Demo")
    logger.info("="*60)
    
    try:
        # Test basic functionality
        agent = await test_web_search_agent_basic()
        
        # Test individual nodes
        await test_individual_nodes()
        
        # Test complete workflow
        await test_full_workflow()
        
        # Test error handling
        await test_error_handling()
        
        logger.info("\n" + "="*60)
        logger.info("Web Search Agent Demo completed successfully!")
        logger.info("The Web Search Agent is ready for:")
        logger.info("  • Generating optimized search queries from paper metadata")
        logger.info("  • Executing Google Scholar searches with retry logic")
        logger.info("  • Filtering results by year and relevance thresholds")
        logger.info("  • Formatting citations with comprehensive metadata")
        logger.info("  • Handling errors gracefully with fallback mechanisms")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())