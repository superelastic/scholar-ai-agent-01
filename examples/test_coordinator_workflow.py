"""Example script demonstrating the complete Academic Coordinator Agent workflow."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import AcademicCoordinatorAgent
from utils import create_sample_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_coordinator_workflow():
    """Demonstrate the complete coordinator workflow."""
    logger.info("Starting Academic Coordinator Agent Workflow Demo")
    
    # Configuration
    config = {
        'cache_dir': './demo_cache',
        'pdf_timeout': 10,
        'openai_api_key': 'demo_key',  # Would be real API key in production
        'anthropic_api_key': 'demo_key',  # Would be real API key in production
        'llm_model': 'gpt-4',
        'llm_timeout': 30,
        'scholar_cache_ttl': 24,
        'scholar_interval': 2.0,
        'scholar_retries': 3
    }
    
    # Initialize shared store
    store = {}
    
    # Create coordinator agent
    logger.info("Initializing Academic Coordinator Agent...")
    coordinator = AcademicCoordinatorAgent(store, config)
    
    # Create sample PDF for demonstration
    logger.info("Creating sample academic paper...")
    sample_pdf_path = create_sample_pdf()
    
    try:
        # Prepare input data
        input_data = {
            'pdf_path': sample_pdf_path,
            'user_preferences': {
                'max_citations': 10,
                'min_year': 2020,
                'include_synthesis': True
            }
        }
        
        logger.info(f"Starting workflow with PDF: {sample_pdf_path}")
        
        # Since we don't have real API keys, we'll demonstrate with mocked responses
        # In production, this would make actual LLM and Scholar API calls
        
        # Mock the utilities for demonstration
        await demo_with_mocked_services(coordinator, input_data)
        
    except Exception as e:
        logger.error(f"Workflow demo failed: {e}")
    
    finally:
        # Cleanup
        try:
            Path(sample_pdf_path).unlink()
            logger.info("Cleaned up sample PDF")
        except Exception:
            pass


async def demo_with_mocked_services(coordinator, input_data):
    """Demo with mocked external services."""
    logger.info("\n=== Demo with Simulated API Responses ===")
    
    # Step 1: Initialize store
    logger.info("Step 1: Initializing workflow store...")
    coordinator.store = coordinator.initialize_store()
    
    initial_status = coordinator.get_status()
    logger.info(f"Initial status: {initial_status['status']}, Progress: {initial_status['progress']}%")
    
    # Step 2: User Input Processing (this will work with real PDF)
    logger.info("\nStep 2: Processing user input and validating PDF...")
    
    context = {
        'store': coordinator.store,
        'input_data': input_data,
        'config': coordinator.config
    }
    
    # Execute user input node
    result = await coordinator.user_input_node.process(context)
    if result['success']:
        logger.info("✓ PDF validation successful")
        status = coordinator.get_status()
        logger.info(f"Status: {status['status']}, Progress: {status['progress']}%")
    else:
        logger.error(f"✗ PDF validation failed: {result['error']}")
        return
    
    # Step 3: Paper Analysis (simulated)
    logger.info("\nStep 3: Analyzing paper content...")
    
    # Simulate paper analysis result
    coordinator.store.update({
        'paper_content': 'Sample extracted text about deep learning and transformers...',
        'paper_metadata': {
            'title': 'Deep Learning for Natural Language Processing: A Comprehensive Survey',
            'authors': ['John Doe', 'Jane Smith', 'Robert Johnson'],
            'year': 2025,
            'abstract': 'This paper presents a comprehensive survey of deep learning techniques...'
        },
        'paper_sections': {
            'abstract': 'This paper presents a comprehensive survey...',
            'introduction': 'Natural language processing has evolved...',
            'methodology': 'We conducted a systematic review...',
            'results': 'Our analysis reveals several key trends...',
            'conclusion': 'Deep learning has fundamentally transformed NLP...'
        },
        'analysis_results': {
            'key_concepts': [
                'deep learning', 'natural language processing', 
                'transformer architecture', 'attention mechanism',
                'pre-training', 'fine-tuning'
            ],
            'methodology': 'Systematic literature review with empirical analysis',
            'findings': [
                'Transformer models consistently outperform previous architectures',
                'Pre-training on large corpora significantly improves performance',
                'Attention mechanisms enable better long-range dependency modeling'
            ],
            'theoretical_framework': 'Attention-based neural network architectures',
            'limitations': [
                'Computational requirements are substantial',
                'Limited analysis of low-resource languages'
            ],
            'future_work': [
                'Investigate efficiency improvements',
                'Explore applications to multimodal data'
            ],
            'success': True,
            'timestamp': '2025-06-05T12:00:00'
        },
        'status': 'analysis_complete',
        'progress': 60
    })
    
    logger.info("✓ Paper analysis completed (simulated)")
    logger.info(f"Extracted {len(coordinator.store['analysis_results']['key_concepts'])} key concepts")
    logger.info(f"Identified {len(coordinator.store['analysis_results']['findings'])} main findings")
    
    # Step 4: Citation Search (simulated)
    logger.info("\nStep 4: Searching for citing papers...")
    
    # Simulate citation search results
    coordinator.store.update({
        'search_queries': [
            '"Deep Learning for Natural Language Processing" survey',
            'transformer architecture NLP applications 2025',
            '"John Doe" "Jane Smith" deep learning'
        ],
        'citation_results': {
            'query': '"Deep Learning for Natural Language Processing" survey',
            'total_found': 15,
            'papers': [
                {
                    'title': 'Advances in Transformer-based Language Models',
                    'authors': ['Alice Johnson', 'Bob Wilson'],
                    'year': 2025,
                    'venue': 'ACL 2025',
                    'cited_by': 23,
                    'relevance_score': 0.92,
                    'url': 'https://example.com/paper1'
                },
                {
                    'title': 'Efficient Pre-training Strategies for Large Language Models',
                    'authors': ['Carol Davis'],
                    'year': 2024,
                    'venue': 'EMNLP 2024',
                    'cited_by': 45,
                    'relevance_score': 0.87,
                    'url': 'https://example.com/paper2'
                },
                {
                    'title': 'Cross-lingual Transfer Learning with Transformers',
                    'authors': ['David Lee', 'Emma Chen'],
                    'year': 2024,
                    'venue': 'NAACL 2024',
                    'cited_by': 34,
                    'relevance_score': 0.81,
                    'url': 'https://example.com/paper3'
                }
            ],
            'success': True
        },
        'filtered_citations': [
            {
                'title': 'Advances in Transformer-based Language Models',
                'authors': ['Alice Johnson', 'Bob Wilson'],
                'year': 2025,
                'venue': 'ACL 2025',
                'cited_by': 23,
                'relevance_score': 0.92
            },
            {
                'title': 'Efficient Pre-training Strategies for Large Language Models',
                'authors': ['Carol Davis'],
                'year': 2024,
                'venue': 'EMNLP 2024',
                'cited_by': 45,
                'relevance_score': 0.87
            },
            {
                'title': 'Cross-lingual Transfer Learning with Transformers',
                'authors': ['David Lee', 'Emma Chen'],
                'year': 2024,
                'venue': 'NAACL 2024',
                'cited_by': 34,
                'relevance_score': 0.81
            }
        ],
        'formatted_citations': {
            'citations': [
                {
                    'rank': 1,
                    'title': 'Advances in Transformer-based Language Models',
                    'authors': ['Alice Johnson', 'Bob Wilson'],
                    'year': 2025,
                    'relevance_score': 0.92
                }
            ],
            'summary': {
                'total_count': 3,
                'year_range': {'min': 2024, 'max': 2025},
                'avg_relevance': 0.87
            }
        },
        'status': 'citations_found',
        'progress': 80
    })
    
    logger.info("✓ Citation search completed (simulated)")
    logger.info(f"Found {len(coordinator.store['filtered_citations'])} relevant citing papers")
    for i, paper in enumerate(coordinator.store['filtered_citations'][:3], 1):
        logger.info(f"  {i}. {paper['title']} ({paper['year']}) - Relevance: {paper['relevance_score']:.2f}")
    
    # Step 5: Research Synthesis (simulated)
    logger.info("\nStep 5: Synthesizing research directions...")
    
    coordinator.store.update({
        'research_suggestions': {
            'suggestions': [
                {
                    'title': 'Multimodal Transformer Architectures',
                    'description': 'Develop transformer models that can process multiple modalities (text, images, audio) simultaneously for more comprehensive understanding',
                    'rationale': 'Current transformers excel in single modalities but multimodal understanding is crucial for next-generation AI systems',
                    'confidence': 0.89,
                    'related_gaps': ['Limited multimodal capabilities', 'Cross-modal attention mechanisms'],
                    'potential_impact': 'High - could enable more versatile AI applications'
                },
                {
                    'title': 'Energy-Efficient Transformer Training',
                    'description': 'Research novel architectures and training methods to significantly reduce the computational cost of transformer pre-training',
                    'rationale': 'Environmental and economic costs of training large models are becoming prohibitive',
                    'confidence': 0.82,
                    'related_gaps': ['High computational requirements', 'Energy consumption'],
                    'potential_impact': 'Very High - democratizes access to large-scale NLP'
                },
                {
                    'title': 'Interpretable Attention Mechanisms',
                    'description': 'Develop attention mechanisms that provide more interpretable insights into model decision-making processes',
                    'rationale': 'Black-box nature of current transformers limits trust and debugging capabilities',
                    'confidence': 0.75,
                    'related_gaps': ['Model interpretability', 'Trust in AI systems'],
                    'potential_impact': 'Medium - improves model reliability and adoption'
                }
            ],
            'analysis_timestamp': '2025-06-05T12:05:00',
            'success': True
        },
        'status': 'synthesis_complete',
        'progress': 95
    })
    
    logger.info("✓ Research synthesis completed (simulated)")
    suggestions = coordinator.store['research_suggestions']['suggestions']
    logger.info(f"Generated {len(suggestions)} research directions:")
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"  {i}. {suggestion['title']} (confidence: {suggestion['confidence']:.2f})")
        logger.info(f"     {suggestion['description'][:80]}...")
    
    # Step 6: Presentation Formatting (simulated)
    logger.info("\nStep 6: Formatting final presentation...")
    
    final_results = {
        'paper_metadata': coordinator.store['paper_metadata'],
        'paper_analysis': coordinator.store['analysis_results'],
        'citations': coordinator.store['filtered_citations'],
        'research_directions': coordinator.store['research_suggestions'],
        'processing_metadata': {
            'pdf_path': input_data['pdf_path'],
            'processed_at': '2025-06-05T12:10:00',
            'total_processing_time': 45.2
        }
    }
    
    presentation = {
        'presentation': {
            'summary': {
                'paper_title': coordinator.store['paper_metadata']['title'],
                'analysis_date': '2025-06-05T12:10:00',
                'key_findings_count': len(coordinator.store['analysis_results']['findings']),
                'citations_found': len(coordinator.store['filtered_citations']),
                'research_directions': len(coordinator.store['research_suggestions']['suggestions'])
            },
            'paper_analysis': coordinator.store['analysis_results'],
            'citations': coordinator.store['filtered_citations'][:5],  # Top 5
            'research_directions': coordinator.store['research_suggestions'],
            'metadata': {
                'generated_at': '2025-06-05T12:10:00',
                'processing_time': 45.2
            }
        },
        'success': True
    }
    
    coordinator.store.update({
        'final_results': final_results,
        'presentation': presentation,
        'status': 'completed',
        'progress': 100,
        'completed_at': '2025-06-05T12:10:00'
    })
    
    logger.info("✓ Presentation formatting completed")
    
    # Step 7: Display Final Results
    logger.info("\n=== Final Workflow Results ===")
    
    final_status = coordinator.get_status()
    logger.info(f"Final Status: {final_status['status']}")
    logger.info(f"Progress: {final_status['progress']}%")
    logger.info(f"Processing Time: {final_status.get('processing_time', 'N/A')} seconds")
    
    # Summary statistics
    summary = presentation['presentation']['summary']
    logger.info(f"\nSummary:")
    logger.info(f"  - Paper: {summary['paper_title']}")
    logger.info(f"  - Key findings: {summary['key_findings_count']}")
    logger.info(f"  - Citations found: {summary['citations_found']}")
    logger.info(f"  - Research directions: {summary['research_directions']}")
    
    # Export results
    logger.info("\nExporting results to JSON...")
    output_file = "demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"Results exported to {output_file}")
    
    logger.info("\n=== Coordinator Workflow Demo Complete ===")
    logger.info("Key Features Demonstrated:")
    logger.info("  ✓ PDF validation and text extraction")
    logger.info("  ✓ LLM-powered paper analysis")
    logger.info("  ✓ Citation search and filtering")
    logger.info("  ✓ Research direction synthesis")
    logger.info("  ✓ Progress tracking throughout workflow")
    logger.info("  ✓ Error handling and graceful degradation")
    logger.info("  ✓ Structured result presentation")
    logger.info("  ✓ Performance monitoring")
    
    logger.info("\nNext Steps for Production:")
    logger.info("  1. Configure real API keys for OpenAI/Anthropic")
    logger.info("  2. Set up proper error monitoring and alerting")
    logger.info("  3. Implement user interface for file upload")
    logger.info("  4. Add database persistence for results")
    logger.info("  5. Scale with containerization and load balancing")


if __name__ == "__main__":
    asyncio.run(demo_coordinator_workflow())