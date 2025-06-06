"""Example script demonstrating LLM analysis functionality."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import LLMAnalysisUtility, PDFExtractorUtility, create_sample_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_llm_analysis():
    """Demonstrate LLM analysis capabilities."""
    logger.info("Starting LLM Analysis Demo")
    
    # Initialize utilities
    # Note: In real usage, provide actual API keys via environment variables
    llm_utility = LLMAnalysisUtility(
        openai_api_key="demo_key",  # Replace with actual key
        model="gpt-4",
        timeout=30
    )
    
    pdf_extractor = PDFExtractorUtility()
    
    # Create and process a sample PDF
    logger.info("Creating sample PDF for analysis...")
    sample_pdf_path = create_sample_pdf()
    
    try:
        # Extract text and metadata from PDF
        logger.info("Extracting text and metadata from PDF...")
        text_result = pdf_extractor.extract_text(sample_pdf_path)
        metadata_result = pdf_extractor.extract_metadata(sample_pdf_path)
        
        paper_text = text_result["text"]
        paper_metadata = metadata_result["metadata"]
        
        logger.info(f"Extracted {len(paper_text)} characters of text")
        logger.info(f"Paper title: {paper_metadata.get('title', 'Unknown')}")
        
        # Demo 1: Paper Analysis (with mock since we don't have real API keys)
        logger.info("\n=== Demo 1: Paper Analysis ===")
        
        # Simulate successful analysis (replace with real LLM call when API key available)
        mock_analysis = {
            "key_concepts": [
                "deep learning", "natural language processing", 
                "transformer architecture", "attention mechanism"
            ],
            "methodology": "Systematic review of deep learning literature with focus on transformer models",
            "findings": [
                "Transformer models consistently outperform previous architectures",
                "Pre-training on large corpora significantly improves performance",
                "Attention mechanisms enable better long-range dependency modeling"
            ],
            "theoretical_framework": "Attention-based neural network architectures",
            "limitations": [
                "Computational requirements are substantial",
                "Limited analysis of low-resource languages"
            ],
            "future_work": [
                "Investigate efficiency improvements",
                "Explore applications to multimodal data",
                "Develop better evaluation metrics"
            ],
            "success": True,
            "timestamp": "2025-06-05T10:00:00"
        }
        
        logger.info("✓ Paper analysis completed (simulated)")
        logger.info(f"Found {len(mock_analysis['key_concepts'])} key concepts")
        logger.info(f"Identified {len(mock_analysis['findings'])} main findings")
        
        # Demo 2: Search Query Generation
        logger.info("\n=== Demo 2: Search Query Generation ===")
        
        enhanced_metadata = {**paper_metadata, **{"key_concepts": mock_analysis["key_concepts"]}}
        
        mock_queries = {
            "queries": [
                '"Deep Learning for Natural Language Processing"',
                '"Deep Learning for Natural Language Processing" transformer',
                'attention mechanism neural networks 2025',
                '"John Doe" "Jane Smith" deep learning survey',
                'transformer architecture NLP applications'
            ],
            "success": True,
            "timestamp": "2025-06-05T10:01:00"
        }
        
        logger.info("✓ Search queries generated (simulated)")
        for i, query in enumerate(mock_queries["queries"], 1):
            logger.info(f"  {i}. {query}")
        
        # Demo 3: Research Direction Synthesis
        logger.info("\n=== Demo 3: Research Direction Synthesis ===")
        
        mock_citations = [
            {"title": "Efficient Transformer Training Methods", "year": 2024},
            {"title": "Multimodal Learning with Transformers", "year": 2024},
            {"title": "Low-Resource NLP with Pre-trained Models", "year": 2025}
        ]
        
        mock_directions = {
            "suggestions": [
                {
                    "title": "Energy-Efficient Transformer Architectures",
                    "description": "Develop transformer variants that require significantly less computational power while maintaining performance",
                    "rationale": "Current transformers are computationally expensive, limiting deployment in resource-constrained environments",
                    "confidence": 0.85
                },
                {
                    "title": "Cross-lingual Transfer Learning for Low-Resource Languages",
                    "description": "Investigate how knowledge from high-resource languages can better transfer to languages with limited training data",
                    "rationale": "Many languages lack sufficient data for effective model training, but cross-lingual methods show promise",
                    "confidence": 0.78
                },
                {
                    "title": "Interpretable Attention Mechanisms",
                    "description": "Create attention mechanisms that provide more interpretable insights into model decision-making",
                    "rationale": "Understanding how models make decisions is crucial for trust and debugging in critical applications",
                    "confidence": 0.72
                }
            ],
            "success": True,
            "timestamp": "2025-06-05T10:02:00"
        }
        
        logger.info("✓ Research directions synthesized (simulated)")
        for i, suggestion in enumerate(mock_directions["suggestions"], 1):
            logger.info(f"  {i}. {suggestion['title']} (confidence: {suggestion['confidence']:.2f})")
            logger.info(f"     {suggestion['description'][:100]}...")
        
        # Demo 4: Presentation Formatting
        logger.info("\n=== Demo 4: Presentation Formatting ===")
        
        combined_results = {
            "paper_metadata": enhanced_metadata,
            "paper_analysis": mock_analysis,
            "citations": mock_citations,
            "research_directions": mock_directions
        }
        
        presentation = await llm_utility.format_presentation(combined_results)
        
        logger.info("✓ Presentation formatted successfully")
        logger.info("Summary:")
        summary = presentation["presentation"]["summary"]
        logger.info(f"  - Key findings: {summary['key_findings_count']}")
        logger.info(f"  - Citations found: {summary['citations_found']}")
        logger.info(f"  - Research directions: {summary['research_directions']}")
        
        # Demo 5: Caching Mechanism
        logger.info("\n=== Demo 5: Caching Mechanism ===")
        
        # Simulate cache operations
        cache_key = llm_utility._get_cache_key("test content", "analysis")
        test_data = {"cached": True, "test": "data"}
        
        llm_utility._save_to_cache(cache_key, test_data)
        cached_result = llm_utility._get_cached_result(cache_key)
        
        if cached_result:
            logger.info("✓ Caching mechanism working correctly")
            logger.info(f"  Cached data retrieved: {cached_result}")
        else:
            logger.warning("✗ Caching mechanism not working")
        
        logger.info("\n=== Demo Complete ===")
        logger.info("Note: This demo uses simulated responses. To use with real LLM APIs:")
        logger.info("1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
        logger.info("2. Remove mock responses and use actual LLM calls")
        logger.info("3. Handle rate limits and costs appropriately")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        try:
            Path(sample_pdf_path).unlink()
            logger.info("Cleaned up sample PDF")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(demo_llm_analysis())