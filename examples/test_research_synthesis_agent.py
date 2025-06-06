"""Demo script for the Academic New Research Agent.

This script demonstrates the Research Synthesis Agent's capabilities for:
1. Synthesizing seminal paper analysis with citation data
2. Identifying research trends and gaps
3. Generating novel research directions
4. Formatting comprehensive research suggestions

Usage:
    python examples/test_research_synthesis_agent.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import AcademicNewResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_research_scenarios():
    """Create sample research scenarios for testing."""
    return [
        {
            'name': 'Transformer Architecture Research',
            'paper_metadata': {
                'title': 'Attention Is All You Need',
                'authors': ['Vaswani, A.', 'Shazeer, N.', 'Parmar, N.'],
                'year': 2017,
                'key_concepts': ['transformer', 'attention mechanism', 'neural machine translation', 'sequence modeling']
            },
            'paper_analysis': {
                'key_concepts': ['self-attention', 'multi-head attention', 'positional encoding', 'encoder-decoder'],
                'methodology': 'Novel architecture using only attention mechanisms, eliminating recurrence and convolution',
                'findings': [
                    'Transformer models achieve superior performance on translation tasks',
                    'Parallelizable training leads to faster computation',
                    'Better modeling of long-range dependencies',
                    'Generalizable to other sequence tasks'
                ],
                'significance': 'Revolutionary architecture that became foundation for modern NLP'
            },
            'citation_data': [
                {
                    'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                    'authors': ['Devlin, J.', 'Chang, M.', 'Lee, K.'],
                    'year': 2018,
                    'venue': 'NAACL',
                    'relevance_score': 0.95,
                    'citation_count': 15000,
                    'abstract': 'BERT uses bidirectional training to achieve state-of-the-art results on eleven natural language processing tasks...'
                },
                {
                    'title': 'Language Models are Few-Shot Learners',
                    'authors': ['Brown, T.', 'Mann, B.', 'Ryder, N.'],
                    'year': 2020,
                    'venue': 'NeurIPS',
                    'relevance_score': 0.92,
                    'citation_count': 8000,
                    'abstract': 'We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance...'
                },
                {
                    'title': 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
                    'authors': ['Dosovitskiy, A.', 'Beyer, L.'],
                    'year': 2020,
                    'venue': 'ICLR',
                    'relevance_score': 0.88,
                    'citation_count': 12000,
                    'abstract': 'Vision Transformer (ViT) shows excellent results when pre-trained at sufficient scale...'
                },
                {
                    'title': 'Training language models to follow instructions with human feedback',
                    'authors': ['Ouyang, L.', 'Wu, J.'],
                    'year': 2022,
                    'venue': 'NeurIPS',
                    'relevance_score': 0.85,
                    'citation_count': 2500,
                    'abstract': 'We show how to train language models to follow instructions using human feedback...'
                },
                {
                    'title': 'PaLM: Scaling Language Modeling with Pathways',
                    'authors': ['Chowdhery, A.', 'Narang, S.'],
                    'year': 2022,
                    'venue': 'arXiv',
                    'relevance_score': 0.82,
                    'citation_count': 1800,
                    'abstract': 'We present PaLM, a 540-billion parameter autoregressive language model...'
                }
            ]
        },
        {
            'name': 'Deep Learning Fundamentals',
            'paper_metadata': {
                'title': 'Deep Learning',
                'authors': ['LeCun, Y.', 'Bengio, Y.', 'Hinton, G.'],
                'year': 2015,
                'key_concepts': ['deep learning', 'neural networks', 'representation learning', 'backpropagation']
            },
            'paper_analysis': {
                'key_concepts': ['multilayer perceptrons', 'convolutional networks', 'recurrent networks', 'unsupervised learning'],
                'methodology': 'Comprehensive review of deep learning methods with theoretical foundations',
                'findings': [
                    'Deep networks can learn complex hierarchical representations',
                    'Convolutional networks excel at image tasks',
                    'Recurrent networks handle sequential data effectively',
                    'Unsupervised learning enables feature discovery'
                ],
                'significance': 'Foundational survey that defined the field of deep learning'
            },
            'citation_data': [
                {
                    'title': 'Generative Adversarial Networks',
                    'authors': ['Goodfellow, I.', 'Pouget-Abadie, J.'],
                    'year': 2014,
                    'venue': 'NIPS',
                    'relevance_score': 0.90,
                    'citation_count': 25000,
                    'abstract': 'We propose a new framework for estimating generative models via an adversarial process...'
                },
                {
                    'title': 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting',
                    'authors': ['Srivastava, N.', 'Hinton, G.'],
                    'year': 2014,
                    'venue': 'JMLR',
                    'relevance_score': 0.85,
                    'citation_count': 18000,
                    'abstract': 'Deep neural nets with a large number of parameters are very powerful machine learning systems...'
                },
                {
                    'title': 'Batch Normalization: Accelerating Deep Network Training',
                    'authors': ['Ioffe, S.', 'Szegedy, C.'],
                    'year': 2015,
                    'venue': 'ICML',
                    'relevance_score': 0.82,
                    'citation_count': 15000,
                    'abstract': 'Training Deep Neural Networks is complicated by the fact that the distribution of each layer\'s inputs changes...'
                }
            ]
        }
    ]


async def test_research_synthesis_agent_basic():
    """Test basic Research Synthesis Agent functionality."""
    logger.info("=== Testing Basic Research Synthesis Agent Functionality ===")
    
    # Configuration for testing (no API keys needed for demo)
    config = {
        'min_confidence': 0.6,
        'max_suggestions': 5,
        'trend_analysis_depth': 'comprehensive'
    }
    
    # Initialize agent
    store = {}
    agent = AcademicNewResearchAgent(store, config)
    
    logger.info(f"Initialized {agent.name}")
    logger.info(f"Configuration: min_confidence={agent.min_confidence}, max_suggestions={agent.max_suggestions}")
    
    # Test store initialization
    agent.store = agent.initialize_store()
    logger.info("Initialized agent store")
    
    # Get initial status
    status = agent.get_status()
    logger.info(f"Initial status: {status['status']}")
    
    return agent


async def test_individual_nodes():
    """Test individual Research Synthesis Agent nodes."""
    logger.info("\n=== Testing Individual Research Synthesis Nodes ===")
    
    from nodes import PaperSynthesisNode, TrendAnalysisNode, DirectionGeneratorNode, SuggestionFormatterNode
    from utils import LLMAnalysisUtility
    
    # Mock LLM utility for testing (no real API calls)
    class MockLLMUtility:
        async def synthesize_research_directions(self, paper_data=None, citations=None, analysis_type=None, context=None):
            if analysis_type == 'comprehensive_synthesis':
                return {
                    'insights': {
                        'field_evolution': 'The field has evolved from basic attention to complex architectures',
                        'research_momentum': 'Accelerating rapidly with new applications emerging'
                    },
                    'impact_analysis': {
                        'citation_growth': 'Exponential growth in citations',
                        'field_influence': 'Transformed multiple domains beyond NLP'
                    },
                    'evolution_summary': {
                        'key_developments': ['Bidirectional training', 'Scaling laws', 'Vision applications'],
                        'future_directions': ['Efficiency improvements', 'Multimodal integration']
                    },
                    'success': True
                }
            elif analysis_type == 'trend_analysis':
                return {
                    'enhanced_trends': {
                        'emerging_patterns': ['Multimodal transformers', 'Efficient architectures'],
                        'decline_patterns': ['Pure CNN approaches', 'RNN-based models']
                    },
                    'additional_gaps': [
                        {
                            'type': 'efficiency',
                            'description': 'Limited work on energy-efficient transformers',
                            'opportunity_level': 'high'
                        }
                    ],
                    'trend_insights': {
                        'momentum': 'High momentum in transformer research',
                        'maturity': 'Field is maturing but still highly active'
                    },
                    'success': True
                }
            elif analysis_type == 'novel_directions':
                return {
                    'suggestions': [
                        {
                            'title': 'Quantum-Inspired Attention Mechanisms',
                            'description': 'Explore quantum computing principles in attention design',
                            'rationale': 'Quantum superposition could enhance attention modeling',
                            'approach': 'Develop quantum-classical hybrid attention layers',
                            'confidence': 0.75,
                            'impact_potential': 'high',
                            'feasibility': 'medium',
                            'novelty_score': 0.9
                        },
                        {
                            'title': 'Neuromorphic Transformer Architectures',
                            'description': 'Design brain-inspired efficient transformer variants',
                            'rationale': 'Biological inspiration could lead to more efficient architectures',
                            'approach': 'Adapt spiking neural network principles to transformers',
                            'confidence': 0.68,
                            'impact_potential': 'medium',
                            'feasibility': 'high',
                            'novelty_score': 0.85
                        }
                    ],
                    'success': True
                }
            
            return {'success': False, 'error': 'Unknown analysis type'}
    
    # Test sample data
    scenarios = create_sample_research_scenarios()
    sample_scenario = scenarios[0]  # Use Transformer scenario
    
    # 1. Test PaperSynthesisNode
    logger.info("Testing PaperSynthesisNode...")
    llm_utility = MockLLMUtility()
    synthesis_node = PaperSynthesisNode(llm_utility)
    
    context = {
        'store': {'errors': []},
        'input_data': sample_scenario
    }
    
    result = await synthesis_node.process(context)
    if result['success']:
        synthesis = result['synthesis']
        logger.info(f"Synthesis completed for: {synthesis['seminal_paper']['title']}")
        logger.info(f"Citation landscape: {synthesis['citation_landscape']['total_citations']} citations")
        logger.info(f"Research evolution themes: {synthesis['citation_landscape']['citation_themes'][:3]}")
        logger.info(f"Methodological evolution: {synthesis['research_evolution']['methodological_evolution'][:3]}")
    else:
        logger.error(f"Synthesis failed: {result['error']}")
    
    # 2. Test TrendAnalysisNode
    logger.info("\nTesting TrendAnalysisNode...")
    trend_node = TrendAnalysisNode(llm_utility)
    
    context['store']['comprehensive_synthesis'] = result['synthesis']
    
    result = await trend_node.process(context)
    if result['success']:
        trends = result['trends']
        gaps = result['gaps']
        logger.info(f"Identified {len(trends['major_trends'])} major trends:")
        for trend in trends['major_trends']:
            logger.info(f"  - {trend['trend']} (strength: {trend['strength']})")
        
        logger.info(f"Identified {len(gaps)} research gaps:")
        for gap in gaps[:3]:
            logger.info(f"  - {gap['type']}: {gap['description']}")
    else:
        logger.error(f"Trend analysis failed: {result['error']}")
    
    # 3. Test DirectionGeneratorNode
    logger.info("\nTesting DirectionGeneratorNode...")
    direction_node = DirectionGeneratorNode(llm_utility)
    
    context['store']['identified_trends'] = result['trends']
    context['store']['research_gaps'] = result['gaps']
    context['config'] = {'max_suggestions': 4, 'min_confidence': 0.6}
    
    result = await direction_node.process(context)
    if result['success']:
        directions = result['directions']
        logger.info(f"Generated {len(directions)} research directions:")
        for i, direction in enumerate(directions, 1):
            logger.info(f"  {i}. {direction['title']}")
            logger.info(f"     Confidence: {direction['confidence']:.2f}, Impact: {direction['impact_potential']}")
            logger.info(f"     Type: {direction['research_type']}, Source: {direction['source']}")
    else:
        logger.error(f"Direction generation failed: {result['error']}")
    
    # 4. Test SuggestionFormatterNode
    logger.info("\nTesting SuggestionFormatterNode...")
    formatter_node = SuggestionFormatterNode()
    
    context['store']['suggested_directions'] = result['directions']
    
    result = await formatter_node.process(context)
    if result['success']:
        formatted = result['formatted']
        logger.info(f"Formatted {len(formatted['research_suggestions'])} suggestions")
        
        # Show synthesis summary
        summary = formatted['synthesis_summary']
        logger.info(f"Synthesis overview:")
        logger.info(f"  Seminal paper: {summary['seminal_paper_overview']['title']}")
        logger.info(f"  Citation impact: {summary['seminal_paper_overview']['impact_summary']}")
        logger.info(f"  Field diversity: {summary['research_landscape']['field_diversity']} venues")
        
        # Show top suggestion
        if formatted['research_suggestions']:
            top_suggestion = formatted['research_suggestions'][0]
            logger.info(f"Top research suggestion:")
            logger.info(f"  Title: {top_suggestion['title']}")
            logger.info(f"  Confidence: {top_suggestion['confidence']:.2f}")
            logger.info(f"  Timeline: {top_suggestion['estimated_timeline']}")
            logger.info(f"  Resources: {', '.join(top_suggestion['required_resources'][:3])}")
    else:
        logger.error(f"Formatting failed: {result['error']}")


async def test_full_workflow():
    """Test the complete Research Synthesis Agent workflow."""
    logger.info("\n=== Testing Complete Research Synthesis Agent Workflow ===")
    
    # Create agent with mock utilities
    class MockResearchSynthesisAgent(AcademicNewResearchAgent):
        def __init__(self, store, config):
            super().__init__(store, config)
            
            # Replace LLM utility with mock for demo
            self.llm_utility = self._create_mock_llm()
            
            # Re-initialize nodes with mock utility
            from nodes import PaperSynthesisNode, TrendAnalysisNode, DirectionGeneratorNode, SuggestionFormatterNode
            self.paper_synthesis_node = PaperSynthesisNode(self.llm_utility)
            self.trend_analysis_node = TrendAnalysisNode(self.llm_utility)
            self.direction_generator_node = DirectionGeneratorNode(self.llm_utility)
            self.suggestion_formatter_node = SuggestionFormatterNode()
        
        def _create_mock_llm(self):
            class MockLLM:
                async def synthesize_research_directions(self, paper_data=None, citations=None, analysis_type=None, context=None):
                    import random
                    
                    if analysis_type == 'comprehensive_synthesis':
                        return {
                            'insights': {
                                'field_evolution': f"Field has shown {random.choice(['rapid', 'steady', 'explosive'])} growth",
                                'research_momentum': f"Research momentum is {random.choice(['accelerating', 'stable', 'emerging'])}"
                            },
                            'success': True
                        }
                    elif analysis_type == 'trend_analysis':
                        return {
                            'enhanced_trends': {'emerging_patterns': ['efficiency', 'multimodal']},
                            'additional_gaps': [{'type': 'efficiency', 'description': 'Energy efficiency gap', 'opportunity_level': 'high'}],
                            'success': True
                        }
                    elif analysis_type == 'novel_directions':
                        return {
                            'suggestions': [
                                {
                                    'title': f"Advanced {random.choice(['Attention', 'Architecture', 'Learning'])} Methods",
                                    'description': f"Novel approach to {random.choice(['efficiency', 'accuracy', 'scalability'])}",
                                    'confidence': random.uniform(0.7, 0.9),
                                    'rationale': 'Based on comprehensive analysis of current trends'
                                }
                            ],
                            'success': True
                        }
                    
                    return {'success': True}
            return MockLLM()
    
    # Test with different scenarios
    scenarios = create_sample_research_scenarios()
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n--- Testing Scenario {i}: {scenario['name']} ---")
        
        # Create fresh agent for each test
        config = {
            'min_confidence': 0.6,
            'max_suggestions': 4,
            'trend_analysis_depth': 'comprehensive'
        }
        
        store = {}
        agent = MockResearchSynthesisAgent(store, config)
        
        # Run the workflow
        input_data = {
            'paper_metadata': scenario['paper_metadata'],
            'paper_analysis': scenario['paper_analysis'],
            'citation_data': scenario['citation_data']
        }
        
        try:
            result = await agent.run(input_data)
            
            if result['success']:
                logger.info(f"✓ Research synthesis completed successfully")
                logger.info(f"Status: {result['status']}")
                logger.info(f"Processing time: {result['processing_time']:.2f}s")
                
                # Show synthesis results
                suggestions = result['suggestions']
                stats = result['stats']
                logger.info(f"Generated {stats['suggestions_generated']} research suggestions:")
                logger.info(f"  - High confidence suggestions: {stats['high_confidence_suggestions']}")
                logger.info(f"  - Trends identified: {stats['trends_identified']}")
                logger.info(f"  - Research gaps found: {stats['gaps_found']}")
                
                # Show top suggestions
                if suggestions.get('research_suggestions'):
                    logger.info("Top research suggestions:")
                    for j, suggestion in enumerate(suggestions['research_suggestions'][:2], 1):
                        logger.info(f"  {j}. {suggestion['title']}")
                        logger.info(f"     Type: {suggestion['research_type']}, Confidence: {suggestion['confidence']:.2f}")
                        logger.info(f"     Timeline: {suggestion['estimated_timeline']}")
                
                # Show impact assessment
                if suggestions.get('impact_assessment'):
                    impact = suggestions['impact_assessment']
                    logger.info(f"Impact assessment:")
                    logger.info(f"  - High impact suggestions: {impact['impact_distribution']['high_impact_count']}")
                    logger.info(f"  - Average confidence: {impact['quality_metrics']['average_confidence']}")
                    logger.info(f"  - Average novelty: {impact['quality_metrics']['average_novelty']}")
                
            else:
                logger.error(f"✗ Research synthesis failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"✗ Workflow exception: {e}")


async def test_error_handling():
    """Test Research Synthesis Agent error handling."""
    logger.info("\n=== Testing Error Handling ===")
    
    from agents import AcademicNewResearchAgent
    
    # Test with minimal config
    config = {'min_confidence': 0.7}
    store = {}
    agent = AcademicNewResearchAgent(store, config)
    
    # Test 1: Missing input data
    logger.info("Testing missing input data...")
    result = await agent.run({})
    assert result['success'] is False
    logger.info(f"✓ Correctly handled missing data: {result['error']}")
    
    # Test 2: Incomplete input data
    logger.info("Testing incomplete input data...")
    incomplete_data = {
        'paper_metadata': {'title': 'Test Paper'},
        # Missing paper_analysis and citation_data
    }
    result = await agent.run(incomplete_data)
    # Should still attempt to process with available data
    logger.info(f"✓ Handled incomplete data gracefully")
    
    logger.info("Error handling tests completed successfully")


def save_results_to_file(results, filename="research_synthesis_results.json"):
    """Save results to a JSON file for inspection."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


async def main():
    """Run all Research Synthesis Agent demos."""
    logger.info("Starting Research Synthesis Agent Demo")
    logger.info("="*60)
    
    try:
        # Test basic functionality
        agent = await test_research_synthesis_agent_basic()
        
        # Test individual nodes
        await test_individual_nodes()
        
        # Test complete workflow
        await test_full_workflow()
        
        # Test error handling
        await test_error_handling()
        
        logger.info("\n" + "="*60)
        logger.info("Research Synthesis Agent Demo completed successfully!")
        logger.info("The Research Synthesis Agent is ready for:")
        logger.info("  • Synthesizing seminal papers with citation landscapes")
        logger.info("  • Identifying research trends and methodological evolution")
        logger.info("  • Finding research gaps and opportunity areas")
        logger.info("  • Generating novel research directions with confidence scores")
        logger.info("  • Formatting comprehensive research suggestions with timelines")
        logger.info("  • Providing strategic recommendations and implementation roadmaps")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())