"""Tests for the Academic New Research Agent and its nodes."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents import AcademicNewResearchAgent
from nodes import (
    PaperSynthesisNode,
    TrendAnalysisNode,
    DirectionGeneratorNode,
    SuggestionFormatterNode,
    ResearchSynthesisNodeError
)
from utils import LLMAnalysisUtility


@pytest.fixture
def research_synthesis_config():
    """Configuration for research synthesis agent."""
    return {
        'min_confidence': 0.6,
        'max_suggestions': 4,
        'trend_analysis_depth': 'comprehensive',
        'openai_api_key': 'test_openai_key',
        'llm_model': 'gpt-4',
        'llm_timeout': 30
    }


@pytest.fixture
def research_synthesis_agent(research_synthesis_config):
    """Create research synthesis agent for testing."""
    store = {}
    return AcademicNewResearchAgent(store, research_synthesis_config)


@pytest.fixture
def sample_input_data():
    """Sample input data for research synthesis."""
    return {
        'paper_metadata': {
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani, A.', 'Shazeer, N.'],
            'year': 2017,
            'key_concepts': ['transformer', 'attention mechanism', 'neural machine translation']
        },
        'paper_analysis': {
            'key_concepts': ['transformer architecture', 'self-attention', 'multi-head attention'],
            'methodology': 'Deep learning approach using attention mechanisms',
            'findings': ['Transformers outperform RNNs', 'Parallelizable architecture', 'Better long-range dependencies'],
            'significance': 'Revolutionary architecture for sequence modeling'
        },
        'citation_data': [
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'authors': ['Devlin, J.', 'Chang, M.'],
                'year': 2018,
                'venue': 'NAACL',
                'relevance_score': 0.95,
                'citation_count': 15000,
                'abstract': 'BERT obtains new state-of-the-art results on eleven natural language processing tasks...'
            },
            {
                'title': 'Language Models are Few-Shot Learners',
                'authors': ['Brown, T.', 'Mann, B.'],
                'year': 2020,
                'venue': 'NeurIPS',
                'relevance_score': 0.88,
                'citation_count': 8000,
                'abstract': 'We demonstrate that scaling up language models greatly improves task-agnostic...'
            },
            {
                'title': 'An Image is Worth 16x16 Words: Transformers for Image Recognition',
                'authors': ['Dosovitskiy, A.'],
                'year': 2020,
                'venue': 'ICLR',
                'relevance_score': 0.82,
                'citation_count': 12000,
                'abstract': 'Vision Transformer (ViT) attains excellent results compared to state-of-the-art...'
            }
        ]
    }


@pytest.fixture
def mock_llm_utility():
    """Create mock LLM utility for testing."""
    mock_llm = Mock(spec=LLMAnalysisUtility)
    
    # Mock synthesize_research_directions method
    mock_llm.synthesize_research_directions = AsyncMock(return_value={
        'suggestions': [
            {
                'title': 'Hierarchical Attention Mechanisms',
                'description': 'Develop multi-level attention for complex reasoning',
                'confidence': 0.85,
                'rationale': 'Current attention is flat, hierarchical could improve reasoning'
            },
            {
                'title': 'Efficient Transformer Architectures',
                'description': 'Reduce computational complexity while maintaining performance',
                'confidence': 0.78,
                'rationale': 'Transformers are computationally expensive, efficiency is crucial'
            }
        ],
        'success': True,
        'insights': {
            'field_evolution': 'Transformers have become foundational',
            'impact_analysis': 'Massive impact across multiple domains'
        }
    })
    
    return mock_llm


def test_research_synthesis_agent_initialization(research_synthesis_agent):
    """Test research synthesis agent initialization."""
    assert research_synthesis_agent.name == "AcademicNewResearchAgent"
    assert research_synthesis_agent.min_confidence == 0.6
    assert research_synthesis_agent.max_suggestions == 4
    assert research_synthesis_agent.trend_analysis_depth == "comprehensive"
    assert research_synthesis_agent.llm_utility is not None
    assert research_synthesis_agent.paper_synthesis_node is not None
    assert research_synthesis_agent.trend_analysis_node is not None
    assert research_synthesis_agent.direction_generator_node is not None
    assert research_synthesis_agent.suggestion_formatter_node is not None


def test_research_synthesis_store_initialization(research_synthesis_agent):
    """Test store initialization."""
    store = research_synthesis_agent.initialize_store()
    
    required_keys = [
        'paper_metadata', 'paper_analysis', 'citation_data',
        'identified_trends', 'research_gaps', 'suggested_directions',
        'formatted_suggestions', 'status', 'errors', 'synthesis_stats'
    ]
    
    for key in required_keys:
        assert key in store
    
    assert store['status'] == 'initialized'
    assert store['synthesis_stats']['trends_identified'] == 0
    assert store['synthesis_stats']['suggestions_generated'] == 0


def test_research_synthesis_status_tracking(research_synthesis_agent):
    """Test status tracking functionality."""
    # Initially not started
    status = research_synthesis_agent.get_status()
    assert status['status'] == 'not_started'
    
    # After initializing store
    research_synthesis_agent.store = research_synthesis_agent.initialize_store()
    status = research_synthesis_agent.get_status()
    assert status['status'] == 'initialized'


# Node Tests

@pytest.mark.asyncio
async def test_paper_synthesis_node_success(mock_llm_utility, sample_input_data):
    """Test successful paper synthesis."""
    node = PaperSynthesisNode(mock_llm_utility)
    
    context = {
        'store': {'errors': []},
        'input_data': sample_input_data
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'synthesis' in result
    
    # Check store updates
    assert context['store']['paper_metadata'] == sample_input_data['paper_metadata']
    assert context['store']['paper_analysis'] == sample_input_data['paper_analysis']
    assert context['store']['citation_data'] == sample_input_data['citation_data']
    assert context['store']['status'] == 'synthesis_complete'
    assert 'comprehensive_synthesis' in context['store']
    
    # Check synthesis structure
    synthesis = result['synthesis']
    assert 'seminal_paper' in synthesis
    assert 'citation_landscape' in synthesis
    assert 'research_evolution' in synthesis
    assert 'synthesis_metadata' in synthesis


@pytest.mark.asyncio
async def test_paper_synthesis_node_no_data():
    """Test paper synthesis node with no input data."""
    mock_llm = Mock(spec=LLMAnalysisUtility)
    node = PaperSynthesisNode(mock_llm)
    
    context = {
        'store': {'errors': []},
        'input_data': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is False
    assert 'No paper metadata or analysis provided' in result['error']


@pytest.mark.asyncio
async def test_trend_analysis_node_success(mock_llm_utility):
    """Test successful trend analysis."""
    node = TrendAnalysisNode(mock_llm_utility)
    
    # Create synthesis data
    synthesis_data = {
        'seminal_paper': {
            'title': 'Test Paper',
            'key_concepts': ['transformer', 'attention']
        },
        'citation_landscape': {
            'total_citations': 3,
            'citation_years': {'citation_growth': 'accelerating'},
            'citing_venues': [{'venue': 'ICML', 'count': 2}, {'venue': 'NeurIPS', 'count': 1}]
        },
        'research_evolution': {
            'methodological_evolution': ['deep learning', 'transformer', 'attention mechanism'],
            'application_domains': ['natural language processing', 'computer vision'],
            'temporal_trends': {
                'research_acceleration': True,
                'recent_period_count': 2,
                'early_period_count': 1
            }
        }
    }
    
    context = {
        'store': {
            'comprehensive_synthesis': synthesis_data,
            'errors': [],
            'synthesis_stats': {}
        },
        'config': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'trends' in result
    assert 'gaps' in result
    
    # Check trends structure
    trends = result['trends']
    assert 'major_trends' in trends
    assert 'methodological_trends' in trends
    assert 'application_trends' in trends
    
    # Check that some trends were identified
    assert len(trends['major_trends']) > 0
    assert context['store']['status'] == 'trend_analysis_complete'


@pytest.mark.asyncio
async def test_trend_analysis_node_no_synthesis():
    """Test trend analysis node with no synthesis data."""
    mock_llm = Mock(spec=LLMAnalysisUtility)
    node = TrendAnalysisNode(mock_llm)
    
    context = {
        'store': {'errors': []},
        'config': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is False
    assert 'No synthesis data available' in result['error']


@pytest.mark.asyncio
async def test_direction_generator_node_success(mock_llm_utility):
    """Test successful direction generation."""
    node = DirectionGeneratorNode(mock_llm_utility)
    
    # Create sample trends and gaps
    trends = {
        'major_trends': [
            {
                'trend': 'Accelerating research interest',
                'evidence': 'Recent citations increasing',
                'strength': 'strong',
                'type': 'growth'
            }
        ],
        'methodological_trends': [
            {
                'trend': 'Increasing adoption of transformer',
                'strength': 'strong'
            }
        ]
    }
    
    gaps = [
        {
            'type': 'methodological',
            'description': 'Limited exploration of hierarchical attention',
            'research_direction': 'Develop multi-level attention mechanisms',
            'opportunity_level': 'high'
        },
        {
            'type': 'application',
            'description': 'Underexplored applications in healthcare',
            'research_direction': 'Apply transformers to medical tasks',
            'opportunity_level': 'medium'
        }
    ]
    
    synthesis_data = {
        'seminal_paper': {
            'title': 'Test Paper',
            'methodology': 'Transformer architecture'
        }
    }
    
    context = {
        'store': {
            'identified_trends': trends,
            'research_gaps': gaps,
            'comprehensive_synthesis': synthesis_data,
            'errors': [],
            'synthesis_stats': {}
        },
        'config': {'max_suggestions': 3, 'min_confidence': 0.6}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'directions' in result
    
    directions = result['directions']
    assert len(directions) > 0
    assert len(directions) <= 3  # Respects max_suggestions
    
    # Check direction structure
    for direction in directions:
        assert 'title' in direction
        assert 'description' in direction
        assert 'confidence' in direction
        assert 'rank' in direction
        
        # Check confidence threshold
        assert direction['confidence'] >= 0.6
    
    assert context['store']['status'] == 'directions_generated'


@pytest.mark.asyncio
async def test_direction_generator_node_no_input():
    """Test direction generator with no trends or gaps."""
    mock_llm = Mock(spec=LLMAnalysisUtility)
    node = DirectionGeneratorNode(mock_llm)
    
    context = {
        'store': {
            'errors': [],
            'synthesis_stats': {}
        },
        'config': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is False
    assert 'No trends, gaps, or synthesis data available' in result['error']


@pytest.mark.asyncio
async def test_suggestion_formatter_node_success():
    """Test successful suggestion formatting."""
    node = SuggestionFormatterNode()
    
    # Create sample data
    directions = [
        {
            'rank': 1,
            'title': 'Hierarchical Attention Mechanisms',
            'description': 'Develop multi-level attention for complex reasoning',
            'rationale': 'Current attention is flat',
            'approach': 'Systematic development with validation',
            'confidence': 0.85,
            'impact_potential': 'high',
            'feasibility': 'medium',
            'novelty_score': 0.8,
            'research_type': 'methodological',
            'source': 'gap_analysis'
        },
        {
            'rank': 2,
            'title': 'Healthcare Applications',
            'description': 'Apply transformers to medical diagnosis',
            'rationale': 'Healthcare is underexplored',
            'approach': 'Domain adaptation with medical experts',
            'confidence': 0.72,
            'impact_potential': 'medium',
            'feasibility': 'high',
            'novelty_score': 0.6,
            'research_type': 'application',
            'source': 'gap_analysis'
        }
    ]
    
    synthesis_data = {
        'seminal_paper': {'title': 'Test Paper'},
        'citation_landscape': {'total_citations': 3},
        'synthesis_metadata': {'data_quality': {'overall_score': 0.8}}
    }
    
    trends = {
        'major_trends': [{'trend': 'Growth', 'strength': 'strong'}]
    }
    
    gaps = [{'type': 'methodological', 'opportunity_level': 'high'}]
    
    context = {
        'store': {
            'suggested_directions': directions,
            'comprehensive_synthesis': synthesis_data,
            'identified_trends': trends,
            'research_gaps': gaps,
            'errors': []
        },
        'config': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'formatted' in result
    
    formatted = result['formatted']
    
    # Check main structure
    assert 'research_suggestions' in formatted
    assert 'synthesis_summary' in formatted
    assert 'methodology_overview' in formatted
    assert 'impact_assessment' in formatted
    assert 'implementation_roadmap' in formatted
    assert 'metadata' in formatted
    
    # Check research suggestions
    suggestions = formatted['research_suggestions']
    assert len(suggestions) == 2
    
    for suggestion in suggestions:
        assert 'rank' in suggestion
        assert 'title' in suggestion
        assert 'confidence' in suggestion
        assert 'estimated_timeline' in suggestion
        assert 'required_resources' in suggestion
        assert 'success_metrics' in suggestion
    
    assert context['store']['status'] == 'suggestions_formatted'


@pytest.mark.asyncio
async def test_suggestion_formatter_node_empty_data():
    """Test suggestion formatting with empty data."""
    node = SuggestionFormatterNode()
    
    context = {
        'store': {
            'suggested_directions': [],
            'comprehensive_synthesis': {},
            'identified_trends': {},
            'research_gaps': [],
            'errors': []
        },
        'config': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    formatted = result['formatted']
    assert formatted['research_suggestions'] == []


# Integration Tests

@pytest.mark.asyncio
async def test_research_synthesis_full_workflow(research_synthesis_agent, sample_input_data):
    """Test complete research synthesis workflow."""
    # Mock LLM utility for the agent
    with patch.object(research_synthesis_agent.llm_utility, 'synthesize_research_directions', new_callable=AsyncMock) as mock_llm:
        
        # Configure mock
        mock_llm.return_value = {
            'suggestions': [
                {
                    'title': 'Novel Attention Architecture',
                    'description': 'Advanced attention mechanisms',
                    'confidence': 0.8,
                    'rationale': 'Based on trend analysis'
                }
            ],
            'success': True,
            'insights': {'impact': 'High potential impact'},
            'enhanced_trends': {'additional_trends': []},
            'additional_gaps': []
        }
        
        # Run workflow
        result = await research_synthesis_agent.run(sample_input_data)
        
        # Verify success
        assert result['success'] is True
        assert result['status'] == 'suggestions_formatted'
        assert 'suggestions' in result
        assert 'stats' in result
        
        # Check suggestions structure
        suggestions = result['suggestions']
        assert 'research_suggestions' in suggestions
        assert 'synthesis_summary' in suggestions
        assert 'methodology_overview' in suggestions
        assert 'impact_assessment' in suggestions


@pytest.mark.asyncio
async def test_research_synthesis_no_input(research_synthesis_agent):
    """Test research synthesis with no input data."""
    result = await research_synthesis_agent.run({})
    
    assert result['success'] is False
    assert 'error' in result


@pytest.mark.asyncio
async def test_research_synthesis_trend_analysis_failure(research_synthesis_agent, sample_input_data):
    """Test research synthesis with trend analysis failure."""
    # Mock to make trend analysis fail but continue workflow
    with patch.object(research_synthesis_agent.trend_analysis_node, 'process', new_callable=AsyncMock) as mock_trend:
        with patch.object(research_synthesis_agent.llm_utility, 'synthesize_research_directions', new_callable=AsyncMock) as mock_llm:
            
            mock_trend.return_value = {
                'success': False,
                'error': 'Trend analysis failed'
            }
            
            mock_llm.return_value = {
                'suggestions': [],
                'success': True
            }
            
            result = await research_synthesis_agent.run(sample_input_data)
            
            # Should still succeed with graceful degradation
            assert result['success'] is True
            # Should record the trend analysis error
            assert research_synthesis_agent.store.get('trend_analysis_error')


def test_research_synthesis_create_flow(research_synthesis_agent):
    """Test flow creation."""
    flow = research_synthesis_agent.create_flow()
    assert flow is not None


# Helper function tests

def test_citation_year_extraction():
    """Test citation year extraction functionality."""
    from nodes.research_synthesis_nodes import PaperSynthesisNode
    from utils import LLMAnalysisUtility
    
    mock_llm = Mock(spec=LLMAnalysisUtility)
    node = PaperSynthesisNode(mock_llm)
    
    citation_data = [
        {'year': 2020},
        {'year': 2021},
        {'year': 2020},
        {'year': 2022}
    ]
    
    result = node._extract_citation_years(citation_data)
    
    assert result['min_year'] == 2020
    assert result['max_year'] == 2022
    assert result['year_distribution'][2020] == 2
    assert result['year_distribution'][2021] == 1
    assert result['year_distribution'][2022] == 1


def test_gap_confidence_calculation():
    """Test gap confidence calculation."""
    from nodes.research_synthesis_nodes import DirectionGeneratorNode
    from utils import LLMAnalysisUtility
    
    mock_llm = Mock(spec=LLMAnalysisUtility)
    node = DirectionGeneratorNode(mock_llm)
    
    # Test methodological gap (should have high confidence)
    methodological_gap = {
        'type': 'methodological',
        'opportunity_level': 'high'
    }
    
    confidence = node._calculate_gap_confidence(methodological_gap)
    assert confidence >= 0.8
    
    # Test theoretical gap (should have lower confidence)
    theoretical_gap = {
        'type': 'theoretical',
        'opportunity_level': 'low'
    }
    
    confidence = node._calculate_gap_confidence(theoretical_gap)
    assert confidence <= 0.6


def test_timeline_estimation():
    """Test timeline estimation for research directions."""
    from nodes.research_synthesis_nodes import SuggestionFormatterNode
    
    node = SuggestionFormatterNode()
    
    # Test methodological research
    methodological_direction = {
        'research_type': 'methodological',
        'feasibility': 'high'
    }
    
    timeline = node._estimate_timeline(methodological_direction)
    assert 'months' in timeline.lower()
    
    # Test theoretical research (should be longer)
    theoretical_direction = {
        'research_type': 'theoretical',
        'feasibility': 'medium'
    }
    
    timeline = node._estimate_timeline(theoretical_direction)
    assert any(num in timeline for num in ['18', '24'])  # Should be 18-24 months


def test_resource_estimation():
    """Test resource estimation for research directions."""
    from nodes.research_synthesis_nodes import SuggestionFormatterNode
    
    node = SuggestionFormatterNode()
    
    # Test methodological research
    methodological_direction = {
        'research_type': 'methodological',
        'approach': 'experimental validation with comparative analysis'
    }
    
    resources = node._estimate_resources(methodological_direction)
    
    assert 'Computational resources' in resources
    assert 'Development expertise' in resources
    assert 'Experimental setup' in resources
    assert 'Multiple baseline implementations' in resources