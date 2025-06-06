"""Tests for the Academic Coordinator Agent and its nodes."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents import AcademicCoordinatorAgent
from nodes import (
    UserInputNode,
    PaperAnalysisNode,
    CitationSearchNode,
    ResearchSynthesisNode,
    PresentationNode,
    CoordinatorNodeError
)
from utils import (
    PDFExtractorUtility,
    LLMAnalysisUtility,
    ScholarSearchUtility,
    create_sample_pdf
)


@pytest.fixture
def coordinator_config():
    """Configuration for coordinator agent."""
    return {
        'cache_dir': './test_cache',
        'pdf_timeout': 5,
        'openai_api_key': 'test_openai_key',
        'llm_model': 'gpt-4',
        'llm_timeout': 30,
        'scholar_cache_ttl': 1,
        'scholar_interval': 0.1,
        'scholar_retries': 2
    }


@pytest.fixture
def coordinator_agent(coordinator_config):
    """Create coordinator agent for testing."""
    store = {}
    return AcademicCoordinatorAgent(store, coordinator_config)


@pytest.fixture
def sample_pdf_path():
    """Create a sample PDF for testing."""
    pdf_path = create_sample_pdf()
    yield pdf_path
    # Cleanup
    try:
        Path(pdf_path).unlink()
    except:
        pass


@pytest.fixture
def mock_utilities():
    """Create mock utilities for isolated testing."""
    pdf_extractor = Mock(spec=PDFExtractorUtility)
    llm_utility = Mock(spec=LLMAnalysisUtility)
    scholar_utility = Mock(spec=ScholarSearchUtility)
    
    # Configure PDF extractor mocks
    pdf_extractor.validate_pdf.return_value = (True, None)
    pdf_extractor.extract_text.return_value = {
        'text': 'Sample paper text about machine learning',
        'success': True
    }
    pdf_extractor.extract_metadata.return_value = {
        'metadata': {
            'title': 'Sample Paper',
            'authors': ['Test Author'],
            'year': 2023
        },
        'success': True
    }
    pdf_extractor.extract_sections.return_value = {
        'sections': {'abstract': 'Sample abstract'},
        'success': True
    }
    pdf_extractor.extract_references.return_value = {
        'references': ['Sample reference'],
        'success': True
    }
    
    return pdf_extractor, llm_utility, scholar_utility


def test_coordinator_agent_initialization(coordinator_agent):
    """Test coordinator agent initialization."""
    assert coordinator_agent.name == "AcademicCoordinatorAgent"
    assert coordinator_agent.config is not None
    assert coordinator_agent.pdf_extractor is not None
    assert coordinator_agent.llm_utility is not None
    assert coordinator_agent.scholar_utility is not None
    assert coordinator_agent.user_input_node is not None
    assert coordinator_agent.paper_analysis_node is not None
    assert coordinator_agent.citation_search_node is not None
    assert coordinator_agent.research_synthesis_node is not None
    assert coordinator_agent.presentation_node is not None


def test_coordinator_store_initialization(coordinator_agent):
    """Test store initialization."""
    store = coordinator_agent.initialize_store()
    
    required_keys = [
        'paper_metadata', 'paper_content', 'analysis_results',
        'citation_results', 'research_suggestions', 'status',
        'progress', 'errors', 'started_at'
    ]
    
    for key in required_keys:
        assert key in store
    
    assert store['status'] == 'initialized'
    assert store['progress'] == 0


def test_coordinator_status_tracking(coordinator_agent):
    """Test status tracking functionality."""
    # Initially not started
    status = coordinator_agent.get_status()
    assert status['status'] == 'not_started'
    assert status['progress'] == 0
    
    # After initializing store
    coordinator_agent.store = coordinator_agent.initialize_store()
    status = coordinator_agent.get_status()
    assert status['status'] == 'initialized'
    assert status['progress'] == 0


# Node Tests

@pytest.mark.asyncio
async def test_user_input_node_success(mock_utilities, sample_pdf_path):
    """Test successful user input processing."""
    pdf_extractor, _, _ = mock_utilities
    node = UserInputNode(pdf_extractor)
    
    context = {
        'store': {},
        'input_data': {'pdf_path': sample_pdf_path}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert context['store']['pdf_path'] == sample_pdf_path
    assert context['store']['status'] == 'pdf_validated'
    assert context['store']['progress'] == 10


@pytest.mark.asyncio
async def test_user_input_node_missing_pdf():
    """Test user input with missing PDF."""
    pdf_extractor = Mock(spec=PDFExtractorUtility)
    node = UserInputNode(pdf_extractor)
    
    context = {
        'store': {},
        'input_data': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is False
    assert 'No PDF path provided' in result['error']


@pytest.mark.asyncio
async def test_user_input_node_invalid_pdf():
    """Test user input with invalid PDF."""
    pdf_extractor = Mock(spec=PDFExtractorUtility)
    pdf_extractor.validate_pdf.return_value = (False, "Invalid PDF format")
    
    node = UserInputNode(pdf_extractor)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b'invalid pdf content')
        f.flush()
        
        context = {
            'store': {},
            'input_data': {'pdf_path': f.name}
        }
        
        result = await node.process(context)
        
        assert result['success'] is False
        assert 'Invalid PDF' in result['error']
        
        Path(f.name).unlink()


@pytest.mark.asyncio
async def test_paper_analysis_node_success(mock_utilities):
    """Test successful paper analysis."""
    pdf_extractor, llm_utility, _ = mock_utilities
    
    # Mock LLM response
    llm_utility.analyze_paper = AsyncMock(return_value={
        'key_concepts': ['machine learning', 'AI'],
        'methodology': 'Experimental study',
        'findings': ['Result 1', 'Result 2'],
        'success': True
    })
    
    node = PaperAnalysisNode(pdf_extractor, llm_utility)
    
    context = {
        'store': {'pdf_path': '/test/paper.pdf'},
        'input_data': {}
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'analysis_results' in context['store']
    assert context['store']['status'] == 'analysis_complete'
    assert context['store']['progress'] == 60


@pytest.mark.asyncio
async def test_citation_search_node_success(mock_utilities):
    """Test successful citation search."""
    _, llm_utility, scholar_utility = mock_utilities
    
    # Mock LLM query generation
    llm_utility.generate_search_queries = AsyncMock(return_value={
        'queries': ['query1', 'query2'],
        'success': True
    })
    
    # Mock Scholar search
    scholar_utility.search_citations.return_value = {
        'papers': [
            {'title': 'Citing Paper 1', 'year': 2024, 'relevance_score': 0.8},
            {'title': 'Citing Paper 2', 'year': 2023, 'relevance_score': 0.6}
        ],
        'success': True
    }
    
    scholar_utility.filter_results.return_value = [
        {'title': 'Citing Paper 1', 'year': 2024, 'relevance_score': 0.8}
    ]
    
    scholar_utility.format_citations.return_value = {
        'citations': [{'title': 'Citing Paper 1'}],
        'summary': {'total_count': 1}
    }
    
    node = CitationSearchNode(scholar_utility, llm_utility)
    
    context = {
        'store': {
            'paper_metadata': {'title': 'Test Paper', 'authors': ['Author'], 'year': 2020},
            'analysis_results': {'key_concepts': ['ML']}
        }
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'citation_results' in context['store']
    assert 'filtered_citations' in context['store']
    assert context['store']['status'] == 'citations_found'


@pytest.mark.asyncio
async def test_research_synthesis_node_success(mock_utilities):
    """Test successful research synthesis."""
    _, llm_utility, _ = mock_utilities
    
    # Mock LLM synthesis
    llm_utility.synthesize_research_directions = AsyncMock(return_value={
        'suggestions': [
            {
                'title': 'Future Direction 1',
                'description': 'Description 1',
                'confidence': 0.85
            }
        ],
        'success': True
    })
    
    node = ResearchSynthesisNode(llm_utility)
    
    context = {
        'store': {
            'analysis_results': {'key_concepts': ['ML']},
            'filtered_citations': [{'title': 'Citation 1'}]
        }
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'research_suggestions' in context['store']
    assert context['store']['status'] == 'synthesis_complete'


@pytest.mark.asyncio
async def test_presentation_node_success(mock_utilities):
    """Test successful presentation formatting with new formatter system."""
    
    # Create PresentationNode with default formats (no need for llm_utility anymore)
    node = PresentationNode(['json', 'markdown'])
    
    context = {
        'store': {
            'paper_metadata': {'title': 'Test Paper', 'authors': ['Test Author']},
            'analysis_results': {'key_concepts': ['ML', 'AI'], 'methodology': 'Test method'},
            'filtered_citations': [{'title': 'Citation 1', 'year': 2020}],
            'research_suggestions': {'suggestions': [{'title': 'Direction 1', 'confidence': 0.8}]}
        }
    }
    
    result = await node.process(context)
    
    assert result['success'] is True
    assert 'results' in result
    assert 'presentations' in result
    assert 'export_files' in result
    assert 'progress_summary' in result
    assert context['store']['status'] == 'completed'
    assert context['store']['progress'] == 100
    assert 'final_results' in context['store']
    assert 'formatted_presentations' in context['store']


# Integration Tests

@pytest.mark.asyncio
async def test_coordinator_full_workflow_success(coordinator_agent, sample_pdf_path):
    """Test complete coordinator workflow."""
    # Mock all LLM calls
    with patch.object(coordinator_agent.llm_utility, 'analyze_paper', new_callable=AsyncMock) as mock_analyze:
        with patch.object(coordinator_agent.llm_utility, 'generate_search_queries', new_callable=AsyncMock) as mock_queries:
            with patch.object(coordinator_agent.llm_utility, 'synthesize_research_directions', new_callable=AsyncMock) as mock_synthesis:
                with patch.object(coordinator_agent.llm_utility, 'format_presentation', new_callable=AsyncMock) as mock_format:
                    
                    # Configure mocks
                    mock_analyze.return_value = {
                        'key_concepts': ['machine learning'],
                        'methodology': 'experimental',
                        'success': True
                    }
                    
                    mock_queries.return_value = {
                        'queries': ['test query'],
                        'success': True
                    }
                    
                    # Mock Scholar search to avoid network calls
                    coordinator_agent.scholar_utility.search_citations = Mock(return_value={
                        'papers': [],
                        'success': True
                    })
                    coordinator_agent.scholar_utility.filter_results = Mock(return_value=[])
                    coordinator_agent.scholar_utility.format_citations = Mock(return_value={
                        'citations': [],
                        'summary': {'total_count': 0}
                    })
                    
                    mock_synthesis.return_value = {
                        'suggestions': [],
                        'success': True
                    }
                    
                    mock_format.return_value = {
                        'presentation': {'status': 'complete'},
                        'success': True
                    }
                    
                    # Run workflow
                    input_data = {'pdf_path': sample_pdf_path}
                    result = await coordinator_agent.run(input_data)
                    
                    # Verify success
                    assert result['success'] is True
                    assert result['status'] == 'completed'
                    assert result['progress'] == 100
                    assert 'results' in result
                    assert 'presentation' in result


@pytest.mark.asyncio
async def test_coordinator_workflow_pdf_error(coordinator_agent):
    """Test coordinator workflow with PDF error."""
    input_data = {'pdf_path': '/nonexistent/file.pdf'}
    result = await coordinator_agent.run(input_data)
    
    assert result['success'] is False
    assert 'error' in result
    assert result['progress'] >= 0


@pytest.mark.asyncio
async def test_coordinator_workflow_graceful_degradation(coordinator_agent, sample_pdf_path):
    """Test coordinator workflow with citation search failure."""
    # Mock LLM analysis to succeed
    with patch.object(coordinator_agent.llm_utility, 'analyze_paper', new_callable=AsyncMock) as mock_analyze:
        with patch.object(coordinator_agent.llm_utility, 'generate_search_queries', new_callable=AsyncMock) as mock_queries:
            with patch.object(coordinator_agent.llm_utility, 'synthesize_research_directions', new_callable=AsyncMock) as mock_synthesis:
                with patch.object(coordinator_agent.llm_utility, 'format_presentation', new_callable=AsyncMock) as mock_format:
                    
                    mock_analyze.return_value = {'key_concepts': ['ML'], 'success': True}
                    mock_queries.return_value = {'queries': ['test'], 'success': True}
                    
                    # Make Scholar search fail
                    coordinator_agent.scholar_utility.search_citations = Mock(return_value={
                        'papers': [],
                        'success': False,
                        'error': 'Network error'
                    })
                    
                    mock_synthesis.return_value = {'suggestions': [], 'success': True}
                    mock_format.return_value = {'presentation': {}, 'success': True}
                    
                    input_data = {'pdf_path': sample_pdf_path}
                    result = await coordinator_agent.run(input_data)
                    
                    # Should still succeed despite citation search failure
                    assert result['success'] is True
                    # Citation search error should be recorded somewhere
                    assert (coordinator_agent.store.get('citation_search_error') or 
                           coordinator_agent.store.get('citation_results', {}).get('error'))


def test_coordinator_create_flow(coordinator_agent):
    """Test flow creation."""
    flow = coordinator_agent.create_flow()
    assert flow is not None
    # Note: Actual flow structure testing would depend on PocketFlow implementation