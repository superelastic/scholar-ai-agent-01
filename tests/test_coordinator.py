"""Tests for the Scholar AI Coordinator."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from coordinator import ScholarAICoordinator
from utils import create_sample_pdf


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def coordinator_config():
    """Configuration for coordinator."""
    return {
        'coordinator': {
            'cache_dir': './test_cache',
            'pdf_timeout': 5
        },
        'web_search': {
            'max_results': 10,
            'year_filter': 2
        },
        'research_synthesis': {
            'min_confidence': 0.6,
            'max_suggestions': 3
        }
    }


@pytest.fixture
def coordinator(coordinator_config, temp_dir):
    """Create a ScholarAICoordinator for testing."""
    return ScholarAICoordinator(
        config=coordinator_config,
        persistence_dir=temp_dir
    )


@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing."""
    pdf_path = create_sample_pdf()
    yield pdf_path
    # Cleanup
    try:
        Path(pdf_path).unlink()
    except:
        pass


class TestScholarAICoordinator:
    """Test cases for ScholarAICoordinator."""
    
    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.shared_store is not None
        assert coordinator.coordinator_agent is not None
        assert coordinator.web_search_agent is not None
        assert coordinator.research_synthesis_agent is not None
        
        # Check communicators
        assert coordinator.coordinator_comm is not None
        assert coordinator.web_search_comm is not None
        assert coordinator.research_synthesis_comm is not None
    
    @pytest.mark.asyncio
    async def test_process_paper_success(self, coordinator, sample_pdf):
        """Test successful paper processing workflow."""
        # Mock agent responses
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            with patch.object(coordinator.web_search_agent, 'run', new_callable=AsyncMock) as mock_search:
                with patch.object(coordinator.research_synthesis_agent, 'run', new_callable=AsyncMock) as mock_synth:
                    
                    # Configure mocks
                    mock_coord.return_value = {
                        'success': True,
                        'results': {
                            'paper_metadata': {
                                'title': 'Test Paper',
                                'authors': ['Author 1'],
                                'year': 2023
                            },
                            'paper_analysis': {
                                'key_concepts': ['concept1'],
                                'methodology': 'test method',
                                'findings': ['finding1']
                            },
                            'paper_content': 'Test content',
                            'paper_sections': {},
                            'paper_references': []
                        },
                        'processing_time': 1.0
                    }
                    
                    mock_search.return_value = {
                        'success': True,
                        'citations': {
                            'citations': [{'title': 'Citation 1'}],
                            'summary': {'total_count': 1},
                            'metadata': {
                                'queries_used': ['query1'],
                                'filter_settings': {}
                            }
                        },
                        'processing_time': 2.0
                    }
                    
                    mock_synth.return_value = {
                        'success': True,
                        'suggestions': {
                            'research_suggestions': [
                                {'title': 'Direction 1', 'confidence': 0.8}
                            ],
                            'synthesis_summary': {},
                            'impact_assessment': {},
                            'implementation_roadmap': {}
                        },
                        'processing_time': 3.0
                    }
                    
                    # Process paper
                    result = await coordinator.process_paper(sample_pdf)
                    
                    # Verify success
                    assert result['success'] is True
                    assert 'results' in result
                    assert 'performance' in result
                    assert 'session_id' in result
                    
                    # Check workflow stages
                    workflow_store = coordinator.shared_store.get_store('workflow_store')
                    assert workflow_store['current_stage'] == 'completed'
                    assert len(workflow_store['completed_stages']) > 0
                    
                    # Check performance metrics
                    assert result['performance']['pdf_processing_time'] == 1.0
                    assert result['performance']['citation_search_time'] == 2.0
                    assert result['performance']['synthesis_time'] == 3.0
    
    @pytest.mark.asyncio
    async def test_process_paper_coordinator_failure(self, coordinator, sample_pdf):
        """Test workflow when coordinator agent fails."""
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            mock_coord.return_value = {
                'success': False,
                'error': 'PDF processing failed'
            }
            
            result = await coordinator.process_paper(sample_pdf)
            
            assert result['success'] is False
            assert 'error' in result
            assert 'Coordinator agent failed' in result['error']
            
            # Check workflow marked as error
            workflow_store = coordinator.shared_store.get_store('workflow_store')
            assert workflow_store['current_stage'] == 'error'
    
    @pytest.mark.asyncio
    async def test_process_paper_citation_failure_graceful(self, coordinator, sample_pdf):
        """Test graceful degradation when citation search fails."""
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            with patch.object(coordinator.web_search_agent, 'run', new_callable=AsyncMock) as mock_search:
                with patch.object(coordinator.research_synthesis_agent, 'run', new_callable=AsyncMock) as mock_synth:
                    
                    # Coordinator succeeds
                    mock_coord.return_value = {
                        'success': True,
                        'results': {
                            'paper_metadata': {'title': 'Test Paper'},
                            'paper_analysis': {'key_concepts': ['concept1']}
                        }
                    }
                    
                    # Citation search fails
                    mock_search.return_value = {
                        'success': False,
                        'error': 'Network error'
                    }
                    
                    # Synthesis succeeds despite no citations
                    mock_synth.return_value = {
                        'success': True,
                        'suggestions': {
                            'research_suggestions': [],
                            'synthesis_summary': {}
                        }
                    }
                    
                    result = await coordinator.process_paper(sample_pdf)
                    
                    # Should still succeed
                    assert result['success'] is True
                    
                    # Check status updates include warning
                    session_store = coordinator.shared_store.get_store('session_store')
                    warnings = [u for u in session_store['status_updates'] if u['level'] == 'warning']
                    assert len(warnings) > 0
    
    @pytest.mark.asyncio
    async def test_process_paper_with_preferences(self, coordinator, sample_pdf):
        """Test processing with user preferences."""
        preferences = {
            'output_format': 'markdown',
            'max_citations': 50,
            'min_relevance': 0.7
        }
        
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            mock_coord.return_value = {'success': True, 'results': {}}
            
            with patch.object(coordinator.web_search_agent, 'run', new_callable=AsyncMock) as mock_search:
                mock_search.return_value = {'success': True, 'citations': {}}
                
                with patch.object(coordinator.research_synthesis_agent, 'run', new_callable=AsyncMock) as mock_synth:
                    mock_synth.return_value = {'success': True, 'suggestions': {}}
                    
                    await coordinator.process_paper(sample_pdf, preferences)
                    
                    # Check preferences stored
                    session_store = coordinator.shared_store.get_store('session_store')
                    assert session_store['preferences']['output_format'] == 'markdown'
                    assert session_store['preferences']['max_citations'] == 50
                    assert session_store['preferences']['min_relevance'] == 0.7
    
    def test_compile_final_results(self, coordinator):
        """Test compiling final results from stores."""
        # Populate stores
        coordinator.shared_store.update_store('paper_store', {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'year': 2023,
            'key_concepts': ['concept1'],
            'methodology': 'test method',
            'findings': ['finding1']
        })
        
        coordinator.shared_store.update_store('citation_store', {
            'total_found': 10,
            'filtered_count': 5,
            'citing_papers': [{'title': 'Citation 1'}]
        })
        
        coordinator.shared_store.update_store('research_store', {
            'suggestions': [{'title': 'Direction 1'}],
            'synthesis_data': {'trends': [], 'gaps': []},
            'implementation_roadmap': {}
        })
        
        # Compile results
        results = coordinator._compile_final_results()
        
        # Verify structure
        assert 'paper_analysis' in results
        assert results['paper_analysis']['metadata']['title'] == 'Test Paper'
        
        assert 'citations' in results
        assert results['citations']['summary']['total_found'] == 10
        
        assert 'research_directions' in results
        assert len(results['research_directions']['suggestions']) == 1
        
        assert 'session_info' in results
    
    def test_compile_partial_results(self, coordinator):
        """Test compiling partial results on failure."""
        # Add minimal data
        coordinator.shared_store.update_store('paper_store', {
            'title': 'Test Paper'
        })
        
        # Should not raise exception
        partial = coordinator._compile_partial_results()
        assert partial is not None
        
        # Should have raw store data as fallback
        if 'paper_store' in partial:
            assert partial['paper_store']['title'] == 'Test Paper'
    
    def test_save_and_load_session(self, coordinator, temp_dir):
        """Test saving and loading coordinator session."""
        # Add some data
        coordinator.shared_store.update_store('paper_store', {
            'title': 'Test Paper',
            'year': 2023
        })
        
        # Save session
        save_path = coordinator.save_session()
        assert os.path.exists(save_path)
        
        # Create new coordinator and load
        new_coordinator = ScholarAICoordinator(persistence_dir=temp_dir)
        new_coordinator.load_session(save_path)
        
        # Check data loaded
        paper_store = new_coordinator.shared_store.get_store('paper_store')
        assert paper_store['title'] == 'Test Paper'
        assert paper_store['year'] == 2023
    
    def test_export_results_json(self, coordinator, temp_dir):
        """Test exporting results as JSON."""
        # Add some data
        coordinator.shared_store.update_store('paper_store', {
            'title': 'Test Paper'
        })
        
        # Export
        export_path = coordinator.export_results('json')
        assert os.path.exists(export_path)
        assert export_path.endswith('.json')
    
    def test_export_results_invalid_format(self, coordinator):
        """Test exporting with invalid format."""
        with pytest.raises(ValueError):
            coordinator.export_results('invalid_format')
    
    def test_get_status(self, coordinator):
        """Test getting coordinator status."""
        # Update some status
        coordinator.coordinator_comm.update_workflow_stage('pdf_processing')
        coordinator.coordinator_comm.update_agent_status('active')
        
        status = coordinator.get_status()
        
        assert status['current_stage'] == 'pdf_processing'
        assert status['agent_status']['coordinator'] == 'active'
        assert 'performance_metrics' in status
        assert 'errors' in status
    
    @pytest.mark.asyncio
    async def test_agent_communication_flow(self, coordinator):
        """Test inter-agent communication through shared store."""
        # Simulate coordinator requesting citation search
        paper_metadata = {
            'title': 'Test Paper',
            'authors': ['Author 1']
        }
        
        coordinator.coordinator_comm.request_citation_search(paper_metadata)
        
        # Check citation store updated
        citation_store = coordinator.shared_store.get_store('citation_store')
        assert citation_store['search_status'] == 'requested'
        
        # Simulate web search agent completing
        coordinator.shared_store.update_store('citation_store', {
            'search_status': 'completed',
            'citing_papers': [{'title': 'Citation 1'}],
            'total_found': 1
        })
        
        # Coordinator should be able to check results
        results = coordinator.coordinator_comm.check_citation_results()
        assert results is not None
        assert results['total_found'] == 1
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, coordinator, sample_pdf):
        """Test performance metric tracking."""
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            with patch.object(coordinator.web_search_agent, 'run', new_callable=AsyncMock) as mock_search:
                with patch.object(coordinator.research_synthesis_agent, 'run', new_callable=AsyncMock) as mock_synth:
                    
                    # Configure mocks with timing
                    mock_coord.return_value = {
                        'success': True,
                        'results': {},
                        'processing_time': 1.5
                    }
                    mock_search.return_value = {
                        'success': True,
                        'citations': {},
                        'processing_time': 2.5
                    }
                    mock_synth.return_value = {
                        'success': True,
                        'suggestions': {},
                        'processing_time': 3.5
                    }
                    
                    result = await coordinator.process_paper(sample_pdf)
                    
                    # Check all metrics recorded
                    metrics = result['performance']
                    assert metrics['pdf_processing_time'] == 1.5
                    assert metrics['citation_search_time'] == 2.5
                    assert metrics['synthesis_time'] == 3.5
                    assert metrics['total_time'] > 0
    
    @pytest.mark.asyncio
    async def test_error_logging(self, coordinator, sample_pdf):
        """Test error logging during workflow."""
        with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
            mock_coord.side_effect = Exception("Test exception")
            
            result = await coordinator.process_paper(sample_pdf)
            
            assert result['success'] is False
            
            # Check error logged
            session_store = coordinator.shared_store.get_store('session_store')
            error_log = session_store['error_log']
            assert len(error_log) > 0
            assert 'Test exception' in error_log[0]['error']