"""Tests for the state management system."""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from utils import (
    SharedStore,
    AgentCommunicator,
    StoreError,
    StoreValidationError,
    StoreAccessError
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def shared_store(temp_dir):
    """Create a SharedStore instance for testing."""
    return SharedStore(persistence_dir=temp_dir)


@pytest.fixture
def communicator(shared_store):
    """Create an AgentCommunicator for testing."""
    return AgentCommunicator(shared_store, 'test_agent')


class TestSharedStore:
    """Test cases for SharedStore class."""
    
    def test_initialization(self, shared_store):
        """Test store initialization."""
        # Check all stores are created
        assert 'paper_store' in shared_store._stores
        assert 'citation_store' in shared_store._stores
        assert 'research_store' in shared_store._stores
        assert 'session_store' in shared_store._stores
        assert 'workflow_store' in shared_store._stores
        
        # Check locks are created
        assert len(shared_store._locks) == 5
        
        # Check metadata
        metadata = shared_store.get_metadata()
        assert 'created_at' in metadata
        assert metadata['access_count'] == 0
    
    def test_get_store(self, shared_store):
        """Test getting a store."""
        paper_store = shared_store.get_store('paper_store')
        
        # Check it returns a copy
        assert paper_store is not shared_store._stores['paper_store']
        
        # Check structure
        assert 'title' in paper_store
        assert 'authors' in paper_store
        assert 'year' in paper_store
        
        # Check metadata is updated
        metadata = shared_store.get_metadata()
        assert metadata['access_count'] == 1
    
    def test_get_invalid_store(self, shared_store):
        """Test getting an invalid store."""
        with pytest.raises(StoreAccessError):
            shared_store.get_store('invalid_store')
    
    def test_update_store(self, shared_store):
        """Test updating a store."""
        updates = {
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2'],
            'year': 2023
        }
        
        shared_store.update_store('paper_store', updates)
        
        # Check updates applied
        paper_store = shared_store.get_store('paper_store')
        assert paper_store['title'] == 'Test Paper'
        assert paper_store['authors'] == ['Author 1', 'Author 2']
        assert paper_store['year'] == 2023
        
        # Check metadata updated
        metadata = shared_store.get_metadata()
        assert metadata['modification_count'] == 1
    
    def test_deep_update(self, shared_store):
        """Test deep update functionality."""
        # Initial nested update
        shared_store.update_store('paper_store', {
            'metadata': {
                'doi': '10.1234/test',
                'venue': 'Test Conference'
            }
        })
        
        # Update nested field
        shared_store.update_store('paper_store', {
            'metadata': {
                'pages': '1-10'
            }
        })
        
        # Check both fields exist
        paper_store = shared_store.get_store('paper_store')
        assert paper_store['metadata']['doi'] == '10.1234/test'
        assert paper_store['metadata']['venue'] == 'Test Conference'
        assert paper_store['metadata']['pages'] == '1-10'
    
    def test_validation_paper_store(self, shared_store):
        """Test paper store validation."""
        # Invalid year
        with pytest.raises(StoreValidationError):
            shared_store.update_store('paper_store', {'year': 1800})
        
        with pytest.raises(StoreValidationError):
            shared_store.update_store('paper_store', {'year': 2200})
        
        # Invalid authors type
        with pytest.raises(StoreValidationError):
            shared_store.update_store('paper_store', {'authors': 'Not a list'})
        
        # Invalid PDF path type
        with pytest.raises(StoreValidationError):
            shared_store.update_store('paper_store', {'pdf_path': 123})
    
    def test_validation_citation_store(self, shared_store):
        """Test citation store validation."""
        # Invalid citing papers type
        with pytest.raises(StoreValidationError):
            shared_store.update_store('citation_store', {'citing_papers': 'Not a list'})
        
        # Negative total found
        with pytest.raises(StoreValidationError):
            shared_store.update_store('citation_store', {'total_found': -1})
        
        # Invalid relevance threshold
        with pytest.raises(StoreValidationError):
            shared_store.update_store('citation_store', {
                'search_metadata': {'relevance_threshold': 1.5}
            })
    
    def test_validation_research_store(self, shared_store):
        """Test research store validation."""
        # Invalid suggestions type
        with pytest.raises(StoreValidationError):
            shared_store.update_store('research_store', {'suggestions': 'Not a list'})
        
        # Invalid confidence score
        with pytest.raises(StoreValidationError):
            shared_store.update_store('research_store', {
                'confidence_scores': {'test': 1.5}
            })
    
    def test_validation_session_store(self, shared_store):
        """Test session store validation."""
        # Invalid output format
        with pytest.raises(StoreValidationError):
            shared_store.update_store('session_store', {
                'preferences': {'output_format': 'invalid'}
            })
        
        # Invalid max citations
        with pytest.raises(StoreValidationError):
            shared_store.update_store('session_store', {
                'preferences': {'max_citations': -1}
            })
    
    def test_validation_workflow_store(self, shared_store):
        """Test workflow store validation."""
        # Invalid stage
        with pytest.raises(StoreValidationError):
            shared_store.update_store('workflow_store', {
                'current_stage': 'invalid_stage'
            })
        
        # Invalid agent status
        with pytest.raises(StoreValidationError):
            shared_store.update_store('workflow_store', {
                'agent_status': {'coordinator': 'invalid_status'}
            })
    
    def test_clear_store(self, shared_store):
        """Test clearing a store."""
        # Add some data
        shared_store.update_store('paper_store', {
            'title': 'Test Paper',
            'year': 2023
        })
        
        # Clear the store
        shared_store.clear_store('paper_store')
        
        # Check it's reset
        paper_store = shared_store.get_store('paper_store')
        assert paper_store['title'] == ''
        assert paper_store['year'] is None
    
    def test_save_and_load_session(self, shared_store, temp_dir):
        """Test saving and loading sessions."""
        # Add some data
        shared_store.update_store('paper_store', {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'year': 2023
        })
        
        shared_store.update_store('citation_store', {
            'total_found': 100,
            'citing_papers': [{'title': 'Citation 1'}]
        })
        
        # Save session
        file_path = shared_store.save_session()
        assert os.path.exists(file_path)
        
        # Create new store and load
        new_store = SharedStore()
        new_store.load_session(file_path)
        
        # Check data loaded correctly
        paper_store = new_store.get_store('paper_store')
        assert paper_store['title'] == 'Test Paper'
        assert paper_store['year'] == 2023
        
        citation_store = new_store.get_store('citation_store')
        assert citation_store['total_found'] == 100
        assert len(citation_store['citing_papers']) == 1
    
    def test_export_as_json(self, shared_store, temp_dir):
        """Test exporting as JSON."""
        # Add some data
        shared_store.update_store('paper_store', {
            'title': 'Test Paper',
            'year': 2023
        })
        
        # Export
        file_path = shared_store.export_as_json()
        assert os.path.exists(file_path)
        
        # Load and check
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert 'stores' in data
        assert 'metadata' in data
        assert data['stores']['paper_store']['title'] == 'Test Paper'
    
    def test_thread_safety(self, shared_store):
        """Test thread-safe operations."""
        results = []
        errors = []
        
        def update_store(store_name, value, index):
            try:
                for i in range(10):
                    shared_store.update_store(store_name, {
                        f'field_{index}': f'{value}_{i}'
                    })
                    time.sleep(0.001)  # Small delay to increase contention
                results.append(f'Thread {index} completed')
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=update_store,
                args=('session_store', f'thread_{i}', i)
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors
        assert len(errors) == 0
        assert len(results) == 5
        
        # Check all updates applied
        session_store = shared_store.get_store('session_store')
        for i in range(5):
            assert f'field_{i}' in session_store


class TestAgentCommunicator:
    """Test cases for AgentCommunicator class."""
    
    def test_initialization(self, communicator):
        """Test communicator initialization."""
        assert communicator.agent_name == 'test_agent'
        assert communicator.shared_store is not None
    
    def test_update_agent_status(self, communicator, shared_store):
        """Test updating agent status."""
        communicator.update_agent_status('active')
        
        workflow_store = shared_store.get_store('workflow_store')
        assert workflow_store['agent_status']['test_agent'] == 'active'
    
    def test_request_citation_search(self, communicator, shared_store):
        """Test requesting citation search."""
        paper_metadata = {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'year': 2023
        }
        
        communicator.request_citation_search(paper_metadata)
        
        # Check message added
        workflow_store = shared_store.get_store('workflow_store')
        messages = workflow_store['inter_agent_messages']
        assert len(messages) == 1
        assert messages[0]['type'] == 'citation_search_request'
        assert messages[0]['from'] == 'test_agent'
        assert messages[0]['to'] == 'web_search'
        
        # Check citation store updated
        citation_store = shared_store.get_store('citation_store')
        assert citation_store['search_status'] == 'requested'
    
    def test_check_citation_results(self, communicator, shared_store):
        """Test checking citation results."""
        # No results initially
        assert communicator.check_citation_results() is None
        
        # Add results
        shared_store.update_store('citation_store', {
            'search_status': 'completed',
            'citing_papers': [{'title': 'Citation 1'}],
            'total_found': 1
        })
        
        results = communicator.check_citation_results()
        assert results is not None
        assert len(results['citing_papers']) == 1
        assert results['total_found'] == 1
    
    def test_request_research_synthesis(self, communicator, shared_store):
        """Test requesting research synthesis."""
        synthesis_data = {
            'paper_metadata': {'title': 'Test Paper'},
            'citation_data': [{'title': 'Citation 1'}]
        }
        
        communicator.request_research_synthesis(synthesis_data)
        
        # Check message added
        workflow_store = shared_store.get_store('workflow_store')
        messages = workflow_store['inter_agent_messages']
        assert len(messages) == 1
        assert messages[0]['type'] == 'synthesis_request'
        
        # Check research store updated
        research_store = shared_store.get_store('research_store')
        assert research_store['synthesis_status'] == 'requested'
    
    def test_check_synthesis_results(self, communicator, shared_store):
        """Test checking synthesis results."""
        # No results initially
        assert communicator.check_synthesis_results() is None
        
        # Add results
        shared_store.update_store('research_store', {
            'synthesis_status': 'completed',
            'suggestions': [{'title': 'Research Direction 1'}]
        })
        
        results = communicator.check_synthesis_results()
        assert results is not None
        assert len(results['suggestions']) == 1
    
    def test_update_workflow_stage(self, communicator, shared_store):
        """Test updating workflow stage."""
        # Initial stage
        communicator.update_workflow_stage('pdf_processing')
        
        workflow_store = shared_store.get_store('workflow_store')
        assert workflow_store['current_stage'] == 'pdf_processing'
        
        # Next stage
        communicator.update_workflow_stage('analysis')
        
        workflow_store = shared_store.get_store('workflow_store')
        assert workflow_store['current_stage'] == 'analysis'
        assert 'pdf_processing' in workflow_store['completed_stages']
    
    def test_add_status_update(self, communicator, shared_store):
        """Test adding status updates."""
        communicator.add_status_update('Processing started')
        
        session_store = shared_store.get_store('session_store')
        updates = session_store['status_updates']
        assert len(updates) == 1
        assert updates[0]['message'] == 'Processing started'
        assert updates[0]['agent'] == 'test_agent'
        assert updates[0]['level'] == 'info'
    
    def test_log_error(self, communicator, shared_store):
        """Test error logging."""
        communicator.log_error('Test error', 'test_error')
        
        session_store = shared_store.get_store('session_store')
        errors = session_store['error_log']
        assert len(errors) == 1
        assert errors[0]['error'] == 'Test error'
        assert errors[0]['type'] == 'test_error'
    
    def test_update_performance_metric(self, communicator, shared_store):
        """Test updating performance metrics."""
        communicator.update_performance_metric('processing_time', 10.5)
        
        session_store = shared_store.get_store('session_store')
        metrics = session_store['performance_metrics']
        assert metrics['processing_time'] == 10.5
    
    def test_get_pending_messages(self, communicator, shared_store):
        """Test getting pending messages."""
        # Add messages for different agents
        messages = [
            {
                'to': 'test_agent',
                'from': 'coordinator',
                'status': 'pending',
                'timestamp': '2023-01-01T00:00:00'
            },
            {
                'to': 'test_agent',
                'from': 'coordinator',
                'status': 'processed',
                'timestamp': '2023-01-01T00:01:00'
            },
            {
                'to': 'other_agent',
                'from': 'coordinator',
                'status': 'pending',
                'timestamp': '2023-01-01T00:02:00'
            }
        ]
        
        shared_store.update_store('workflow_store', {
            'inter_agent_messages': messages
        })
        
        # Get pending messages for test_agent
        pending = communicator.get_pending_messages()
        assert len(pending) == 1
        assert pending[0]['to'] == 'test_agent'
        assert pending[0]['status'] == 'pending'
    
    def test_mark_message_processed(self, communicator, shared_store):
        """Test marking messages as processed."""
        # Add a message
        message = {
            'to': 'test_agent',
            'from': 'coordinator',
            'status': 'pending',
            'timestamp': '2023-01-01T00:00:00'
        }
        
        shared_store.update_store('workflow_store', {
            'inter_agent_messages': [message]
        })
        
        # Mark as processed
        communicator.mark_message_processed(message)
        
        # Check status updated
        workflow_store = shared_store.get_store('workflow_store')
        assert workflow_store['inter_agent_messages'][0]['status'] == 'processed'
        assert 'processed_at' in workflow_store['inter_agent_messages'][0]


class TestIntegration:
    """Integration tests for state management system."""
    
    def test_multi_agent_workflow(self, temp_dir):
        """Test a multi-agent workflow scenario."""
        # Create shared store
        store = SharedStore(persistence_dir=temp_dir)
        
        # Create communicators for three agents
        coordinator_comm = AgentCommunicator(store, 'coordinator')
        web_search_comm = AgentCommunicator(store, 'web_search')
        synthesis_comm = AgentCommunicator(store, 'research_synthesis')
        
        # Simulate workflow
        # 1. Coordinator starts
        coordinator_comm.update_agent_status('active')
        coordinator_comm.update_workflow_stage('pdf_processing')
        coordinator_comm.add_status_update('Processing PDF')
        
        # Add paper data
        store.update_store('paper_store', {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'year': 2023,
            'key_concepts': ['concept1', 'concept2']
        })
        
        # 2. Request citation search
        coordinator_comm.request_citation_search({
            'title': 'Test Paper',
            'authors': ['Author 1']
        })
        
        # Web search agent picks up request
        pending = web_search_comm.get_pending_messages()
        assert len(pending) == 1  # Should have citation search request message
        assert pending[0]['type'] == 'citation_search_request'
        
        citation_store = store.get_store('citation_store')
        assert citation_store['search_status'] == 'requested'
        
        # Web search completes
        web_search_comm.update_agent_status('active')
        store.update_store('citation_store', {
            'search_status': 'completed',
            'citing_papers': [
                {'title': 'Citation 1', 'year': 2024},
                {'title': 'Citation 2', 'year': 2023}
            ],
            'total_found': 2
        })
        web_search_comm.update_agent_status('completed')
        
        # 3. Request synthesis
        coordinator_comm.request_research_synthesis({
            'paper_data': store.get_store('paper_store'),
            'citation_data': store.get_store('citation_store')['citing_papers']
        })
        
        # Synthesis completes
        synthesis_comm.update_agent_status('active')
        store.update_store('research_store', {
            'synthesis_status': 'completed',
            'suggestions': [
                {
                    'title': 'Research Direction 1',
                    'confidence': 0.85
                }
            ]
        })
        synthesis_comm.update_agent_status('completed')
        
        # 4. Complete workflow
        coordinator_comm.update_workflow_stage('completed')
        coordinator_comm.update_agent_status('completed')
        
        # Verify final state
        workflow_store = store.get_store('workflow_store')
        assert workflow_store['current_stage'] == 'completed'
        assert all(status == 'completed' for status in workflow_store['agent_status'].values())
        
        # Check we can export results
        export_path = store.export_as_json()
        assert os.path.exists(export_path)
        
        with open(export_path, 'r') as f:
            exported = json.load(f)
        
        assert exported['stores']['paper_store']['title'] == 'Test Paper'
        assert len(exported['stores']['citation_store']['citing_papers']) == 2
        assert len(exported['stores']['research_store']['suggestions']) == 1