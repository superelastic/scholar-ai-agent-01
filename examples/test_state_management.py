"""Demo script for the State Management and Coordination System.

This script demonstrates:
1. SharedStore for thread-safe state management
2. AgentCommunicator for inter-agent communication
3. ScholarAICoordinator for orchestrating the complete workflow
4. Session persistence and recovery

Usage:
    python examples/test_state_management.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coordinator import ScholarAICoordinator
from utils import SharedStore, AgentCommunicator, create_sample_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_shared_store_basics():
    """Test basic SharedStore functionality."""
    logger.info("=== Testing SharedStore Basics ===")
    
    # Create a shared store
    store = SharedStore()
    
    # Test store access
    logger.info("Testing store access...")
    paper_store = store.get_store('paper_store')
    logger.info(f"Paper store initialized with {len(paper_store)} fields")
    
    # Test store updates
    logger.info("\nTesting store updates...")
    store.update_store('paper_store', {
        'title': 'Attention Is All You Need',
        'authors': ['Vaswani, A.', 'Shazeer, N.'],
        'year': 2017,
        'key_concepts': ['transformer', 'attention mechanism']
    })
    
    # Retrieve updated data
    updated_store = store.get_store('paper_store')
    logger.info(f"Updated paper title: {updated_store['title']}")
    logger.info(f"Authors: {', '.join(updated_store['authors'])}")
    
    # Test validation
    logger.info("\nTesting validation...")
    try:
        store.update_store('paper_store', {'year': 3000})  # Invalid year
    except Exception as e:
        logger.info(f"✓ Validation correctly caught invalid year: {e}")
    
    # Test metadata
    metadata = store.get_metadata()
    logger.info(f"\nStore metadata:")
    logger.info(f"  Access count: {metadata['access_count']}")
    logger.info(f"  Modification count: {metadata['modification_count']}")
    
    return store


def test_thread_safety():
    """Test thread-safe operations on SharedStore."""
    logger.info("\n=== Testing Thread Safety ===")
    
    store = SharedStore()
    results = []
    
    def worker(worker_id, iterations=100):
        """Worker function for thread testing."""
        for i in range(iterations):
            try:
                # Read
                data = store.get_store('session_store')
                
                # Update
                store.update_store('session_store', {
                    f'worker_{worker_id}_update': f'iteration_{i}',
                    'last_activity': datetime.now().isoformat()
                })
                
                if i % 20 == 0:
                    results.append(f"Worker {worker_id}: Completed {i} iterations")
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    # Create multiple threads
    threads = []
    num_workers = 5
    
    logger.info(f"Starting {num_workers} concurrent workers...")
    start_time = time.time()
    
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    logger.info(f"All workers completed in {elapsed:.2f} seconds")
    
    # Check final state
    final_store = store.get_store('session_store')
    worker_updates = [k for k in final_store.keys() if k.startswith('worker_')]
    logger.info(f"Final store contains {len(worker_updates)} worker updates")
    
    metadata = store.get_metadata()
    logger.info(f"Total access count: {metadata['access_count']}")
    logger.info(f"Total modification count: {metadata['modification_count']}")


def test_agent_communication():
    """Test AgentCommunicator functionality."""
    logger.info("\n=== Testing Agent Communication ===")
    
    # Create shared store and communicators
    store = SharedStore()
    
    coord_comm = AgentCommunicator(store, 'coordinator')
    search_comm = AgentCommunicator(store, 'web_search')
    synth_comm = AgentCommunicator(store, 'research_synthesis')
    
    # Simulate workflow communication
    logger.info("Simulating agent workflow communication...")
    
    # 1. Coordinator updates status
    coord_comm.update_agent_status('active')
    coord_comm.update_workflow_stage('pdf_processing')
    coord_comm.add_status_update('Starting PDF processing')
    
    # 2. Coordinator requests citation search
    paper_metadata = {
        'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
        'authors': ['Devlin, J.', 'Chang, M.'],
        'year': 2018
    }
    
    coord_comm.request_citation_search(paper_metadata)
    logger.info("Coordinator requested citation search")
    
    # 3. Web search agent processes request
    citation_store = store.get_store('citation_store')
    logger.info(f"Citation search status: {citation_store['search_status']}")
    
    # Simulate search completion
    search_comm.update_agent_status('active')
    store.update_store('citation_store', {
        'search_status': 'completed',
        'citing_papers': [
            {'title': 'RoBERTa: A Robustly Optimized BERT', 'year': 2019},
            {'title': 'ALBERT: A Lite BERT', 'year': 2019},
            {'title': 'DistilBERT: A distilled version of BERT', 'year': 2019}
        ],
        'total_found': 3
    })
    search_comm.update_agent_status('completed')
    
    # 4. Coordinator checks results
    results = coord_comm.check_citation_results()
    if results:
        logger.info(f"Citation results available: {results['total_found']} papers found")
        for paper in results['citing_papers']:
            logger.info(f"  - {paper['title']}")
    
    # 5. Request synthesis
    synthesis_data = {
        'paper_metadata': paper_metadata,
        'citation_data': results['citing_papers']
    }
    coord_comm.request_research_synthesis(synthesis_data)
    logger.info("\nCoordinator requested research synthesis")
    
    # 6. Show workflow status
    workflow_store = store.get_store('workflow_store')
    logger.info(f"\nWorkflow status:")
    logger.info(f"  Current stage: {workflow_store['current_stage']}")
    logger.info(f"  Agent statuses: {workflow_store['agent_status']}")
    logger.info(f"  Inter-agent messages: {len(workflow_store['inter_agent_messages'])}")
    
    # 7. Show session updates
    session_store = store.get_store('session_store')
    logger.info(f"\nStatus updates:")
    for update in session_store['status_updates'][-5:]:  # Last 5
        logger.info(f"  [{update['timestamp'][:19]}] {update['agent']}: {update['message']}")
    
    return store


def test_session_persistence():
    """Test session save and load functionality."""
    logger.info("\n=== Testing Session Persistence ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and populate store
        store = SharedStore(persistence_dir=temp_dir)
        
        # Add comprehensive data
        store.update_store('paper_store', {
            'title': 'GPT-3: Language Models are Few-Shot Learners',
            'authors': ['Brown, T.', 'Mann, B.'],
            'year': 2020,
            'key_concepts': ['language model', 'few-shot learning', 'GPT-3']
        })
        
        store.update_store('citation_store', {
            'citing_papers': [
                {'title': 'GPT-4 Technical Report', 'year': 2023}
            ],
            'total_found': 1
        })
        
        store.update_store('research_store', {
            'suggestions': [
                {
                    'title': 'Efficient Few-Shot Learning Methods',
                    'confidence': 0.85
                }
            ]
        })
        
        # Save session
        logger.info("Saving session...")
        save_path = store.save_session()
        logger.info(f"Session saved to: {save_path}")
        
        # Export as JSON for inspection
        json_path = store.export_as_json()
        logger.info(f"Exported to JSON: {json_path}")
        
        # Create new store and load session
        logger.info("\nLoading session in new store...")
        new_store = SharedStore()
        new_store.load_session(save_path)
        
        # Verify data loaded correctly
        paper_store = new_store.get_store('paper_store')
        logger.info(f"Loaded paper title: {paper_store['title']}")
        
        citation_store = new_store.get_store('citation_store')
        logger.info(f"Loaded citations: {citation_store['total_found']} papers")
        
        research_store = new_store.get_store('research_store')
        logger.info(f"Loaded suggestions: {len(research_store['suggestions'])} directions")
        
        # Show JSON export sample
        logger.info("\nJSON export sample:")
        with open(json_path, 'r') as f:
            export_data = json.load(f)
        
        logger.info(json.dumps(export_data['metadata'], indent=2))


async def test_coordinator_workflow():
    """Test the complete coordinator workflow."""
    logger.info("\n=== Testing Complete Coordinator Workflow ===")
    
    # Create temporary directory for session
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration
        config = {
            'coordinator': {
                'cache_dir': temp_dir,
                'pdf_timeout': 5
            },
            'web_search': {
                'max_results': 5,
                'year_filter': 3
            },
            'research_synthesis': {
                'min_confidence': 0.6,
                'max_suggestions': 3
            }
        }
        
        # Create coordinator
        coordinator = ScholarAICoordinator(
            config=config,
            persistence_dir=temp_dir
        )
        
        # Create sample PDF
        pdf_path = create_sample_pdf()
        
        try:
            logger.info(f"Processing sample PDF: {pdf_path}")
            
            # Mock the agent responses for demo
            from unittest.mock import AsyncMock, patch
            
            with patch.object(coordinator.coordinator_agent, 'run', new_callable=AsyncMock) as mock_coord:
                with patch.object(coordinator.web_search_agent, 'run', new_callable=AsyncMock) as mock_search:
                    with patch.object(coordinator.research_synthesis_agent, 'run', new_callable=AsyncMock) as mock_synth:
                        
                        # Configure mocks
                        mock_coord.return_value = {
                            'success': True,
                            'results': {
                                'paper_metadata': {
                                    'title': 'Sample Research Paper',
                                    'authors': ['Demo Author'],
                                    'year': 2023
                                },
                                'paper_analysis': {
                                    'key_concepts': ['machine learning', 'AI'],
                                    'methodology': 'Experimental study',
                                    'findings': ['Finding 1', 'Finding 2']
                                }
                            },
                            'processing_time': 0.5
                        }
                        
                        mock_search.return_value = {
                            'success': True,
                            'citations': {
                                'citations': [
                                    {'title': 'Citing Paper 1', 'year': 2024},
                                    {'title': 'Citing Paper 2', 'year': 2023}
                                ],
                                'summary': {'total_count': 2},
                                'metadata': {'queries_used': ['sample query']}
                            },
                            'processing_time': 1.0
                        }
                        
                        mock_synth.return_value = {
                            'success': True,
                            'suggestions': {
                                'research_suggestions': [
                                    {
                                        'title': 'Future Direction 1',
                                        'confidence': 0.85,
                                        'description': 'Explore advanced applications'
                                    }
                                ],
                                'synthesis_summary': {'trends': {}, 'gaps': {}}
                            },
                            'processing_time': 1.5
                        }
                        
                        # Process paper with preferences
                        preferences = {
                            'output_format': 'json',
                            'max_citations': 10,
                            'min_relevance': 0.6
                        }
                        
                        result = await coordinator.process_paper(pdf_path, preferences)
                        
                        if result['success']:
                            logger.info("✓ Workflow completed successfully!")
                            
                            # Show performance metrics
                            logger.info(f"\nPerformance metrics:")
                            for metric, value in result['performance'].items():
                                if value:
                                    logger.info(f"  {metric}: {value:.2f}s")
                            
                            # Show results summary
                            results = result['results']
                            logger.info(f"\nResults summary:")
                            logger.info(f"  Paper: {results['paper_analysis']['metadata']['title']}")
                            logger.info(f"  Citations found: {results['citations']['summary']['total_found']}")
                            logger.info(f"  Research directions: {len(results['research_directions']['suggestions'])}")
                            
                            # Get workflow status
                            status = coordinator.get_status()
                            logger.info(f"\nFinal workflow status:")
                            logger.info(f"  Stage: {status['current_stage']}")
                            logger.info(f"  Completed stages: {', '.join(status['completed_stages'])}")
                            
                            # Save session
                            session_path = coordinator.save_session()
                            logger.info(f"\nSession saved to: {session_path}")
                            
                            # Export results
                            export_path = coordinator.export_results('json')
                            logger.info(f"Results exported to: {export_path}")
                            
                        else:
                            logger.error(f"✗ Workflow failed: {result['error']}")
                        
        finally:
            # Cleanup
            try:
                Path(pdf_path).unlink()
            except:
                pass


def main():
    """Run all state management demos."""
    logger.info("Starting State Management and Coordination Demo")
    logger.info("="*60)
    
    try:
        # Test SharedStore basics
        store = test_shared_store_basics()
        
        # Test thread safety
        test_thread_safety()
        
        # Test agent communication
        comm_store = test_agent_communication()
        
        # Test session persistence
        test_session_persistence()
        
        # Test complete coordinator workflow
        asyncio.run(test_coordinator_workflow())
        
        logger.info("\n" + "="*60)
        logger.info("State Management Demo completed successfully!")
        logger.info("\nThe state management system provides:")
        logger.info("  • Thread-safe shared storage across agents")
        logger.info("  • Structured stores for different data types")
        logger.info("  • Built-in validation and error handling")
        logger.info("  • Inter-agent communication mechanisms")
        logger.info("  • Session persistence and recovery")
        logger.info("  • Complete workflow orchestration")
        logger.info("  • Performance tracking and metrics")
        logger.info("  • Export capabilities for results")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()