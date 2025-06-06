"""State management system for Scholar AI Agent.

This module provides a robust state management system for storing and sharing
data between agents, with proper synchronization and error handling.
"""

import copy
import json
import logging
import os
import pickle
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class StoreError(Exception):
    """Base exception for store operations."""
    pass


class StoreValidationError(StoreError):
    """Raised when store validation fails."""
    pass


class StoreAccessError(StoreError):
    """Raised when store access is denied."""
    pass


class SharedStore:
    """Thread-safe shared store for inter-agent communication.
    
    Provides separate stores for different data types with validation,
    synchronization, and persistence capabilities.
    """
    
    def __init__(self, persistence_dir: Optional[str] = None):
        """Initialize the shared store.
        
        Args:
            persistence_dir: Directory for persisting store data
        """
        self.persistence_dir = persistence_dir
        if persistence_dir:
            Path(persistence_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize individual stores
        self._stores = {
            'paper_store': self._create_paper_store(),
            'citation_store': self._create_citation_store(),
            'research_store': self._create_research_store(),
            'session_store': self._create_session_store(),
            'workflow_store': self._create_workflow_store()
        }
        
        # Create locks for thread-safe access
        self._locks = {store_name: threading.RLock() for store_name in self._stores}
        
        # Store metadata
        self._metadata = {
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'modification_count': 0
        }
        
        logger.info(f"Initialized SharedStore with {len(self._stores)} stores")
    
    def _create_paper_store(self) -> Dict[str, Any]:
        """Create initial paper store structure."""
        return {
            'title': '',
            'authors': [],
            'year': None,
            'abstract': '',
            'key_concepts': [],
            'methodology': '',
            'findings': [],
            'pdf_path': '',
            'pdf_content': '',
            'extracted_at': None,
            'extraction_method': None,
            'metadata': {
                'doi': None,
                'venue': None,
                'pages': None,
                'volume': None,
                'issue': None
            },
            'sections': {},
            'references': [],
            'citations_count': 0,
            'analysis_complete': False
        }
    
    def _create_citation_store(self) -> Dict[str, Any]:
        """Create initial citation store structure."""
        return {
            'citing_papers': [],
            'search_queries': [],
            'total_found': 0,
            'filtered_count': 0,
            'search_metadata': {
                'sources': [],
                'search_date': None,
                'filter_criteria': {},
                'relevance_threshold': 0.5
            },
            'citation_analysis': {
                'year_distribution': {},
                'venue_distribution': {},
                'author_network': {},
                'topic_evolution': []
            },
            'search_status': 'not_started',
            'last_updated': None
        }
    
    def _create_research_store(self) -> Dict[str, Any]:
        """Create initial research store structure."""
        return {
            'suggestions': [],
            'analysis_timestamp': None,
            'synthesis_data': {
                'trends': [],
                'gaps': [],
                'opportunities': []
            },
            'confidence_scores': {},
            'implementation_roadmap': [],
            'potential_collaborators': [],
            'resource_requirements': {},
            'synthesis_status': 'not_started',
            'synthesis_metadata': {
                'method': None,
                'confidence_level': None,
                'data_quality': None
            }
        }
    
    def _create_session_store(self) -> Dict[str, Any]:
        """Create initial session store structure."""
        return {
            'session_id': str(uuid.uuid4()),
            'user_id': None,
            'start_time': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'status_updates': [],
            'preferences': {
                'output_format': 'json',
                'max_citations': 20,
                'min_relevance': 0.5,
                'year_filter': None,
                'language': 'en'
            },
            'interaction_history': [],
            'performance_metrics': {
                'pdf_processing_time': None,
                'citation_search_time': None,
                'synthesis_time': None,
                'total_time': None
            },
            'error_log': []
        }
    
    def _create_workflow_store(self) -> Dict[str, Any]:
        """Create workflow coordination store."""
        return {
            'current_stage': 'initialized',
            'completed_stages': [],
            'pending_stages': [],
            'agent_status': {
                'coordinator': 'idle',
                'web_search': 'idle',
                'research_synthesis': 'idle'
            },
            'inter_agent_messages': [],
            'synchronization_points': {},
            'workflow_metadata': {
                'started_at': None,
                'completed_at': None,
                'total_duration': None
            }
        }
    
    def get_store(self, store_name: str) -> Dict[str, Any]:
        """Get a deep copy of a specific store.
        
        Args:
            store_name: Name of the store to retrieve
            
        Returns:
            Deep copy of the store data
            
        Raises:
            StoreAccessError: If store name is invalid
        """
        if store_name not in self._stores:
            raise StoreAccessError(f"Unknown store: {store_name}")
        
        with self._locks[store_name]:
            self._update_access_metadata()
            return copy.deepcopy(self._stores[store_name])
    
    def update_store(self, store_name: str, updates: Dict[str, Any], 
                    validate: bool = True, merge: bool = True) -> None:
        """Update a specific store with new data.
        
        Args:
            store_name: Name of the store to update
            updates: Dictionary of updates to apply
            validate: Whether to validate updates before applying
            merge: Whether to merge updates or replace entirely
            
        Raises:
            StoreAccessError: If store name is invalid
            StoreValidationError: If validation fails
        """
        if store_name not in self._stores:
            raise StoreAccessError(f"Unknown store: {store_name}")
        
        with self._locks[store_name]:
            if validate:
                self._validate_updates(store_name, updates)
            
            store = self._stores[store_name]
            
            if merge:
                self._deep_update(store, updates)
            else:
                self._stores[store_name] = updates
            
            self._update_modification_metadata()
            logger.debug(f"Updated {store_name} with {len(updates)} changes")
    
    def _deep_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries.
        
        Args:
            target: Target dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _validate_updates(self, store_name: str, updates: Dict[str, Any]) -> None:
        """Validate updates for a specific store.
        
        Args:
            store_name: Name of the store
            updates: Updates to validate
            
        Raises:
            StoreValidationError: If validation fails
        """
        validators = {
            'paper_store': self._validate_paper_updates,
            'citation_store': self._validate_citation_updates,
            'research_store': self._validate_research_updates,
            'session_store': self._validate_session_updates,
            'workflow_store': self._validate_workflow_updates
        }
        
        validator = validators.get(store_name)
        if validator:
            validator(updates)
    
    def _validate_paper_updates(self, updates: Dict[str, Any]) -> None:
        """Validate paper store updates."""
        if 'year' in updates and updates['year'] is not None:
            if not isinstance(updates['year'], int) or updates['year'] < 1900 or updates['year'] > 2100:
                raise StoreValidationError(f"Invalid year: {updates['year']}")
        
        if 'authors' in updates and not isinstance(updates['authors'], list):
            raise StoreValidationError("Authors must be a list")
        
        if 'pdf_path' in updates and updates['pdf_path']:
            if not isinstance(updates['pdf_path'], str):
                raise StoreValidationError("PDF path must be a string")
    
    def _validate_citation_updates(self, updates: Dict[str, Any]) -> None:
        """Validate citation store updates."""
        if 'citing_papers' in updates and not isinstance(updates['citing_papers'], list):
            raise StoreValidationError("Citing papers must be a list")
        
        if 'total_found' in updates and updates['total_found'] < 0:
            raise StoreValidationError("Total found cannot be negative")
        
        if 'search_metadata' in updates and 'relevance_threshold' in updates['search_metadata']:
            threshold = updates['search_metadata']['relevance_threshold']
            if not 0 <= threshold <= 1:
                raise StoreValidationError(f"Relevance threshold must be between 0 and 1: {threshold}")
    
    def _validate_research_updates(self, updates: Dict[str, Any]) -> None:
        """Validate research store updates."""
        if 'suggestions' in updates and not isinstance(updates['suggestions'], list):
            raise StoreValidationError("Suggestions must be a list")
        
        if 'confidence_scores' in updates:
            for key, score in updates['confidence_scores'].items():
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise StoreValidationError(f"Invalid confidence score for {key}: {score}")
    
    def _validate_session_updates(self, updates: Dict[str, Any]) -> None:
        """Validate session store updates."""
        if 'preferences' in updates:
            prefs = updates['preferences']
            if 'output_format' in prefs and prefs['output_format'] not in ['json', 'pdf', 'markdown']:
                raise StoreValidationError(f"Invalid output format: {prefs['output_format']}")
            
            if 'max_citations' in prefs and prefs['max_citations'] <= 0:
                raise StoreValidationError("Max citations must be positive")
    
    def _validate_workflow_updates(self, updates: Dict[str, Any]) -> None:
        """Validate workflow store updates."""
        valid_stages = ['initialized', 'pdf_processing', 'analysis', 'citation_search', 
                       'synthesis', 'formatting', 'completed', 'error']
        
        if 'current_stage' in updates and updates['current_stage'] not in valid_stages:
            raise StoreValidationError(f"Invalid workflow stage: {updates['current_stage']}")
        
        valid_statuses = ['idle', 'active', 'completed', 'error']
        if 'agent_status' in updates:
            for agent, status in updates['agent_status'].items():
                if status not in valid_statuses:
                    raise StoreValidationError(f"Invalid status for {agent}: {status}")
    
    def _update_access_metadata(self) -> None:
        """Update metadata for store access."""
        self._metadata['last_accessed'] = datetime.now().isoformat()
        self._metadata['access_count'] += 1
    
    def _update_modification_metadata(self) -> None:
        """Update metadata for store modification."""
        self._metadata['last_modified'] = datetime.now().isoformat()
        self._metadata['modification_count'] += 1
    
    def save_session(self, file_path: Optional[str] = None) -> str:
        """Save all stores to a file.
        
        Args:
            file_path: Path to save file (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"scholar_ai_session_{timestamp}.pkl"
            file_path = os.path.join(self.persistence_dir or ".", file_name)
        
        session_data = {
            'stores': {},
            'metadata': self._metadata,
            'version': '1.0'
        }
        
        # Get all stores with locks
        for store_name in self._stores:
            with self._locks[store_name]:
                session_data['stores'][store_name] = copy.deepcopy(self._stores[store_name])
        
        # Save to file
        with open(file_path, 'wb') as f:
            pickle.dump(session_data, f)
        
        logger.info(f"Saved session to {file_path}")
        return file_path
    
    def load_session(self, file_path: str) -> None:
        """Load stores from a saved session file.
        
        Args:
            file_path: Path to session file
            
        Raises:
            StoreError: If file cannot be loaded
        """
        try:
            with open(file_path, 'rb') as f:
                session_data = pickle.load(f)
            
            # Validate version
            if session_data.get('version') != '1.0':
                raise StoreError(f"Unsupported session version: {session_data.get('version')}")
            
            # Load stores
            for store_name, store_data in session_data['stores'].items():
                if store_name in self._stores:
                    with self._locks[store_name]:
                        self._stores[store_name] = store_data
            
            # Update metadata
            self._metadata = session_data['metadata']
            self._metadata['last_loaded'] = datetime.now().isoformat()
            
            logger.info(f"Loaded session from {file_path}")
            
        except Exception as e:
            raise StoreError(f"Failed to load session: {e}")
    
    def clear_store(self, store_name: str) -> None:
        """Clear a specific store to its initial state.
        
        Args:
            store_name: Name of store to clear
        """
        if store_name not in self._stores:
            raise StoreAccessError(f"Unknown store: {store_name}")
        
        with self._locks[store_name]:
            if store_name == 'paper_store':
                self._stores[store_name] = self._create_paper_store()
            elif store_name == 'citation_store':
                self._stores[store_name] = self._create_citation_store()
            elif store_name == 'research_store':
                self._stores[store_name] = self._create_research_store()
            elif store_name == 'session_store':
                self._stores[store_name] = self._create_session_store()
            elif store_name == 'workflow_store':
                self._stores[store_name] = self._create_workflow_store()
            
            logger.info(f"Cleared {store_name}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get store metadata.
        
        Returns:
            Store metadata dictionary
        """
        return copy.deepcopy(self._metadata)
    
    def export_as_json(self, file_path: Optional[str] = None) -> str:
        """Export all stores as JSON for inspection.
        
        Args:
            file_path: Path to save JSON file
            
        Returns:
            Path to saved file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"scholar_ai_export_{timestamp}.json"
            file_path = os.path.join(self.persistence_dir or ".", file_name)
        
        export_data = {
            'stores': {},
            'metadata': self._metadata,
            'exported_at': datetime.now().isoformat()
        }
        
        # Get all stores
        for store_name in self._stores:
            export_data['stores'][store_name] = self.get_store(store_name)
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported stores to {file_path}")
        return file_path


class AgentCommunicator:
    """Facilitates communication between agents through the shared store."""
    
    def __init__(self, shared_store: SharedStore, agent_name: str):
        """Initialize communicator for a specific agent.
        
        Args:
            shared_store: SharedStore instance
            agent_name: Name of the agent using this communicator
        """
        self.shared_store = shared_store
        self.agent_name = agent_name
        logger.info(f"Initialized AgentCommunicator for {agent_name}")
    
    def update_agent_status(self, status: str) -> None:
        """Update the status of the current agent.
        
        Args:
            status: New status (idle, active, completed, error)
        """
        self.shared_store.update_store('workflow_store', {
            'agent_status': {self.agent_name: status}
        })
    
    def request_citation_search(self, paper_metadata: Dict[str, Any]) -> None:
        """Request citation search from Web Search Agent.
        
        Args:
            paper_metadata: Paper metadata for search
        """
        message = {
            'type': 'citation_search_request',
            'from': self.agent_name,
            'to': 'web_search',
            'data': paper_metadata,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Add to inter-agent messages
        workflow_store = self.shared_store.get_store('workflow_store')
        messages = workflow_store.get('inter_agent_messages', [])
        messages.append(message)
        
        self.shared_store.update_store('workflow_store', {
            'inter_agent_messages': messages
        })
        
        # Update citation store with request
        self.shared_store.update_store('citation_store', {
            'search_status': 'requested',
            'search_metadata': {
                'requested_by': self.agent_name,
                'requested_at': datetime.now().isoformat()
            }
        })
        
        logger.info(f"{self.agent_name} requested citation search")
    
    def check_citation_results(self) -> Optional[Dict[str, Any]]:
        """Check if citation search results are available.
        
        Returns:
            Citation results if available, None otherwise
        """
        citation_store = self.shared_store.get_store('citation_store')
        
        if citation_store.get('search_status') == 'completed':
            return {
                'citing_papers': citation_store.get('citing_papers', []),
                'total_found': citation_store.get('total_found', 0),
                'filtered_count': citation_store.get('filtered_count', 0),
                'search_metadata': citation_store.get('search_metadata', {})
            }
        
        return None
    
    def request_research_synthesis(self, synthesis_data: Dict[str, Any]) -> None:
        """Request research synthesis from Research Synthesis Agent.
        
        Args:
            synthesis_data: Data for synthesis including paper and citations
        """
        message = {
            'type': 'synthesis_request',
            'from': self.agent_name,
            'to': 'research_synthesis',
            'data': synthesis_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Add to inter-agent messages
        workflow_store = self.shared_store.get_store('workflow_store')
        messages = workflow_store.get('inter_agent_messages', [])
        messages.append(message)
        
        self.shared_store.update_store('workflow_store', {
            'inter_agent_messages': messages
        })
        
        # Update research store with request
        self.shared_store.update_store('research_store', {
            'synthesis_status': 'requested',
            'synthesis_metadata': {
                'requested_by': self.agent_name,
                'requested_at': datetime.now().isoformat()
            }
        })
        
        logger.info(f"{self.agent_name} requested research synthesis")
    
    def check_synthesis_results(self) -> Optional[Dict[str, Any]]:
        """Check if research synthesis results are available.
        
        Returns:
            Synthesis results if available, None otherwise
        """
        research_store = self.shared_store.get_store('research_store')
        
        if research_store.get('synthesis_status') == 'completed':
            return {
                'suggestions': research_store.get('suggestions', []),
                'synthesis_data': research_store.get('synthesis_data', {}),
                'confidence_scores': research_store.get('confidence_scores', {}),
                'implementation_roadmap': research_store.get('implementation_roadmap', [])
            }
        
        return None
    
    def update_workflow_stage(self, stage: str) -> None:
        """Update the current workflow stage.
        
        Args:
            stage: New workflow stage
        """
        workflow_store = self.shared_store.get_store('workflow_store')
        completed_stages = workflow_store.get('completed_stages', [])
        
        # Add current stage to completed if not already there
        current = workflow_store.get('current_stage')
        if current and current not in completed_stages and current != 'initialized':
            completed_stages.append(current)
        
        self.shared_store.update_store('workflow_store', {
            'current_stage': stage,
            'completed_stages': completed_stages
        })
        
        logger.info(f"Workflow stage updated to: {stage}")
    
    def add_status_update(self, message: str, level: str = 'info') -> None:
        """Add a status update to the session store.
        
        Args:
            message: Status message
            level: Message level (info, warning, error)
        """
        session_store = self.shared_store.get_store('session_store')
        status_updates = session_store.get('status_updates', [])
        
        status_updates.append({
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'message': message,
            'level': level
        })
        
        self.shared_store.update_store('session_store', {
            'status_updates': status_updates,
            'last_activity': datetime.now().isoformat()
        })
    
    def log_error(self, error: str, error_type: str = 'general') -> None:
        """Log an error to the session store.
        
        Args:
            error: Error message
            error_type: Type of error
        """
        session_store = self.shared_store.get_store('session_store')
        error_log = session_store.get('error_log', [])
        
        error_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'error': error,
            'type': error_type
        })
        
        self.shared_store.update_store('session_store', {
            'error_log': error_log
        })
        
        logger.error(f"{self.agent_name} error: {error}")
    
    def update_performance_metric(self, metric_name: str, value: float) -> None:
        """Update a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.shared_store.update_store('session_store', {
            'performance_metrics': {metric_name: value}
        })
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get pending messages for this agent.
        
        Returns:
            List of pending messages
        """
        workflow_store = self.shared_store.get_store('workflow_store')
        messages = workflow_store.get('inter_agent_messages', [])
        
        # Filter messages for this agent that are pending
        pending = [
            msg for msg in messages
            if msg.get('to') == self.agent_name and msg.get('status') == 'pending'
        ]
        
        return pending
    
    def mark_message_processed(self, message: Dict[str, Any]) -> None:
        """Mark a message as processed.
        
        Args:
            message: Message to mark as processed
        """
        workflow_store = self.shared_store.get_store('workflow_store')
        messages = workflow_store.get('inter_agent_messages', [])
        
        # Find and update the message
        for msg in messages:
            if (msg.get('timestamp') == message.get('timestamp') and 
                msg.get('from') == message.get('from')):
                msg['status'] = 'processed'
                msg['processed_at'] = datetime.now().isoformat()
                break
        
        self.shared_store.update_store('workflow_store', {
            'inter_agent_messages': messages
        })