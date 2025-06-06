"""Progress tracking utility for long-running operations.

This module provides progress tracking and status indicators for the Scholar AI workflow.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProgressStage:
    """Represents a stage in the workflow progress."""
    name: str
    description: str
    weight: float = 1.0  # Relative weight for progress calculation
    estimated_duration: Optional[float] = None  # Estimated duration in seconds
    status: str = 'pending'  # pending, active, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percent: float = 0.0
    substages: List['ProgressStage'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Tracks progress of multi-stage workflows with real-time updates."""
    
    def __init__(self, stages: List[ProgressStage], callback: Optional[Callable] = None):
        """Initialize progress tracker.
        
        Args:
            stages: List of workflow stages to track
            callback: Optional callback function for progress updates
        """
        self.stages = stages
        self.callback = callback
        self.current_stage_index = 0
        self.overall_progress = 0.0
        self.status = 'initialized'
        self.start_time = None
        self.end_time = None
        self.errors = []
        self._lock = threading.RLock()
        
        # Calculate stage weights
        self.total_weight = sum(stage.weight for stage in stages)
        
        logger.info(f"Initialized ProgressTracker with {len(stages)} stages")
    
    def start(self) -> None:
        """Start progress tracking."""
        with self._lock:
            self.start_time = datetime.now()
            self.status = 'active'
            
            if self.stages:
                self.stages[0].status = 'active'
                self.stages[0].start_time = self.start_time
            
            self._notify_progress()
            logger.info("Progress tracking started")
    
    def advance_stage(self, stage_name: Optional[str] = None, progress_percent: float = 100.0) -> None:
        """Advance to the next stage or update current stage progress.
        
        Args:
            stage_name: Optional stage name to advance to (defaults to next stage)
            progress_percent: Progress percentage for current stage
        """
        with self._lock:
            if self.current_stage_index < len(self.stages):
                current_stage = self.stages[self.current_stage_index]
                
                if stage_name and current_stage.name != stage_name:
                    # Find the named stage
                    for i, stage in enumerate(self.stages):
                        if stage.name == stage_name:
                            self.current_stage_index = i
                            current_stage = stage
                            break
                
                # Update current stage progress
                current_stage.progress_percent = progress_percent
                
                # If stage is complete, mark it and move to next
                if progress_percent >= 100.0:
                    current_stage.status = 'completed'
                    current_stage.end_time = datetime.now()
                    current_stage.progress_percent = 100.0
                    
                    # Move to next stage
                    self.current_stage_index += 1
                    if self.current_stage_index < len(self.stages):
                        next_stage = self.stages[self.current_stage_index]
                        next_stage.status = 'active'
                        next_stage.start_time = datetime.now()
                        logger.info(f"Advanced to stage: {next_stage.name}")
                
                self._calculate_overall_progress()
                self._notify_progress()
    
    def update_stage_progress(self, progress_percent: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress of the current stage.
        
        Args:
            progress_percent: Progress percentage (0-100)
            metadata: Optional metadata to store with progress
        """
        with self._lock:
            if self.current_stage_index < len(self.stages):
                current_stage = self.stages[self.current_stage_index]
                current_stage.progress_percent = min(100.0, max(0.0, progress_percent))
                
                if metadata:
                    current_stage.metadata.update(metadata)
                
                self._calculate_overall_progress()
                self._notify_progress()
    
    def fail_current_stage(self, error_message: str) -> None:
        """Mark the current stage as failed.
        
        Args:
            error_message: Error message describing the failure
        """
        with self._lock:
            if self.current_stage_index < len(self.stages):
                current_stage = self.stages[self.current_stage_index]
                current_stage.status = 'failed'
                current_stage.end_time = datetime.now()
                current_stage.metadata['error'] = error_message
                
                self.errors.append({
                    'stage': current_stage.name,
                    'error': error_message,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.status = 'failed'
                self._notify_progress()
                logger.error(f"Stage '{current_stage.name}' failed: {error_message}")
    
    def complete(self) -> None:
        """Mark the entire workflow as completed."""
        with self._lock:
            # Mark any remaining stages as completed
            for stage in self.stages:
                if stage.status in ['pending', 'active']:
                    stage.status = 'completed'
                    stage.progress_percent = 100.0
                    if not stage.end_time:
                        stage.end_time = datetime.now()
            
            self.status = 'completed'
            self.end_time = datetime.now()
            self.overall_progress = 100.0
            self.current_stage_index = len(self.stages)  # Set beyond last stage
            self._notify_progress()
            logger.info("Progress tracking completed")
    
    def get_current_stage(self) -> Optional[ProgressStage]:
        """Get the current active stage.
        
        Returns:
            Current stage or None if all stages are complete
        """
        with self._lock:
            if self.current_stage_index < len(self.stages):
                return self.stages[self.current_stage_index]
            return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress.
        
        Returns:
            Progress summary dictionary
        """
        with self._lock:
            current_stage = self.get_current_stage()
            
            return {
                'overall_progress': self.overall_progress,
                'status': self.status,
                'current_stage': current_stage.name if current_stage else None,
                'current_stage_progress': current_stage.progress_percent if current_stage else 100.0,
                'completed_stages': len([s for s in self.stages if s.status == 'completed']),
                'total_stages': len(self.stages),
                'elapsed_time': self._get_elapsed_time(),
                'estimated_remaining_time': self._estimate_remaining_time(),
                'errors': self.errors.copy()
            }
    
    def get_detailed_progress(self) -> Dict[str, Any]:
        """Get detailed progress information for all stages.
        
        Returns:
            Detailed progress dictionary
        """
        with self._lock:
            return {
                'summary': self.get_progress_summary(),
                'stages': [
                    {
                        'name': stage.name,
                        'description': stage.description,
                        'status': stage.status,
                        'progress_percent': stage.progress_percent,
                        'weight': stage.weight,
                        'estimated_duration': stage.estimated_duration,
                        'actual_duration': self._get_stage_duration(stage),
                        'start_time': stage.start_time.isoformat() if stage.start_time else None,
                        'end_time': stage.end_time.isoformat() if stage.end_time else None,
                        'metadata': stage.metadata.copy()
                    }
                    for stage in self.stages
                ]
            }
    
    def _calculate_overall_progress(self) -> None:
        """Calculate overall progress based on stage weights and progress."""
        if not self.stages:
            self.overall_progress = 100.0
            return
        
        total_progress = 0.0
        
        for stage in self.stages:
            stage_contribution = (stage.weight / self.total_weight) * stage.progress_percent
            total_progress += stage_contribution
        
        self.overall_progress = min(100.0, total_progress)
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on progress and elapsed time.
        
        Returns:
            Estimated remaining time in seconds or None
        """
        if not self.start_time or self.overall_progress <= 0:
            return None
        
        elapsed = self._get_elapsed_time()
        if elapsed <= 0:
            return None
        
        # Simple linear estimation
        progress_rate = self.overall_progress / elapsed
        remaining_progress = 100.0 - self.overall_progress
        
        if progress_rate > 0:
            return remaining_progress / progress_rate
        
        return None
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if not self.start_time:
            return 0.0
        
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def _get_stage_duration(self, stage: ProgressStage) -> Optional[float]:
        """Get the duration of a stage in seconds.
        
        Args:
            stage: Progress stage
            
        Returns:
            Duration in seconds or None
        """
        if not stage.start_time:
            return None
        
        end_time = stage.end_time or (datetime.now() if stage.status == 'active' else None)
        if not end_time:
            return None
        
        return (end_time - stage.start_time).total_seconds()
    
    def _notify_progress(self) -> None:
        """Notify callback of progress update."""
        if self.callback:
            try:
                self.callback(self.get_progress_summary())
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


class ScholarAIProgressTracker(ProgressTracker):
    """Specialized progress tracker for Scholar AI workflow."""
    
    @classmethod
    def create_default(cls, callback: Optional[Callable] = None) -> 'ScholarAIProgressTracker':
        """Create a progress tracker with default Scholar AI stages.
        
        Args:
            callback: Optional callback for progress updates
            
        Returns:
            Configured progress tracker
        """
        stages = [
            ProgressStage(
                name="pdf_processing",
                description="Processing PDF file and extracting content",
                weight=1.5,
                estimated_duration=10.0
            ),
            ProgressStage(
                name="paper_analysis", 
                description="Analyzing paper content with AI",
                weight=2.0,
                estimated_duration=15.0
            ),
            ProgressStage(
                name="citation_search",
                description="Searching for citing papers",
                weight=2.5,
                estimated_duration=20.0
            ),
            ProgressStage(
                name="research_synthesis",
                description="Generating research directions",
                weight=2.0,
                estimated_duration=15.0
            ),
            ProgressStage(
                name="result_formatting",
                description="Formatting results for presentation",
                weight=1.0,
                estimated_duration=5.0
            )
        ]
        
        return cls(stages, callback)


def create_progress_indicator(title: str = "Processing", width: int = 50) -> Callable:
    """Create a simple console progress indicator.
    
    Args:
        title: Title to display
        width: Width of progress bar
        
    Returns:
        Callback function for progress updates
    """
    def progress_callback(progress_summary: Dict[str, Any]) -> None:
        """Console progress callback."""
        progress = progress_summary.get('overall_progress', 0.0)
        status = progress_summary.get('status', 'unknown')
        current_stage = progress_summary.get('current_stage', 'Processing')
        
        # Create progress bar
        filled = int((progress / 100.0) * width)
        bar = '█' * filled + '░' * (width - filled)
        
        # Format time
        elapsed = progress_summary.get('elapsed_time', 0.0)
        remaining = progress_summary.get('estimated_remaining_time')
        
        time_str = f"{elapsed:.1f}s"
        if remaining:
            time_str += f" (est. {remaining:.1f}s remaining)"
        
        # Print progress line
        print(f"\r{title}: {bar} {progress:.1f}% | {current_stage} | {time_str}", end='', flush=True)
        
        # Print newline when complete
        if status in ['completed', 'failed']:
            print()
    
    return progress_callback