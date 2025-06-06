"""Tests for progress tracking functionality."""

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from utils import (
    ProgressTracker,
    ProgressStage,
    ScholarAIProgressTracker,
    create_progress_indicator
)


@pytest.fixture
def sample_stages():
    """Create sample progress stages for testing."""
    return [
        ProgressStage(
            name="stage1",
            description="First stage",
            weight=1.0,
            estimated_duration=10.0
        ),
        ProgressStage(
            name="stage2",
            description="Second stage",
            weight=2.0,
            estimated_duration=20.0
        ),
        ProgressStage(
            name="stage3",
            description="Third stage",
            weight=1.5,
            estimated_duration=15.0
        )
    ]


@pytest.fixture
def progress_tracker(sample_stages):
    """Create a progress tracker for testing."""
    return ProgressTracker(sample_stages)


class TestProgressStage:
    """Test cases for ProgressStage."""
    
    def test_stage_initialization(self):
        """Test stage initialization with default values."""
        stage = ProgressStage("test_stage", "Test stage")
        
        assert stage.name == "test_stage"
        assert stage.description == "Test stage"
        assert stage.weight == 1.0
        assert stage.estimated_duration is None
        assert stage.status == 'pending'
        assert stage.start_time is None
        assert stage.end_time is None
        assert stage.progress_percent == 0.0
        assert stage.substages == []
        assert stage.metadata == {}
    
    def test_stage_with_custom_values(self):
        """Test stage initialization with custom values."""
        stage = ProgressStage(
            name="custom_stage",
            description="Custom stage",
            weight=2.5,
            estimated_duration=30.0
        )
        
        assert stage.weight == 2.5
        assert stage.estimated_duration == 30.0


class TestProgressTracker:
    """Test cases for ProgressTracker."""
    
    def test_initialization(self, sample_stages):
        """Test tracker initialization."""
        tracker = ProgressTracker(sample_stages)
        
        assert len(tracker.stages) == 3
        assert tracker.current_stage_index == 0
        assert tracker.overall_progress == 0.0
        assert tracker.status == 'initialized'
        assert tracker.start_time is None
        assert tracker.end_time is None
        assert tracker.errors == []
        assert tracker.total_weight == 4.5  # 1.0 + 2.0 + 1.5
    
    def test_start_tracking(self, progress_tracker):
        """Test starting progress tracking."""
        progress_tracker.start()
        
        assert progress_tracker.status == 'active'
        assert progress_tracker.start_time is not None
        assert progress_tracker.stages[0].status == 'active'
        assert progress_tracker.stages[0].start_time is not None
    
    def test_advance_stage(self, progress_tracker):
        """Test advancing through stages."""
        progress_tracker.start()
        
        # Complete first stage
        progress_tracker.advance_stage(progress_percent=100.0)
        
        assert progress_tracker.stages[0].status == 'completed'
        assert progress_tracker.stages[0].end_time is not None
        assert progress_tracker.stages[0].progress_percent == 100.0
        assert progress_tracker.current_stage_index == 1
        assert progress_tracker.stages[1].status == 'active'
        
        # Complete second stage
        progress_tracker.advance_stage(progress_percent=100.0)
        
        assert progress_tracker.stages[1].status == 'completed'
        assert progress_tracker.current_stage_index == 2
        assert progress_tracker.stages[2].status == 'active'
    
    def test_update_stage_progress(self, progress_tracker):
        """Test updating current stage progress."""
        progress_tracker.start()
        
        # Update progress of first stage
        progress_tracker.update_stage_progress(50.0, {'test_key': 'test_value'})
        
        assert progress_tracker.stages[0].progress_percent == 50.0
        assert progress_tracker.stages[0].metadata['test_key'] == 'test_value'
        
        # Overall progress should be calculated based on weights
        # Stage 1: 50% of weight 1.0 = 0.5
        # Total weight: 4.5
        # Expected overall: (0.5 / 4.5) * 100 = ~11.11%
        expected_overall = (0.5 / 4.5) * 100
        assert abs(progress_tracker.overall_progress - expected_overall) < 0.1
    
    def test_fail_current_stage(self, progress_tracker):
        """Test failing the current stage."""
        progress_tracker.start()
        
        error_message = "Something went wrong"
        progress_tracker.fail_current_stage(error_message)
        
        assert progress_tracker.stages[0].status == 'failed'
        assert progress_tracker.stages[0].end_time is not None
        assert progress_tracker.stages[0].metadata['error'] == error_message
        assert progress_tracker.status == 'failed'
        assert len(progress_tracker.errors) == 1
        assert progress_tracker.errors[0]['error'] == error_message
    
    def test_complete_workflow(self, progress_tracker):
        """Test completing the entire workflow."""
        progress_tracker.start()
        
        progress_tracker.complete()
        
        assert progress_tracker.status == 'completed'
        assert progress_tracker.end_time is not None
        assert progress_tracker.overall_progress == 100.0
        
        # All stages should be marked as completed
        for stage in progress_tracker.stages:
            assert stage.status == 'completed'
            assert stage.progress_percent == 100.0
            assert stage.end_time is not None
    
    def test_get_current_stage(self, progress_tracker):
        """Test getting current active stage."""
        progress_tracker.start()
        
        current = progress_tracker.get_current_stage()
        assert current.name == "stage1"
        
        # Advance to next stage
        progress_tracker.advance_stage(progress_percent=100.0)
        
        current = progress_tracker.get_current_stage()
        assert current.name == "stage2"
        
        # Complete all stages
        progress_tracker.complete()
        
        current = progress_tracker.get_current_stage()
        assert current is None  # No active stage when completed
    
    def test_get_progress_summary(self, progress_tracker):
        """Test getting progress summary."""
        progress_tracker.start()
        progress_tracker.update_stage_progress(25.0)
        
        summary = progress_tracker.get_progress_summary()
        
        assert 'overall_progress' in summary
        assert 'status' in summary
        assert 'current_stage' in summary
        assert 'current_stage_progress' in summary
        assert 'completed_stages' in summary
        assert 'total_stages' in summary
        assert 'elapsed_time' in summary
        assert 'errors' in summary
        
        assert summary['status'] == 'active'
        assert summary['current_stage'] == 'stage1'
        assert summary['current_stage_progress'] == 25.0
        assert summary['completed_stages'] == 0
        assert summary['total_stages'] == 3
    
    def test_get_detailed_progress(self, progress_tracker):
        """Test getting detailed progress information."""
        progress_tracker.start()
        progress_tracker.update_stage_progress(75.0, {'detail': 'processing'})
        
        detailed = progress_tracker.get_detailed_progress()
        
        assert 'summary' in detailed
        assert 'stages' in detailed
        
        stages_info = detailed['stages']
        assert len(stages_info) == 3
        
        first_stage = stages_info[0]
        assert first_stage['name'] == 'stage1'
        assert first_stage['status'] == 'active'
        assert first_stage['progress_percent'] == 75.0
        assert first_stage['metadata']['detail'] == 'processing'
        assert first_stage['start_time'] is not None
    
    def test_thread_safety(self, progress_tracker):
        """Test thread-safe operations."""
        progress_tracker.start()
        
        results = []
        errors = []
        
        def update_progress(progress_value, thread_id):
            try:
                for i in range(10):
                    progress_tracker.update_stage_progress(
                        progress_value + i,
                        {'thread_id': thread_id, 'iteration': i}
                    )
                    time.sleep(0.001)  # Small delay
                results.append(f'Thread {thread_id} completed')
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_progress, args=(i * 10, i))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        assert len(results) == 5
        
        # Progress should be set to some value
        assert progress_tracker.stages[0].progress_percent >= 0
    
    def test_callback_notification(self, sample_stages):
        """Test callback notification on progress updates."""
        callback_calls = []
        
        def test_callback(progress_summary):
            callback_calls.append(progress_summary.copy())
        
        tracker = ProgressTracker(sample_stages, callback=test_callback)
        
        tracker.start()
        tracker.update_stage_progress(50.0)
        tracker.advance_stage(progress_percent=100.0)
        
        # Should have received multiple callback calls
        assert len(callback_calls) >= 3  # start, update, advance
        
        # Last call should reflect current state
        last_call = callback_calls[-1]
        assert last_call['current_stage'] == 'stage2'
    
    def test_callback_error_handling(self, sample_stages):
        """Test error handling in callback."""
        def failing_callback(progress_summary):
            raise Exception("Callback error")
        
        tracker = ProgressTracker(sample_stages, callback=failing_callback)
        
        # Should not raise exception even if callback fails
        tracker.start()
        tracker.update_stage_progress(50.0)
        
        # Tracker should continue working normally
        assert tracker.status == 'active'
        assert tracker.overall_progress > 0


class TestScholarAIProgressTracker:
    """Test cases for ScholarAIProgressTracker."""
    
    def test_create_default(self):
        """Test creating default Scholar AI progress tracker."""
        tracker = ScholarAIProgressTracker.create_default()
        
        assert len(tracker.stages) == 5
        
        # Check expected stages
        stage_names = [stage.name for stage in tracker.stages]
        expected_names = [
            'pdf_processing',
            'paper_analysis',
            'citation_search',
            'research_synthesis',
            'result_formatting'
        ]
        
        assert stage_names == expected_names
        
        # Check weights and durations are set
        for stage in tracker.stages:
            assert stage.weight > 0
            assert stage.estimated_duration is not None
            assert stage.estimated_duration > 0
    
    def test_full_workflow_simulation(self):
        """Test simulating a full Scholar AI workflow."""
        callback_calls = []
        
        def progress_callback(summary):
            callback_calls.append(summary['overall_progress'])
        
        tracker = ScholarAIProgressTracker.create_default(callback=progress_callback)
        tracker.start()
        
        # Simulate PDF processing
        tracker.update_stage_progress(50.0)
        tracker.advance_stage(progress_percent=100.0)
        
        # Simulate paper analysis
        tracker.update_stage_progress(75.0)
        tracker.advance_stage(progress_percent=100.0)
        
        # Simulate citation search
        tracker.update_stage_progress(30.0)
        tracker.update_stage_progress(80.0)
        tracker.advance_stage(progress_percent=100.0)
        
        # Simulate research synthesis
        tracker.advance_stage(progress_percent=100.0)
        
        # Complete formatting
        tracker.complete()
        
        assert tracker.status == 'completed'
        assert tracker.overall_progress == 100.0
        
        # Should have received progress updates
        assert len(callback_calls) > 5
        assert callback_calls[-1] == 100.0  # Final progress should be 100%


class TestProgressIndicator:
    """Test cases for progress indicator creation."""
    
    def test_create_progress_indicator(self):
        """Test creating a console progress indicator."""
        indicator = create_progress_indicator("Test Process", width=30)
        
        # Should be callable
        assert callable(indicator)
        
        # Test with sample progress data
        progress_data = {
            'overall_progress': 45.5,
            'status': 'active',
            'current_stage': 'processing',
            'elapsed_time': 12.5,
            'estimated_remaining_time': 15.2
        }
        
        # Should not raise exception
        indicator(progress_data)
    
    def test_progress_indicator_edge_cases(self):
        """Test progress indicator with edge cases."""
        indicator = create_progress_indicator()
        
        # Test with minimal data
        minimal_data = {'overall_progress': 0.0}
        indicator(minimal_data)
        
        # Test with completed status
        completed_data = {
            'overall_progress': 100.0,
            'status': 'completed',
            'current_stage': 'done',
            'elapsed_time': 60.0
        }
        indicator(completed_data)
        
        # Test with failed status
        failed_data = {
            'overall_progress': 50.0,
            'status': 'failed',
            'current_stage': 'error',
            'elapsed_time': 30.0
        }
        indicator(failed_data)


class TestProgressTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_stages_list(self):
        """Test tracker with no stages."""
        tracker = ProgressTracker([])
        
        tracker.start()
        tracker.complete()
        
        assert tracker.status == 'completed'
        assert tracker.overall_progress == 100.0
        assert tracker.get_current_stage() is None
    
    def test_advance_beyond_last_stage(self, progress_tracker):
        """Test advancing beyond the last stage."""
        progress_tracker.start()
        
        # Complete all stages
        for _ in range(len(progress_tracker.stages)):
            progress_tracker.advance_stage(progress_percent=100.0)
        
        # Try to advance again - should not cause error
        progress_tracker.advance_stage(progress_percent=100.0)
        
        assert progress_tracker.current_stage_index == len(progress_tracker.stages)
        assert progress_tracker.get_current_stage() is None
    
    def test_negative_progress(self, progress_tracker):
        """Test handling negative progress values."""
        progress_tracker.start()
        
        progress_tracker.update_stage_progress(-10.0)
        
        # Should clamp to 0
        assert progress_tracker.stages[0].progress_percent == 0.0
    
    def test_progress_over_100(self, progress_tracker):
        """Test handling progress values over 100."""
        progress_tracker.start()
        
        progress_tracker.update_stage_progress(150.0)
        
        # Should clamp to 100
        assert progress_tracker.stages[0].progress_percent == 100.0
    
    def test_advance_to_specific_stage(self, progress_tracker):
        """Test advancing to a specific named stage."""
        progress_tracker.start()
        
        # Advance directly to stage3
        progress_tracker.advance_stage(stage_name="stage3", progress_percent=50.0)
        
        # Should have found and advanced to stage3
        current = progress_tracker.get_current_stage()
        assert current.name == "stage3"
        assert current.progress_percent == 50.0
    
    def test_advance_to_nonexistent_stage(self, progress_tracker):
        """Test advancing to a stage that doesn't exist."""
        progress_tracker.start()
        
        # Try to advance to non-existent stage
        progress_tracker.advance_stage(stage_name="nonexistent", progress_percent=50.0)
        
        # Should stay on current stage
        current = progress_tracker.get_current_stage()
        assert current.name == "stage1"
        assert current.progress_percent == 50.0