"""Tests for export manager functionality."""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import (
    ExportManager,
    ExportError,
    export_analysis_results
)


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for export testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def export_manager(temp_export_dir):
    """Create an export manager for testing."""
    return ExportManager(temp_export_dir)


@pytest.fixture
def sample_analysis_data():
    """Sample analysis data for export testing."""
    return {
        'paper_metadata': {
            'title': 'Test Paper: A Comprehensive Study',
            'authors': ['Alice Smith', 'Bob Johnson'],
            'year': 2023,
            'doi': '10.1000/test.doi',
            'abstract': 'This is a test paper for export functionality.'
        },
        'paper_analysis': {
            'key_concepts': ['machine learning', 'data analysis', 'testing'],
            'methodology': 'Experimental approach with controlled testing.',
            'findings': [
                'Export functionality works correctly',
                'Multiple formats are supported'
            ],
            'contributions': [
                'Novel export system',
                'Comprehensive format support'
            ]
        },
        'citations': [
            {
                'title': 'Related Work 1',
                'authors': ['John Doe'],
                'year': 2022,
                'venue': 'Test Conference',
                'relevance_score': 0.85,
                'url': 'https://example.com/paper1'
            },
            {
                'title': 'Related Work 2',
                'authors': ['Jane Doe'],
                'year': 2021,
                'venue': 'Test Journal',
                'relevance_score': 0.72,
                'url': 'https://example.com/paper2'
            }
        ],
        'research_directions': {
            'suggestions': [
                {
                    'title': 'Future Direction 1',
                    'description': 'Explore advanced export features.',
                    'confidence': 0.8,
                    'difficulty': 'medium'
                },
                {
                    'title': 'Future Direction 2',
                    'description': 'Implement real-time export monitoring.',
                    'confidence': 0.9,
                    'difficulty': 'high'
                }
            ]
        },
        'processing_metadata': {
            'pdf_path': '/test/path/paper.pdf',
            'processed_at': '2023-01-01T12:00:00',
            'total_processing_time': 30.5,
            'status': 'completed'
        }
    }


class TestExportManager:
    """Test cases for ExportManager."""
    
    def test_initialization(self, temp_export_dir):
        """Test export manager initialization."""
        manager = ExportManager(temp_export_dir)
        
        assert manager.base_export_dir == Path(temp_export_dir)
        assert manager.base_export_dir.exists()
    
    def test_export_single_format(self, export_manager, sample_analysis_data):
        """Test exporting in a single format."""
        result_path = export_manager.export_single_file(
            sample_analysis_data,
            'json'
        )
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.json')
        
        # Verify content
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
        
        assert 'paper_analysis' in data
        assert data['paper_analysis']['metadata']['title'] == 'Test Paper: A Comprehensive Study'
    
    def test_export_multiple_formats(self, export_manager, sample_analysis_data):
        """Test exporting in multiple formats."""
        formats = ['json', 'markdown', 'html', 'txt']
        
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            formats,
            export_name='test_export'
        )
        
        assert export_info['success']
        assert len(export_info['exported_formats']) == 4
        assert len(export_info['failed_formats']) == 0
        assert len(export_info['files']) == 4
        
        # Check all files exist
        for format_type, file_path in export_info['files'].items():
            assert os.path.exists(file_path)
            assert format_type in ['json', 'markdown', 'html', 'txt']
        
        # Check metadata file exists
        assert os.path.exists(export_info['metadata_file'])
        
        # Check bundle file exists
        assert export_info['bundle_file'] is not None
        assert os.path.exists(export_info['bundle_file'])
        assert export_info['bundle_file'].endswith('.zip')
    
    def test_export_with_custom_name(self, export_manager, sample_analysis_data):
        """Test exporting with custom name."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            ['json'],
            export_name='custom_export_name'
        )
        
        assert export_info['success']
        assert 'custom_export_name' in export_info['export_session']['export_name']
        
        # Check file path contains custom name
        json_file = export_info['files']['json']
        assert 'custom_export_name' in json_file
    
    def test_export_without_bundle(self, export_manager, sample_analysis_data):
        """Test exporting without creating a bundle."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            ['json', 'markdown'],
            create_bundle=False
        )
        
        assert export_info['success']
        assert export_info['bundle_file'] is None
        assert len(export_info['files']) == 2
    
    def test_export_with_failing_format(self, export_manager, sample_analysis_data):
        """Test handling of failing format exports."""
        # Patch formatter to fail for HTML
        with patch('utils.formatters.FormatterFactory.create_formatter') as mock_factory:
            def side_effect(format_type):
                if format_type == 'html':
                    raise Exception("Mock HTML formatter error")
                # Return real formatters for other types
                from utils.formatters import FormatterFactory
                return FormatterFactory.__dict__['_formatters'][format_type]()
            
            mock_factory.side_effect = side_effect
            
            export_info = export_manager.export_analysis_results(
                sample_analysis_data,
                ['json', 'html', 'markdown']
            )
            
            assert export_info['success']  # Should still succeed partially
            assert len(export_info['exported_formats']) == 2  # json and markdown
            assert len(export_info['failed_formats']) == 1  # html failed
            assert export_info['failed_formats'][0]['format'] == 'html'
    
    def test_export_metadata_file_content(self, export_manager, sample_analysis_data):
        """Test content of generated metadata file."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            ['json']
        )
        
        metadata_file = export_info['metadata_file']
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert 'export_metadata' in metadata
        assert 'paper_metadata' in metadata
        assert 'analysis_summary' in metadata
        assert 'export_info' in metadata
        
        # Check specific content
        assert metadata['paper_metadata']['title'] == 'Test Paper: A Comprehensive Study'
        assert metadata['analysis_summary']['key_concepts_count'] == 3
        assert metadata['analysis_summary']['citations_count'] == 2
        assert metadata['export_info']['total_files'] == 1
    
    def test_export_bundle_content(self, export_manager, sample_analysis_data):
        """Test content of export bundle ZIP file."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            ['json', 'markdown']
        )
        
        bundle_file = export_info['bundle_file']
        
        with zipfile.ZipFile(bundle_file, 'r') as zf:
            file_list = zf.namelist()
            
            # Should contain exported files and metadata
            assert len(file_list) >= 3  # json, markdown, metadata
            
            # Check file extensions
            extensions = [Path(f).suffix for f in file_list]
            assert '.json' in extensions
            assert '.md' in extensions
    
    def test_list_exports(self, export_manager, sample_analysis_data):
        """Test listing exports."""
        # Create a few exports
        export_manager.export_analysis_results(sample_analysis_data, ['json'], export_name='export1')
        export_manager.export_analysis_results(sample_analysis_data, ['markdown'], export_name='export2')
        
        exports = export_manager.list_exports()
        
        assert len(exports) >= 2
        
        # Check export structure
        for export in exports:
            assert 'export_name' in export
            assert 'paper_title' in export
            assert 'created_at' in export
            assert 'formats' in export
            assert 'directory' in export
    
    def test_cleanup_old_exports(self, export_manager, sample_analysis_data):
        """Test cleaning up old exports."""
        import time
        
        # Create some exports with different names to avoid timestamp collision
        export_manager.export_analysis_results(sample_analysis_data, ['json'], export_name='export1')
        time.sleep(0.1)  # Small delay to ensure different timestamps
        export_manager.export_analysis_results(sample_analysis_data, ['markdown'], export_name='export2')
        
        # List exports before cleanup
        exports_before = export_manager.list_exports()
        assert len(exports_before) >= 2
        
        # Clean up (using 0 days to remove everything)
        cleanup_summary = export_manager.cleanup_old_exports(keep_days=0)
        
        assert cleanup_summary['deleted_directories'] >= 2
        assert cleanup_summary['errors'] == 0
        
        # List exports after cleanup
        exports_after = export_manager.list_exports()
        assert len(exports_after) == 0
    
    def test_safe_filename_generation(self, export_manager):
        """Test safe filename generation."""
        # Test with problematic characters
        problematic_titles = [
            'Paper with/slashes\\and<>quotes"',
            'Paper: With Colons & Ampersands',
            'Very Long Paper Title That Exceeds The Maximum Length Limit For Safe Filename Generation',
            '',
            None
        ]
        
        for title in problematic_titles:
            if title is None:
                safe_name = export_manager._make_safe_filename('')
            else:
                safe_name = export_manager._make_safe_filename(title)
            
            # Should not contain problematic characters
            assert '/' not in safe_name
            assert '\\' not in safe_name
            assert '<' not in safe_name
            assert '>' not in safe_name
            assert '"' not in safe_name
            assert ':' not in safe_name
            
            # Should not be empty
            assert len(safe_name) > 0
            
            # Should not be too long
            assert len(safe_name) <= 50
    
    def test_export_with_unicode_content(self, export_manager):
        """Test exporting content with Unicode characters."""
        unicode_data = {
            'paper_metadata': {
                'title': 'Étude sur les réseaux de neurones 神经网络研究',
                'authors': ['François Müller', '山田太郎'],
                'year': 2023
            },
            'paper_analysis': {
                'key_concepts': ['machine learning', 'réseau neuronal', '人工智能']
            },
            'citations': [],
            'research_directions': {'suggestions': []},
            'processing_metadata': {}
        }
        
        export_info = export_manager.export_analysis_results(
            unicode_data,
            ['json', 'markdown']
        )
        
        assert export_info['success']
        
        # Check that Unicode content is preserved
        json_file = export_info['files']['json']
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Étude' in content
            assert '神经网络' in content
    
    def test_export_custom_file_path(self, export_manager, sample_analysis_data, temp_export_dir):
        """Test exporting to a custom file path."""
        custom_path = os.path.join(temp_export_dir, 'custom_export.json')
        
        result_path = export_manager.export_single_file(
            sample_analysis_data,
            'json',
            custom_path
        )
        
        assert result_path == custom_path
        assert os.path.exists(custom_path)
    
    def test_export_error_handling(self, export_manager):
        """Test error handling during export."""
        # Test with invalid data
        invalid_data = None
        
        with pytest.raises(ExportError):
            export_manager.export_analysis_results(invalid_data, ['json'])
    
    def test_export_directory_permissions(self, temp_export_dir):
        """Test export with directory permission issues."""
        # Create manager with non-existent parent directory
        non_existent_path = os.path.join(temp_export_dir, 'non_existent', 'exports')
        manager = ExportManager(non_existent_path)
        
        # Should create directory automatically
        assert manager.base_export_dir.exists()


class TestExportConvenienceFunction:
    """Test cases for convenience export function."""
    
    def test_export_analysis_results_function(self, sample_analysis_data, temp_export_dir):
        """Test the convenience export function."""
        export_info = export_analysis_results(
            sample_analysis_data,
            formats=['json', 'markdown'],
            output_dir=temp_export_dir,
            export_name='convenience_test'
        )
        
        assert export_info['success']
        assert len(export_info['exported_formats']) == 2
        assert export_info['bundle_file'] is not None
    
    def test_export_with_default_formats(self, sample_analysis_data, temp_export_dir):
        """Test export with default formats."""
        export_info = export_analysis_results(
            sample_analysis_data,
            output_dir=temp_export_dir
        )
        
        assert export_info['success']
        # Should use all default formats
        assert len(export_info['exported_formats']) == 4
        expected_formats = {'json', 'markdown', 'html', 'txt'}
        assert set(export_info['exported_formats']) == expected_formats


class TestExportEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_export_empty_data(self, export_manager):
        """Test exporting empty analysis data."""
        empty_data = {}
        
        export_info = export_manager.export_analysis_results(
            empty_data,
            ['json']
        )
        
        # Should succeed even with empty data
        assert export_info['success']
        assert len(export_info['exported_formats']) == 1
    
    def test_export_malformed_data(self, export_manager):
        """Test exporting malformed data."""
        malformed_data = {
            'paper_metadata': 'not a dict',
            'citations': 'not a list',
            'research_directions': None
        }
        
        # Should handle gracefully
        export_info = export_manager.export_analysis_results(
            malformed_data,
            ['json']
        )
        
        assert export_info['success']
    
    def test_export_no_formats_requested(self, export_manager, sample_analysis_data):
        """Test export when no formats are requested."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            []
        )
        
        # Should succeed but with no exports
        assert export_info['success']
        assert len(export_info['exported_formats']) == 0
        assert len(export_info['failed_formats']) == 0
        assert export_info['bundle_file'] is None
    
    def test_export_invalid_format(self, export_manager, sample_analysis_data):
        """Test export with invalid format."""
        export_info = export_manager.export_analysis_results(
            sample_analysis_data,
            ['invalid_format']
        )
        
        # Should fail for invalid format
        assert not export_info['success'] or len(export_info['failed_formats']) > 0
        if export_info['failed_formats']:
            assert export_info['failed_formats'][0]['format'] == 'invalid_format'
    
    def test_export_very_large_data(self, export_manager):
        """Test exporting very large datasets."""
        # Create large dataset
        large_data = {
            'paper_metadata': {'title': 'Large Scale Test'},
            'paper_analysis': {
                'key_concepts': [f'concept_{i}' for i in range(1000)]
            },
            'citations': [
                {
                    'title': f'Citation {i}',
                    'authors': [f'Author {i}'],
                    'year': 2020 + (i % 5),
                    'relevance_score': 0.5 + (i % 50) / 100
                }
                for i in range(500)
            ],
            'research_directions': {
                'suggestions': [
                    {
                        'title': f'Direction {i}',
                        'description': f'Description for research direction {i} with substantial content',
                        'confidence': 0.5 + (i % 50) / 100
                    }
                    for i in range(100)
                ]
            },
            'processing_metadata': {}
        }
        
        export_info = export_manager.export_analysis_results(
            large_data,
            ['json', 'markdown']
        )
        
        assert export_info['success']
        
        # Check file sizes are reasonable
        for file_path in export_info['files'].values():
            file_size = os.path.getsize(file_path)
            assert file_size > 1000  # Should be substantial content
            assert file_size < 50 * 1024 * 1024  # But not unreasonably large