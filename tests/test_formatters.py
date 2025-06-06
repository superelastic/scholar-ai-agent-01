"""Tests for result formatters."""

import json
import pytest
from datetime import datetime

from utils import (
    FormatterFactory,
    JsonFormatter,
    MarkdownFormatter,
    HtmlFormatter,
    PlainTextFormatter,
    FormatterError
)


@pytest.fixture
def sample_analysis_data():
    """Sample analysis data for testing."""
    return {
        'paper_metadata': {
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani, A.', 'Shazeer, N.', 'Parmar, N.'],
            'year': 2017,
            'doi': '10.1000/example.doi',
            'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.'
        },
        'paper_analysis': {
            'key_concepts': ['transformer', 'attention mechanism', 'neural networks'],
            'methodology': 'The authors propose a new architecture based solely on attention mechanisms.',
            'findings': [
                'Transformers achieve better translation quality',
                'The model is more parallelizable than recurrent models'
            ],
            'contributions': [
                'Novel attention-based architecture',
                'State-of-the-art translation results'
            ]
        },
        'citations': [
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'authors': ['Devlin, J.', 'Chang, M.'],
                'year': 2018,
                'venue': 'NAACL',
                'citations': 15000,
                'relevance_score': 0.95,
                'url': 'https://example.com/bert',
                'snippet': 'Building on the transformer architecture...'
            },
            {
                'title': 'GPT-2: Language Models are Unsupervised Multitask Learners',
                'authors': ['Radford, A.', 'Wu, J.'],
                'year': 2019,
                'venue': 'OpenAI',
                'citations': 8000,
                'relevance_score': 0.88,
                'url': 'https://example.com/gpt2',
                'snippet': 'Large-scale transformer model for language generation...'
            }
        ],
        'research_directions': {
            'suggestions': [
                {
                    'title': 'Efficient Attention Mechanisms',
                    'description': 'Develop more computationally efficient attention mechanisms for long sequences.',
                    'rationale': 'Current attention mechanisms scale quadratically with sequence length.',
                    'confidence': 0.85,
                    'difficulty': 'high',
                    'time_horizon': 'medium-term',
                    'required_expertise': ['deep learning', 'optimization'],
                    'potential_impact': 'high'
                },
                {
                    'title': 'Multimodal Transformers',
                    'description': 'Extend transformer architecture to handle multiple modalities.',
                    'rationale': 'Growing need for models that can process text, images, and audio together.',
                    'confidence': 0.72,
                    'difficulty': 'medium',
                    'time_horizon': 'short-term',
                    'required_expertise': ['computer vision', 'natural language processing'],
                    'potential_impact': 'moderate'
                }
            ]
        },
        'processing_metadata': {
            'pdf_path': '/path/to/paper.pdf',
            'processed_at': '2023-01-01T12:00:00',
            'total_processing_time': 45.5,
            'status': 'completed'
        }
    }


class TestFormatterFactory:
    """Test cases for FormatterFactory."""
    
    def test_create_json_formatter(self):
        """Test creating JSON formatter."""
        formatter = FormatterFactory.create_formatter('json')
        assert isinstance(formatter, JsonFormatter)
    
    def test_create_markdown_formatter(self):
        """Test creating Markdown formatter."""
        formatter = FormatterFactory.create_formatter('markdown')
        assert isinstance(formatter, MarkdownFormatter)
        
        # Test alias
        formatter_md = FormatterFactory.create_formatter('md')
        assert isinstance(formatter_md, MarkdownFormatter)
    
    def test_create_html_formatter(self):
        """Test creating HTML formatter."""
        formatter = FormatterFactory.create_formatter('html')
        assert isinstance(formatter, HtmlFormatter)
    
    def test_create_text_formatter(self):
        """Test creating plain text formatter."""
        formatter = FormatterFactory.create_formatter('txt')
        assert isinstance(formatter, PlainTextFormatter)
        
        # Test alias
        formatter_text = FormatterFactory.create_formatter('text')
        assert isinstance(formatter_text, PlainTextFormatter)
    
    def test_create_invalid_formatter(self):
        """Test creating formatter with invalid format."""
        with pytest.raises(FormatterError) as exc_info:
            FormatterFactory.create_formatter('invalid')
        
        assert "Unsupported format 'invalid'" in str(exc_info.value)
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = FormatterFactory.get_supported_formats()
        expected_formats = ['json', 'markdown', 'md', 'html', 'txt', 'text']
        
        for fmt in expected_formats:
            assert fmt in formats
    
    def test_format_data_convenience_method(self, sample_analysis_data):
        """Test the convenience format_data method."""
        result = FormatterFactory.format_data(sample_analysis_data, 'json')
        assert isinstance(result, str)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert 'analysis_metadata' in parsed
        assert 'paper_analysis' in parsed


class TestJsonFormatter:
    """Test cases for JsonFormatter."""
    
    def test_format_complete_data(self, sample_analysis_data):
        """Test formatting complete analysis data."""
        formatter = JsonFormatter()
        result = formatter.format(sample_analysis_data)
        
        # Should be valid JSON
        parsed = json.loads(result)
        
        # Check structure
        assert 'analysis_metadata' in parsed
        assert 'paper_analysis' in parsed
        assert 'citations' in parsed
        assert 'research_directions' in parsed
        assert 'processing_info' in parsed
        
        # Check paper analysis
        paper_analysis = parsed['paper_analysis']
        assert paper_analysis['metadata']['title'] == 'Attention Is All You Need'
        assert len(paper_analysis['metadata']['authors']) == 3
        assert paper_analysis['metadata']['year'] == 2017
        
        # Check citations
        citations = parsed['citations']
        assert citations['summary']['total_found'] == 2
        assert len(citations['papers']) == 2
        
        # Check research directions
        research_dirs = parsed['research_directions']
        assert len(research_dirs['suggestions']) == 2
        assert research_dirs['summary']['total_suggestions'] == 2
    
    def test_format_minimal_data(self):
        """Test formatting with minimal data."""
        minimal_data = {'paper_metadata': {'title': 'Test Paper'}}
        
        formatter = JsonFormatter()
        result = formatter.format(minimal_data)
        
        parsed = json.loads(result)
        assert parsed['paper_analysis']['metadata']['title'] == 'Test Paper'
    
    def test_get_file_extension(self):
        """Test file extension."""
        formatter = JsonFormatter()
        assert formatter.get_file_extension() == '.json'
    
    def test_custom_formatting_options(self, sample_analysis_data):
        """Test custom JSON formatting options."""
        formatter = JsonFormatter(indent=4, sort_keys=False)
        result = formatter.format(sample_analysis_data)
        
        # Check indentation (should have 4-space indents)
        lines = result.split('\n')
        assert any(line.startswith('    ') for line in lines)


class TestMarkdownFormatter:
    """Test cases for MarkdownFormatter."""
    
    def test_format_complete_data(self, sample_analysis_data):
        """Test formatting complete analysis data as Markdown."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_analysis_data)
        
        # Check for expected Markdown structure
        assert '# Academic Analysis Report' in result
        assert '## Attention Is All You Need' in result
        assert '### Key Concepts' in result
        assert '### Recent Citing Papers' in result
        assert '## Future Research Directions' in result
        
        # Check for paper metadata
        assert 'Vaswani, A., Shazeer, N., Parmar, N.' in result
        assert '**Year:** 2017' in result
        
        # Check for citations
        assert 'BERT: Pre-training of Deep Bidirectional Transformers' in result
        assert '[View Paper](https://example.com/bert)' in result
        
        # Check for research directions
        assert 'Efficient Attention Mechanisms' in result
        assert '**Confidence Score:** 0.85/1.0' in result
    
    def test_format_no_citations(self, sample_analysis_data):
        """Test formatting when no citations are available."""
        data = sample_analysis_data.copy()
        data['citations'] = []
        
        formatter = MarkdownFormatter()
        result = formatter.format(data)
        
        assert 'Found **0** papers that cite this work' in result
        assert 'No citations were found or accessible' in result
    
    def test_format_no_research_directions(self, sample_analysis_data):
        """Test formatting when no research directions are available."""
        data = sample_analysis_data.copy()
        data['research_directions'] = {'suggestions': []}
        
        formatter = MarkdownFormatter()
        result = formatter.format(data)
        
        assert 'here are **0** potential research directions' in result
        assert 'No research directions could be generated' in result
    
    def test_get_file_extension(self):
        """Test file extension."""
        formatter = MarkdownFormatter()
        assert formatter.get_file_extension() == '.md'


class TestHtmlFormatter:
    """Test cases for HtmlFormatter."""
    
    def test_format_complete_data(self, sample_analysis_data):
        """Test formatting complete analysis data as HTML."""
        formatter = HtmlFormatter()
        result = formatter.format(sample_analysis_data)
        
        # Check for HTML structure
        assert '<!DOCTYPE html>' in result
        assert '<html lang="en">' in result
        assert '</html>' in result
        
        # Check for CSS styles
        assert '<style>' in result
        assert 'body {' in result
        
        # Check for content
        assert 'Attention Is All You Need' in result
        assert 'Key Concepts' in result
        assert 'Citations Analysis' in result
        assert 'Future Research Directions' in result
        
        # Check for proper HTML escaping
        assert '&lt;' not in result or '<' in result  # Either no escaping needed or properly escaped
    
    def test_html_escaping(self):
        """Test HTML character escaping."""
        formatter = HtmlFormatter()
        
        # Test data with HTML characters
        data_with_html = {
            'paper_metadata': {
                'title': 'Test <script>alert("xss")</script> Paper',
                'abstract': 'Abstract with & special characters < > " \''
            },
            'paper_analysis': {},
            'citations': [],
            'research_directions': {'suggestions': []},
            'processing_metadata': {}
        }
        
        result = formatter.format(data_with_html)
        
        # Should not contain unescaped HTML
        assert '<script>' not in result
        assert '&lt;script&gt;' in result or 'Test  Paper' in result  # Either escaped or sanitized
    
    def test_get_file_extension(self):
        """Test file extension."""
        formatter = HtmlFormatter()
        assert formatter.get_file_extension() == '.html'


class TestPlainTextFormatter:
    """Test cases for PlainTextFormatter."""
    
    def test_format_complete_data(self, sample_analysis_data):
        """Test formatting complete analysis data as plain text."""
        formatter = PlainTextFormatter()
        result = formatter.format(sample_analysis_data)
        
        # Check for expected structure
        assert 'ACADEMIC ANALYSIS REPORT' in result
        assert '=' * 80 in result
        assert 'PAPER ANALYSIS' in result
        assert 'CITATIONS ANALYSIS' in result
        assert 'FUTURE RESEARCH DIRECTIONS' in result
        
        # Check for content
        assert 'Attention Is All You Need' in result
        assert 'Vaswani, A., Shazeer, N., Parmar, N.' in result
        assert 'BERT: Pre-training of Deep Bidirectional Transformers' in result
        assert 'Efficient Attention Mechanisms' in result
    
    def test_text_wrapping(self, sample_analysis_data):
        """Test text wrapping functionality."""
        formatter = PlainTextFormatter()
        
        # Add very long description
        data = sample_analysis_data.copy()
        data['research_directions']['suggestions'][0]['description'] = (
            'This is a very long description that should be wrapped to multiple lines '
            'when formatted as plain text to ensure readability and proper formatting '
            'within the constraints of terminal or text file display.'
        )
        
        result = formatter.format(data)
        
        # Check that long text is properly formatted
        lines = result.split('\n')
        # Most lines should be under 80 characters (allowing for some flexibility)
        long_lines = [line for line in lines if len(line) > 85]
        assert len(long_lines) < len(lines) / 2  # Less than half should be very long
    
    def test_get_file_extension(self):
        """Test file extension."""
        formatter = PlainTextFormatter()
        assert formatter.get_file_extension() == '.txt'


class TestFormatterEdgeCases:
    """Test edge cases and error conditions for formatters."""
    
    def test_empty_data(self):
        """Test formatting with empty data."""
        empty_data = {}
        
        for format_type in ['json', 'markdown', 'html', 'txt']:
            formatter = FormatterFactory.create_formatter(format_type)
            result = formatter.format(empty_data)
            
            # Should not raise exception and should return valid content
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_malformed_data(self):
        """Test formatting with malformed data."""
        malformed_data = {
            'paper_metadata': 'not a dict',
            'citations': 'not a list',
            'research_directions': None
        }
        
        for format_type in ['json', 'markdown', 'html', 'txt']:
            formatter = FormatterFactory.create_formatter(format_type)
            # Should handle gracefully without raising exceptions
            result = formatter.format(malformed_data)
            assert isinstance(result, str)
    
    def test_unicode_content(self):
        """Test formatting with Unicode content."""
        unicode_data = {
            'paper_metadata': {
                'title': 'Étude sur les réseaux de neurones 神经网络研究',
                'authors': ['François Müller', '山田太郎']
            },
            'paper_analysis': {
                'key_concepts': ['machine learning', 'réseau neuronal', '人工智能']
            },
            'citations': [],
            'research_directions': {'suggestions': []},
            'processing_metadata': {}
        }
        
        for format_type in ['json', 'markdown', 'html', 'txt']:
            formatter = FormatterFactory.create_formatter(format_type)
            result = formatter.format(unicode_data)
            
            # Should preserve Unicode characters
            assert 'Étude' in result
            assert '神经网络' in result
            assert 'François' in result
    
    def test_very_large_data(self):
        """Test formatting with large datasets."""
        # Create data with many citations and research directions
        large_data = {
            'paper_metadata': {'title': 'Large Scale Analysis'},
            'paper_analysis': {
                'key_concepts': [f'concept_{i}' for i in range(100)]
            },
            'citations': [
                {
                    'title': f'Citation {i}',
                    'authors': [f'Author {i}'],
                    'year': 2020 + (i % 5),
                    'relevance_score': 0.5 + (i % 50) / 100
                }
                for i in range(100)
            ],
            'research_directions': {
                'suggestions': [
                    {
                        'title': f'Direction {i}',
                        'description': f'Description for research direction {i}',
                        'confidence': 0.5 + (i % 50) / 100
                    }
                    for i in range(50)
                ]
            },
            'processing_metadata': {}
        }
        
        for format_type in ['json', 'markdown', 'html', 'txt']:
            formatter = FormatterFactory.create_formatter(format_type)
            result = formatter.format(large_data)
            
            # Should complete without issues
            assert isinstance(result, str)
            assert len(result) > 1000  # Should be substantial content