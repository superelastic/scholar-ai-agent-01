"""Tests for Google Scholar search utilities."""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from utils import (
    ScholarSearchUtility, 
    ScholarSearchError, 
    ScholarRateLimitError,
    ScholarParsingError
)


@pytest.fixture
def scholar_utility():
    """Create Scholar search utility for testing."""
    return ScholarSearchUtility(
        cache_ttl_hours=1,
        min_request_interval=0.1,  # Faster for testing
        max_retries=2,
        timeout=5
    )


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing."""
    return {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017
    }


@pytest.fixture
def mock_scholar_html():
    """Mock Google Scholar HTML response."""
    return """
    <html>
    <body>
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com/paper1">BERT: Pre-training of Deep Bidirectional Transformers</a>
            </h3>
            <div class="gs_a">Jacob Devlin, Ming-Wei Chang - 2018 - arxiv.org</div>
            <span class="gs_rs">We introduce a new language representation model called BERT...</span>
            <div class="gs_fl">
                <a href="#">Cited by 12345</a>
            </div>
        </div>
        <div class="gs_r gs_or gs_scl">
            <h3 class="gs_rt">
                <a href="https://example.com/paper2">GPT-2: Language Models are Unsupervised Multitask Learners</a>
            </h3>
            <div class="gs_a">Alec Radford, Jeff Wu - 2019 - openai.com</div>
            <span class="gs_rs">Natural language processing tasks are framed as a text generation problem...</span>
            <div class="gs_fl">
                <a href="#">Cited by 8765</a>
            </div>
        </div>
    </body>
    </html>
    """


def test_scholar_utility_initialization():
    """Test Scholar utility initialization."""
    utility = ScholarSearchUtility(
        cache_ttl_hours=12,
        min_request_interval=1.5,
        max_retries=5
    )
    
    assert utility.cache_ttl == timedelta(hours=12)
    assert utility.min_request_interval == 1.5
    assert utility.max_retries == 5
    assert utility.cache == {}
    assert len(utility.user_agents) > 0


def test_cache_key_generation(scholar_utility):
    """Test cache key generation."""
    key1 = scholar_utility._get_cache_key("Test Paper", ["Author A"], 2020)
    key2 = scholar_utility._get_cache_key("Test Paper", ["Author A"], 2020)
    key3 = scholar_utility._get_cache_key("Different Paper", ["Author A"], 2020)
    
    assert key1 == key2  # Same inputs should generate same key
    assert key1 != key3  # Different inputs should generate different keys


def test_cache_operations(scholar_utility):
    """Test cache save and retrieve operations."""
    test_data = {"results": ["paper1", "paper2"]}
    cache_key = "test_key"
    
    # Save to cache
    scholar_utility._save_to_cache(cache_key, test_data)
    
    # Retrieve from cache
    cached_result = scholar_utility._get_cached_result(cache_key)
    assert cached_result == test_data
    
    # Test cache expiry
    scholar_utility.cache_ttl = timedelta(seconds=-1)  # Expired
    cached_result = scholar_utility._get_cached_result(cache_key)
    assert cached_result is None


def test_rate_limiting(scholar_utility):
    """Test rate limiting functionality."""
    start_time = time.time()
    
    # First request should not wait
    scholar_utility._respect_rate_limit()
    first_request_time = time.time() - start_time
    
    # Second request should wait
    scholar_utility._respect_rate_limit()
    second_request_time = time.time() - start_time
    
    # Should have waited at least the minimum interval
    assert second_request_time >= scholar_utility.min_request_interval


def test_query_building(scholar_utility, sample_paper_data):
    """Test search query construction."""
    query = scholar_utility._build_citation_query(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    assert "attention" in query.lower()
    assert "vaswani" in query.lower()
    assert '"' in query  # Should have quoted phrases


def test_scholar_url_building(scholar_utility):
    """Test Scholar URL construction."""
    query = "machine learning"
    url = scholar_utility._build_scholar_url(query, start=10)
    
    assert "scholar.google.com" in url
    assert "machine%20learning" in url or "machine+learning" in url
    assert "start=10" in url


def test_random_headers(scholar_utility):
    """Test random header generation."""
    headers1 = scholar_utility._get_random_headers()
    headers2 = scholar_utility._get_random_headers()
    
    assert 'User-Agent' in headers1
    assert 'User-Agent' in headers2
    assert len(headers1['User-Agent']) > 20  # Should be a real user agent


def test_parse_scholar_page(scholar_utility, mock_scholar_html):
    """Test parsing of Scholar results page."""
    results = scholar_utility._parse_scholar_page(mock_scholar_html)
    
    assert len(results) == 2
    
    # Check first result
    assert results[0]['title'] == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert "Jacob Devlin" in results[0]['authors']
    assert results[0]['year'] == 2018
    assert results[0]['cited_by'] == 12345
    assert "https://example.com/paper1" in results[0]['url']
    
    # Check second result
    assert results[1]['title'] == "GPT-2: Language Models are Unsupervised Multitask Learners"
    assert results[1]['year'] == 2019
    assert results[1]['cited_by'] == 8765


def test_relevance_score_calculation(scholar_utility, sample_paper_data):
    """Test relevance score calculation."""
    result = {
        'title': 'Transformer Networks for Attention Mechanisms',
        'authors': ['Ashish Vaswani', 'Other Author'],
        'year': 2018,
        'cited_by': 50
    }
    
    score = scholar_utility._calculate_relevance_score(
        result,
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    assert 0 <= score <= 1
    assert score > 0  # Should have some relevance due to title and author overlap


def test_filter_results(scholar_utility):
    """Test result filtering functionality."""
    results = [
        {'year': 2020, 'relevance_score': 0.8, 'title': 'Recent Paper'},
        {'year': 2015, 'relevance_score': 0.9, 'title': 'Old Paper'},
        {'year': 2021, 'relevance_score': 0.2, 'title': 'Low Relevance Paper'},
        {'year': 2019, 'relevance_score': 0.7, 'title': 'Good Paper'}
    ]
    
    # Filter by year and relevance
    filtered = scholar_utility.filter_results(
        results, 
        min_year=2018, 
        relevance_threshold=0.5
    )
    
    assert len(filtered) == 2  # Should keep 2020 and 2019 papers with high relevance
    assert all(r['year'] >= 2018 for r in filtered)
    assert all(r['relevance_score'] >= 0.5 for r in filtered)


def test_format_citations(scholar_utility):
    """Test citation formatting."""
    results = [
        {
            'title': 'Paper 1',
            'authors': ['Author A', 'Author B'],
            'year': 2020,
            'venue': 'Conference A',
            'cited_by': 100,
            'relevance_score': 0.9
        },
        {
            'title': 'Paper 2',
            'authors': ['Author C'],
            'year': 2021,
            'venue': 'Journal B',
            'cited_by': 50,
            'relevance_score': 0.7
        }
    ]
    
    formatted = scholar_utility.format_citations(results)
    
    assert formatted['summary']['total_count'] == 2
    assert formatted['summary']['year_range']['min'] == 2020
    assert formatted['summary']['year_range']['max'] == 2021
    assert len(formatted['citations']) == 2
    assert formatted['citations'][0]['rank'] == 1
    assert formatted['citations'][1]['rank'] == 2


@patch('requests.get')
def test_search_citations_success(mock_get, scholar_utility, sample_paper_data, mock_scholar_html):
    """Test successful citation search."""
    # Mock successful response
    mock_response = Mock()
    mock_response.text = mock_scholar_html
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"],
        max_results=5
    )
    
    assert result['success'] is True
    assert result['total_found'] >= 2
    assert len(result['papers']) >= 2
    assert 'query' in result
    assert 'search_timestamp' in result


@patch('requests.get')
def test_search_citations_cache_hit(mock_get, scholar_utility, sample_paper_data):
    """Test that search uses cache when available."""
    # First call
    mock_response = Mock()
    mock_response.text = "<html><body></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result1 = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    # Second call should use cache
    result2 = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    # Should only make one HTTP request
    assert mock_get.call_count == 1
    assert result1 == result2


@patch('requests.get')
def test_search_citations_rate_limit_error(mock_get, scholar_utility, sample_paper_data):
    """Test handling of rate limit responses."""
    # Mock rate limit response
    mock_response = Mock()
    mock_response.text = "unusual traffic from your computer network"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    assert result['success'] is False
    assert 'error' in result


@patch('requests.get')
def test_search_citations_network_error(mock_get, scholar_utility, sample_paper_data):
    """Test handling of network errors."""
    # Mock network error
    mock_get.side_effect = requests.RequestException("Network error")
    
    result = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    
    assert result['success'] is False
    assert 'error' in result


@patch('requests.get')
def test_search_citations_retry_logic(mock_get, scholar_utility, sample_paper_data):
    """Test retry logic with failures."""
    # First call fails, second succeeds
    mock_get.side_effect = [
        requests.RequestException("Temporary failure"),
        Mock(text="<html><body></body></html>", raise_for_status=Mock())
    ]
    
    start_time = time.time()
    result = scholar_utility.search_citations(
        sample_paper_data["title"],
        sample_paper_data["authors"],
        sample_paper_data["year"]
    )
    end_time = time.time()
    
    # Should have made 2 requests
    assert mock_get.call_count == 2
    # Should have had some delay from retry
    assert end_time - start_time >= 1.0


def test_format_citations_empty_results(scholar_utility):
    """Test formatting with empty results."""
    formatted = scholar_utility.format_citations([])
    
    assert formatted['summary']['total_count'] == 0
    assert formatted['summary']['year_range'] is None
    assert len(formatted['citations']) == 0