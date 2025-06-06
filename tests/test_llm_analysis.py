"""Tests for LLM analysis utilities."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils import LLMAnalysisUtility, LLMAnalysisError, LLMTimeoutError, LLMAPIError


@pytest.fixture
def llm_utility():
    """Create LLM utility with mock credentials."""
    return LLMAnalysisUtility(
        openai_api_key="test_openai_key",
        anthropic_api_key="test_anthropic_key",
        model="gpt-4",
        timeout=5,
        cache_ttl_hours=1
    )


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing."""
    return {
        "text": "This is a sample academic paper about machine learning and neural networks. The paper presents a novel approach to deep learning using transformer architectures.",
        "metadata": {
            "title": "Novel Approaches to Deep Learning",
            "authors": ["John Doe", "Jane Smith"],
            "year": 2023,
            "key_concepts": ["machine learning", "neural networks", "deep learning"]
        }
    }


@pytest.fixture
def sample_citations():
    """Sample citations data for testing."""
    return [
        {
            "title": "Advances in Neural Architecture Search",
            "authors": ["Alice Johnson"],
            "year": 2024,
            "relevance_score": 0.9
        },
        {
            "title": "Transformer Models for Computer Vision",
            "authors": ["Bob Wilson"],
            "year": 2024,
            "relevance_score": 0.8
        }
    ]


def test_llm_utility_initialization():
    """Test LLM utility initialization."""
    utility = LLMAnalysisUtility(
        openai_api_key="test_key",
        model="gpt-4",
        timeout=30
    )
    
    assert utility.openai_api_key == "test_key"
    assert utility.model == "gpt-4"
    assert utility.timeout == 30
    assert utility.cache == {}


def test_cache_key_generation(llm_utility):
    """Test cache key generation."""
    key1 = llm_utility._get_cache_key("test text", "analysis")
    key2 = llm_utility._get_cache_key("test text", "analysis")
    key3 = llm_utility._get_cache_key("different text", "analysis")
    
    assert key1 == key2  # Same inputs should generate same key
    assert key1 != key3  # Different inputs should generate different keys


def test_cache_operations(llm_utility):
    """Test cache save and retrieve operations."""
    cache_key = "test_key"
    test_data = {"result": "test_analysis"}
    
    # Save to cache
    llm_utility._save_to_cache(cache_key, test_data)
    
    # Retrieve from cache
    cached_result = llm_utility._get_cached_result(cache_key)
    assert cached_result == test_data
    
    # Test cache expiry
    llm_utility.cache_ttl = timedelta(seconds=-1)  # Expired
    cached_result = llm_utility._get_cached_result(cache_key)
    assert cached_result is None


def test_prompt_creation(llm_utility, sample_paper_data):
    """Test prompt creation for different operations."""
    # Test analysis prompt
    analysis_prompt = llm_utility._create_analysis_prompt(
        sample_paper_data["text"], 
        sample_paper_data["metadata"]
    )
    
    assert "Novel Approaches to Deep Learning" in analysis_prompt
    assert "John Doe" in analysis_prompt
    assert "machine learning" in analysis_prompt
    
    # Test query generation prompt
    query_prompt = llm_utility._create_query_generation_prompt(
        sample_paper_data["metadata"]
    )
    
    assert "Google Scholar" in query_prompt
    assert "Novel Approaches to Deep Learning" in query_prompt


@pytest.mark.asyncio
async def test_analyze_paper_success(llm_utility, sample_paper_data):
    """Test successful paper analysis."""
    mock_response = {
        "key_concepts": ["machine learning", "neural networks"],
        "methodology": "Deep learning with transformers",
        "findings": ["Improved accuracy", "Reduced training time"],
        "theoretical_framework": "Transformer architecture",
        "limitations": ["Limited dataset size"],
        "future_work": ["Explore larger datasets"]
    }
    
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(mock_response)
        
        result = await llm_utility.analyze_paper(
            sample_paper_data["text"],
            sample_paper_data["metadata"]
        )
        
        assert result["success"] is True
        assert result["key_concepts"] == ["machine learning", "neural networks"]
        assert result["methodology"] == "Deep learning with transformers"
        assert "timestamp" in result


@pytest.mark.asyncio
async def test_analyze_paper_cache_hit(llm_utility, sample_paper_data):
    """Test that analysis uses cache when available."""
    # First call
    mock_response = {"key_concepts": ["test"]}
    
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(mock_response)
        
        result1 = await llm_utility.analyze_paper(
            sample_paper_data["text"],
            sample_paper_data["metadata"]
        )
        
        # Second call should use cache
        result2 = await llm_utility.analyze_paper(
            sample_paper_data["text"],
            sample_paper_data["metadata"]
        )
        
        # LLM should only be called once
        assert mock_llm.call_count == 1
        assert result1 == result2


@pytest.mark.asyncio
async def test_analyze_paper_fallback(llm_utility, sample_paper_data):
    """Test fallback when LLM calls fail."""
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = LLMAPIError("API failed")
        
        result = await llm_utility.analyze_paper(
            sample_paper_data["text"],
            sample_paper_data["metadata"]
        )
        
        assert "fallback_used" in result
        assert result["fallback_used"] is True
        assert "findings" in result


@pytest.mark.asyncio
async def test_generate_search_queries_success(llm_utility, sample_paper_data):
    """Test successful query generation."""
    mock_response = {
        "queries": [
            '"Novel Approaches to Deep Learning"',
            '"Novel Approaches to Deep Learning" "John Doe"',
            'machine learning neural networks 2023'
        ]
    }
    
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(mock_response)
        
        result = await llm_utility.generate_search_queries(
            sample_paper_data["metadata"]
        )
        
        assert result["success"] is True
        assert len(result["queries"]) == 3
        assert "Novel Approaches to Deep Learning" in result["queries"][0]


@pytest.mark.asyncio
async def test_generate_search_queries_fallback(llm_utility, sample_paper_data):
    """Test query generation fallback."""
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = LLMAPIError("API failed")
        
        result = await llm_utility.generate_search_queries(
            sample_paper_data["metadata"]
        )
        
        assert result["success"] is False
        assert result["fallback_used"] is True
        assert len(result["queries"]) > 0


@pytest.mark.asyncio
async def test_synthesize_research_directions(llm_utility, sample_paper_data, sample_citations):
    """Test research direction synthesis."""
    mock_response = {
        "suggestions": [
            {
                "title": "Efficient Transformer Training",
                "description": "Develop methods to reduce training time",
                "rationale": "Current training is computationally expensive",
                "confidence": 0.85
            },
            {
                "title": "Multi-modal Applications",
                "description": "Apply transformers to multiple data types",
                "rationale": "Expanding application domains",
                "confidence": 0.75
            }
        ]
    }
    
    with patch.object(llm_utility, '_call_llm_with_retry', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(mock_response)
        
        result = await llm_utility.synthesize_research_directions(
            sample_paper_data["metadata"],
            sample_citations
        )
        
        assert result["success"] is True
        assert len(result["suggestions"]) == 2
        assert result["suggestions"][0]["confidence"] == 0.85


@pytest.mark.asyncio
async def test_format_presentation(llm_utility):
    """Test presentation formatting."""
    analysis_results = {
        "paper_metadata": {"title": "Test Paper"},
        "paper_analysis": {"findings": ["finding1", "finding2"]},
        "citations": [{"title": "Citation 1"}, {"title": "Citation 2"}],
        "research_directions": {"suggestions": [{"title": "Direction 1"}]}
    }
    
    result = await llm_utility.format_presentation(analysis_results)
    
    assert result["success"] is True
    assert "summary" in result["presentation"]
    assert result["presentation"]["summary"]["key_findings_count"] == 2
    assert result["presentation"]["summary"]["citations_found"] == 2


@pytest.mark.asyncio
async def test_llm_timeout_handling(llm_utility):
    """Test timeout handling in LLM calls."""
    with patch.object(llm_utility, '_call_openai', new_callable=AsyncMock) as mock_openai:
        mock_openai.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(LLMTimeoutError):
            await llm_utility._call_llm_with_retry("test prompt")


@pytest.mark.asyncio
async def test_llm_retry_logic(llm_utility):
    """Test retry logic with exponential backoff."""
    with patch.object(llm_utility, '_call_openai', new_callable=AsyncMock) as mock_openai:
        # Fail twice, then succeed
        mock_openai.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            "Success response"
        ]
        
        start_time = time.time()
        result = await llm_utility._call_llm_with_retry("test prompt")
        end_time = time.time()
        
        assert result == "Success response"
        assert mock_openai.call_count == 3
        # Should have delays from retries
        assert end_time - start_time >= 3.0  # 1 + 2 seconds delay


def test_simplified_analysis_generation(llm_utility, sample_paper_data):
    """Test simplified analysis generation."""
    result = llm_utility._generate_simplified_analysis(
        sample_paper_data["text"],
        sample_paper_data["metadata"]
    )
    
    assert "fallback_used" in result
    assert result["fallback_used"] is True
    assert "key_concepts" in result
    assert "timestamp" in result