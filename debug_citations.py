#!/usr/bin/env python3
"""Debug script to test and fix citation search for papers."""

import asyncio
import logging
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_scholar_search_url(query: str, start: int = 0) -> str:
    """Build Google Scholar search URL."""
    import urllib.parse
    base_url = "https://scholar.google.com/scholar"
    params = {
        'q': query,
        'hl': 'en',
        'as_sdt': '0,5',
        'start': start
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def get_citations_url(cluster_id: str, start: int = 0) -> str:
    """Build Google Scholar citations URL using cluster ID."""
    import urllib.parse
    base_url = "https://scholar.google.com/scholar"
    params = {
        'cites': cluster_id,
        'hl': 'en',
        'as_sdt': '0,5',
        'start': start
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def search_for_paper(title: str, author: str = None) -> Optional[Dict]:
    """Search for a specific paper and get its cluster ID."""
    # Build query
    query = f'"{title}"'
    if author:
        query += f' {author}'
    
    url = get_scholar_search_url(query)
    logger.info(f"Searching for paper with query: {query}")
    logger.info(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36'
    }
    
    time.sleep(2)  # Rate limiting
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        logger.error(f"Failed to search: HTTP {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.text, 'lxml')
    
    # Find first result
    first_result = soup.find('div', class_='gs_r gs_or gs_scl')
    if not first_result:
        logger.error("No results found")
        return None
    
    # Extract title
    title_elem = first_result.find('h3', class_='gs_rt')
    if not title_elem:
        logger.error("No title found")
        return None
    
    found_title = title_elem.get_text().strip()
    logger.info(f"Found paper: {found_title}")
    
    # Find "Cited by" link to get cluster ID
    cited_by_link = first_result.find('a', string=re.compile(r'Cited by'))
    if not cited_by_link:
        logger.error("No 'Cited by' link found")
        return None
    
    href = cited_by_link.get('href', '')
    
    # Extract cluster ID from the cited by link
    # The link format is like: /scholar?cites=CLUSTER_ID&...
    cluster_match = re.search(r'cites=(\d+)', href)
    if not cluster_match:
        logger.error("Could not extract cluster ID from cited by link")
        return None
    
    cluster_id = cluster_match.group(1)
    
    # Extract citation count
    cited_text = cited_by_link.get_text()
    cited_count_match = re.search(r'Cited by (\d+)', cited_text)
    cited_count = int(cited_count_match.group(1)) if cited_count_match else 0
    
    return {
        'title': found_title,
        'cluster_id': cluster_id,
        'citation_count': cited_count,
        'cited_by_url': f"https://scholar.google.com{href}"
    }


def get_citing_papers(cluster_id: str, max_results: int = 10) -> List[Dict]:
    """Get papers that cite a work using its cluster ID."""
    url = get_citations_url(cluster_id)
    logger.info(f"Getting citations using cluster ID: {cluster_id}")
    logger.info(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36'
    }
    
    time.sleep(3)  # Rate limiting
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        logger.error(f"Failed to get citations: HTTP {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'lxml')
    
    # Parse citing papers
    results = []
    entries = soup.find_all('div', class_='gs_r gs_or gs_scl')
    
    for i, entry in enumerate(entries[:max_results]):
        # Extract title
        title_elem = entry.find('h3', class_='gs_rt')
        if not title_elem:
            continue
        
        title = title_elem.get_text().strip()
        
        # Extract authors and year
        authors_elem = entry.find('div', class_='gs_a')
        authors_text = authors_elem.get_text() if authors_elem else ''
        
        # Parse year
        year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
        year = int(year_match.group()) if year_match else None
        
        # Extract snippet
        snippet_elem = entry.find('div', class_='gs_rs')
        snippet = snippet_elem.get_text().strip() if snippet_elem else ''
        
        results.append({
            'title': title,
            'authors_text': authors_text,
            'year': year,
            'snippet': snippet[:200]
        })
        
        logger.info(f"Citation {i+1}: {title} ({year})")
    
    return results


def test_attention_paper():
    """Test with the Attention Is All You Need paper."""
    print("\n" + "="*60)
    print("Testing Citation Search for 'Attention Is All You Need'")
    print("="*60 + "\n")
    
    # First, find the paper and get its cluster ID
    paper_info = search_for_paper(
        title="Attention is all you need",
        author="Vaswani"
    )
    
    if not paper_info:
        print("❌ Failed to find the paper!")
        return
    
    print(f"\n✅ Found paper:")
    print(f"   Title: {paper_info['title']}")
    print(f"   Cluster ID: {paper_info['cluster_id']}")
    print(f"   Citation Count: {paper_info['citation_count']:,}")
    print(f"   Citations URL: {paper_info['cited_by_url']}")
    
    # Now get citing papers
    print(f"\n📚 Fetching citing papers...")
    citing_papers = get_citing_papers(paper_info['cluster_id'], max_results=5)
    
    if citing_papers:
        print(f"\n✅ Found {len(citing_papers)} citing papers:")
        for i, paper in enumerate(citing_papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Year: {paper['year']}")
            print(f"   Snippet: {paper['snippet']}...")
    else:
        print("❌ No citing papers found!")
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    print("\nThe issue is that the current implementation searches for the paper")
    print("by title, but doesn't use Google Scholar's citation feature properly.")
    print("\nTo fix this, we need to:")
    print("1. First find the paper and extract its cluster ID")
    print("2. Then use the 'cites=CLUSTER_ID' parameter to get citing papers")
    print("\nAlternatively, we can search for papers that mention the title")
    print("in their text/references, but that's less accurate.")


if __name__ == "__main__":
    test_attention_paper()