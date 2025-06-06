#!/usr/bin/env python3
"""Test the fixed citation search functionality."""

import logging
from utils.scholar_search import ScholarSearchUtility

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_attention_paper():
    """Test citation search for Attention Is All You Need paper."""
    print("\n" + "="*60)
    print("Testing Fixed Citation Search")
    print("="*60 + "\n")
    
    # Initialize scholar search utility
    scholar = ScholarSearchUtility()
    
    # Paper details
    paper_title = "Attention Is All You Need"
    authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", 
               "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", 
               "Lukasz Kaiser", "Illia Polosukhin"]
    year = 2017
    
    print(f"🔍 Searching for citations of: {paper_title}")
    print(f"👥 Authors: {', '.join(authors[:3])} et al.")
    print(f"📅 Year: {year}")
    print("\n" + "-"*50 + "\n")
    
    # Search for citations
    results = scholar.search_citations(
        paper_title=paper_title,
        authors=authors,
        year=year,
        max_results=10
    )
    
    if results['success']:
        print(f"✅ Search successful!")
        print(f"📊 Total citations found: {results['total_found']}")
        print(f"🔍 Query used: {results['query']}")
        print("\n📚 Recent citing papers:\n")
        
        for i, paper in enumerate(results['papers'][:5], 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors']) if paper['authors'] else 'Unknown'}")
            print(f"   Year: {paper['year']}")
            print(f"   Relevance: {paper['relevance_score']:.2f}")
            if paper.get('snippet'):
                print(f"   Snippet: {paper['snippet'][:150]}...")
            print()
    else:
        print(f"❌ Search failed: {results.get('error', 'Unknown error')}")
    
    print("="*60)


if __name__ == "__main__":
    test_attention_paper()