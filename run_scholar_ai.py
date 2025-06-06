#!/usr/bin/env python3
"""Simple command-line interface for running the Scholar AI Agent.

This script provides an easy way to analyze academic papers using the Scholar AI system.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from coordinator import ScholarAICoordinator
from utils import export_analysis_results


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("🎓 Scholar AI Agent - Academic Paper Analysis System")
    print("=" * 60)
    print("Analyzes academic papers, finds citations, and suggests research directions")
    print()


def print_results_summary(results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    if not results.get('success'):
        print("❌ Analysis failed:", results.get('error', 'Unknown error'))
        return
    
    data = results['results']
    print("\n✅ Analysis completed successfully!")
    print("=" * 50)
    
    # Paper info
    paper_analysis = data.get('paper_analysis', {})
    metadata = paper_analysis.get('metadata', {})
    analysis = paper_analysis.get('analysis', {})
    
    print(f"📄 Paper: {metadata.get('title', 'Unknown')}")
    if metadata.get('authors'):
        print(f"👥 Authors: {', '.join(metadata['authors'][:3])}")
        if len(metadata['authors']) > 3:
            print(f"          + {len(metadata['authors']) - 3} more")
    if metadata.get('year'):
        print(f"📅 Year: {metadata['year']}")
    
    # Key concepts
    key_concepts = analysis.get('key_concepts', [])
    if key_concepts:
        print(f"\n🔑 Key Concepts ({len(key_concepts)}):")
        for concept in key_concepts[:5]:
            print(f"   • {concept}")
        if len(key_concepts) > 5:
            print(f"   + {len(key_concepts) - 5} more")
    
    # Citations
    citations = data.get('citations', {})
    citation_count = citations.get('summary', {}).get('filtered_count', 0)
    print(f"\n📚 Citations Found: {citation_count}")
    
    # Research directions
    research_dirs = data.get('research_directions', {})
    suggestions = research_dirs.get('suggestions', [])
    print(f"🔬 Research Directions: {len(suggestions)}")
    
    # Performance
    if 'performance' in results:
        total_time = results['performance'].get('total_time', 0)
        print(f"⏱️  Processing Time: {total_time:.1f} seconds")
    
    print("=" * 50)


async def analyze_paper(
    pdf_path: str, 
    output_dir: Optional[str] = None, 
    formats: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Analyze a single paper and export results.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for output files
        formats: List of output formats
        verbose: Enable verbose logging
        
    Returns:
        Analysis results
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Validate PDF file
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    print(f"📄 Analyzing: {pdf_file.name}")
    print("⏳ This may take 30-60 seconds...")
    print()
    
    # Initialize coordinator
    coordinator = ScholarAICoordinator(
        config={
            'coordinator': {
                'timeout': 300,  # 5 minutes
                'upload_dir': './uploads'
            },
            'web_search': {
                'max_results': 20,
                'year_filter': 2020  # Papers from 2020 onwards
            },
            'research_synthesis': {
                'min_confidence': 0.7
            }
        },
        persistence_dir='./sessions'
    )
    
    # Process the paper
    try:
        results = await coordinator.process_paper(str(pdf_file.absolute()))
        
        # Export results if requested
        if results.get('success') and (output_dir or formats):
            export_formats = formats or ['json', 'markdown', 'html']
            export_dir = output_dir or './exports'
            
            print(f"\n📁 Exporting results to: {export_dir}")
            export_info = export_analysis_results(
                results['results'],
                formats=export_formats,
                output_dir=export_dir,
                export_name=pdf_file.stem
            )
            
            if export_info['success']:
                print("✅ Export successful!")
                for format_type, file_path in export_info['files'].items():
                    print(f"   {format_type.upper()}: {file_path}")
                
                if export_info.get('bundle_file'):
                    print(f"   ZIP Bundle: {export_info['bundle_file']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scholar AI Agent - Analyze academic papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scholar_ai.py paper.pdf
  python run_scholar_ai.py paper.pdf --output-dir ./results
  python run_scholar_ai.py paper.pdf --formats json markdown --verbose
  python run_scholar_ai.py paper.pdf --export-all
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the PDF file to analyze'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Directory for output files (default: ./exports)',
        default=None
    )
    
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['json', 'markdown', 'html', 'txt'],
        help='Output formats (default: json markdown html)',
        default=None
    )
    
    parser.add_argument(
        '--export-all',
        action='store_true',
        help='Export in all available formats'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip file export, just show results'
    )
    
    args = parser.parse_args()
    
    # Handle format options
    if args.export_all:
        formats: Optional[List[str]] = ['json', 'markdown', 'html', 'txt']
    elif args.no_export:
        formats = None
        args.output_dir = None
    else:
        formats = args.formats
    
    print_welcome()
    
    try:
        # Run the analysis
        results = asyncio.run(analyze_paper(
            args.pdf_path,
            args.output_dir,
            formats,
            args.verbose
        ))
        
        # Print results summary
        print_results_summary(results)
        
        # Exit with appropriate code
        if results.get('success'):
            print("🎉 Analysis complete! Check the exported files for detailed results.")
            sys.exit(0)
        else:
            print("❌ Analysis failed. Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()