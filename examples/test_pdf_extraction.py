"""Example script demonstrating PDF extraction functionality."""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import PDFExtractorUtility, PDFExtractionError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pdf_extraction(pdf_path: str):
    """Test PDF extraction on a given file."""
    logger.info(f"Testing PDF extraction on: {pdf_path}")
    
    # Initialize extractor
    extractor = PDFExtractorUtility(cache_dir="./cache")
    
    # 1. Validate PDF
    logger.info("Step 1: Validating PDF...")
    is_valid, error = extractor.validate_pdf(pdf_path)
    if not is_valid:
        logger.error(f"PDF validation failed: {error}")
        return
    logger.info("✓ PDF is valid")
    
    # 2. Extract text
    logger.info("Step 2: Extracting text...")
    try:
        text_result = extractor.extract_text(pdf_path)
        logger.info(f"✓ Extracted {len(text_result['text'])} characters")
        logger.info(f"First 200 chars: {text_result['text'][:200]}...")
    except PDFExtractionError as e:
        logger.error(f"Text extraction failed: {e}")
        return
    
    # 3. Extract metadata
    logger.info("Step 3: Extracting metadata...")
    metadata_result = extractor.extract_metadata(pdf_path)
    if metadata_result["success"]:
        metadata = metadata_result["metadata"]
        logger.info(f"✓ Title: {metadata.get('title', 'Not found')}")
        logger.info(f"✓ Authors: {', '.join(metadata.get('authors', [])) or 'Not found'}")
        logger.info(f"✓ Year: {metadata.get('year', 'Not found')}")
        logger.info(f"✓ Abstract: {metadata.get('abstract', 'Not found')[:100]}...")
    else:
        logger.warning("Metadata extraction had issues")
    
    # 4. Extract sections
    logger.info("Step 4: Extracting sections...")
    sections_result = extractor.extract_sections(pdf_path)
    if sections_result["success"]:
        sections = sections_result["sections"]
        found_sections = [k for k, v in sections.items() if v]
        logger.info(f"✓ Found sections: {', '.join(found_sections)}")
    
    # 5. Extract references
    logger.info("Step 5: Extracting references...")
    refs_result = extractor.extract_references(pdf_path)
    if refs_result["success"]:
        logger.info(f"✓ Found {refs_result['count']} references")
        if refs_result['references']:
            logger.info(f"First reference: {refs_result['references'][0][:100]}...")
    
    logger.info("PDF extraction completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if Path(pdf_path).exists():
            test_pdf_extraction(pdf_path)
        else:
            logger.error(f"File not found: {pdf_path}")
    else:
        logger.info("Usage: python test_pdf_extraction.py <pdf_file_path>")
        logger.info("Creating a sample PDF for testing...")
        
        # Create a sample PDF for demonstration
        from utils.test_helpers import create_sample_pdf
        sample_path = create_sample_pdf()
        test_pdf_extraction(sample_path)