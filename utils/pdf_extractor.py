"""PDF extraction and processing utilities."""

import hashlib
import json
import logging
import os
import re
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


class PDFExtractorUtility:
    """Utility class for PDF parsing, validation, and text extraction."""
    
    def __init__(self, cache_dir: str = "./cache", timeout_seconds: int = 5):
        """Initialize PDF extractor with caching support.
        
        Args:
            cache_dir: Directory for caching processed PDFs
            timeout_seconds: Maximum seconds allowed for PDF processing
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.executor = ThreadPoolExecutor(max_workers=1)
        logger.info(f"Initialized PDFExtractorUtility with cache at {self.cache_dir}")
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Unique cache key for the file
        """
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str, operation: str) -> Optional[Dict]:
        """Retrieve cached result if available.
        
        Args:
            cache_key: Unique cache key
            operation: Type of operation (text, metadata, etc.)
            
        Returns:
            Cached result or None
        """
        cache_file = self.cache_dir / f"{cache_key}_{operation}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    logger.info(f"Cache hit for {operation} operation")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, operation: str, data: Dict) -> None:
        """Save result to cache.
        
        Args:
            cache_key: Unique cache key
            operation: Type of operation
            data: Data to cache
        """
        cache_file = self.cache_dir / f"{cache_key}_{operation}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Cached result for {operation} operation")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Check if file is a valid PDF and safe to process.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists and has .pdf extension
            path = Path(file_path)
            if not path.exists():
                return False, "File does not exist"
            
            if path.suffix.lower() != '.pdf':
                return False, "File is not a PDF"
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                return False, f"File too large ({file_size_mb:.1f}MB > 50MB)"
            
            # Try to open and read PDF
            with open(file_path, 'rb') as f:
                # Check PDF header
                header = f.read(5)
                if header != b'%PDF-':
                    return False, "Invalid PDF header"
                
                # Try to parse PDF
                f.seek(0)
                pdf = PdfReader(f)
                
                if len(pdf.pages) == 0:
                    return False, "PDF contains no pages"
                
                # Basic sanitization - check for JavaScript
                if pdf.metadata and '/JavaScript' in str(pdf.metadata):
                    return False, "PDF contains JavaScript (potential security risk)"
                
            logger.info(f"PDF validation successful: {file_path}")
            return True, None
            
        except PdfReadError as e:
            return False, f"PDF read error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _extract_text_worker(self, file_path: str) -> str:
        """Worker function to extract text with timeout protection.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text_content = []
        
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            total_pages = len(pdf.pages)
            
            for page_num in range(total_pages):
                try:
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                    text_content.append(f"--- Page {page_num + 1} ---\n[Extraction failed]")
        
        return "\n\n".join(text_content)
    
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """Extract full text content from PDF with timeout handling.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with text content and metadata
        """
        cache_key = self._get_cache_key(file_path)
        cached = self._get_cached_result(cache_key, "text")
        if cached:
            return cached
        
        # Validate PDF first
        is_valid, error = self.validate_pdf(file_path)
        if not is_valid:
            raise PDFExtractionError(f"Invalid PDF: {error}")
        
        try:
            # Extract text with timeout
            future = self.executor.submit(self._extract_text_worker, file_path)
            text = future.result(timeout=self.timeout_seconds)
            
            result = {
                "text": text,
                "file_path": file_path,
                "extracted_at": datetime.now().isoformat(),
                "success": True
            }
            
            self._save_to_cache(cache_key, "text", result)
            return result
            
        except TimeoutError:
            raise PDFExtractionError(f"Text extraction timed out after {self.timeout_seconds} seconds")
        except Exception as e:
            raise PDFExtractionError(f"Text extraction failed: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict[str, any]:
        """Extract metadata like title, authors, and year from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted metadata
        """
        cache_key = self._get_cache_key(file_path)
        cached = self._get_cached_result(cache_key, "metadata")
        if cached:
            return cached
        
        # Extract text first
        text_data = self.extract_text(file_path)
        text = text_data["text"]
        
        metadata = {
            "title": None,
            "authors": [],
            "year": None,
            "abstract": None
        }
        
        try:
            # Extract from PDF metadata
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                if pdf.metadata:
                    metadata["title"] = pdf.metadata.get('/Title', None)
                    if pdf.metadata.get('/Author'):
                        metadata["authors"] = [pdf.metadata.get('/Author')]
            
            # Extract from text content if not in metadata
            if not metadata["title"] and text:
                # Look for title in first few lines, skip page markers
                lines = text.split('\n')[:30]
                for line in lines:
                    line = line.strip()
                    # Skip page markers and other non-title lines
                    if (line.startswith('---') and line.endswith('---')) or \
                       line.lower().startswith('provided') or \
                       line.lower().startswith('copyright') or \
                       '@' in line or \
                       len(line) < 10:
                        continue
                    # Likely title: proper length, not all lowercase, no special markers
                    if 10 < len(line) < 200 and not line.islower():
                        metadata["title"] = line
                        break
            
            # Extract authors if not in metadata
            if not metadata["authors"] and text:
                # Look for author names near the title
                title_index = text.find(metadata["title"]) if metadata["title"] else 0
                if title_index >= 0:
                    # Look in the next 1000 characters after title
                    author_text = text[title_index:title_index + 1000]
                    # Find lines with email addresses (often indicate authors)
                    email_pattern = r'([A-Za-z\s\.]+)\s*[\n\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
                    author_matches = re.findall(email_pattern, author_text)
                    if author_matches:
                        metadata["authors"] = [match[0].strip() for match in author_matches[:8]]  # Limit to 8 authors
            
            # Extract year using regex - look for 4-digit years
            year_pattern = r'\b(19[89]\d|20[0-2]\d)\b'  # Years 1980-2029
            years = re.findall(year_pattern, text[:5000])  # Look in first part
            if years:
                # Take the most recent year found
                metadata["year"] = max(int(y) for y in years)
            
            # Extract abstract
            abstract_match = re.search(r'abstract[:\s]*(.*?)(?:introduction|keywords|\n\n)', 
                                     text[:10000], re.IGNORECASE | re.DOTALL)
            if abstract_match:
                metadata["abstract"] = abstract_match.group(1).strip()[:1000]
            
            result = {
                "metadata": metadata,
                "file_path": file_path,
                "extracted_at": datetime.now().isoformat(),
                "success": True
            }
            
            self._save_to_cache(cache_key, "metadata", result)
            return result
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {
                "metadata": metadata,
                "file_path": file_path,
                "error": str(e),
                "success": False
            }
    
    def extract_sections(self, file_path: str) -> Dict[str, any]:
        """Identify and extract paper sections.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with identified sections
        """
        cache_key = self._get_cache_key(file_path)
        cached = self._get_cached_result(cache_key, "sections")
        if cached:
            return cached
        
        text_data = self.extract_text(file_path)
        text = text_data["text"].lower()
        
        sections = {
            "abstract": None,
            "introduction": None,
            "methodology": None,
            "results": None,
            "discussion": None,
            "conclusion": None,
            "references": None
        }
        
        # Define section patterns
        section_patterns = {
            "abstract": r"abstract[:\s]*(.*?)(?:introduction|keywords|\n\n)",
            "introduction": r"introduction[:\s]*(.*?)(?:methodology|methods|related work|\n\n)",
            "methodology": r"(?:methodology|methods)[:\s]*(.*?)(?:results|experiments|\n\n)",
            "results": r"results[:\s]*(.*?)(?:discussion|conclusion|\n\n)",
            "discussion": r"discussion[:\s]*(.*?)(?:conclusion|references|\n\n)",
            "conclusion": r"conclusion[:\s]*(.*?)(?:references|acknowledgments|\n\n)",
            "references": r"(?:references|bibliography)[:\s]*(.*?)(?:appendix|$)"
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                sections[section] = content[:2000]  # Limit section length
        
        result = {
            "sections": sections,
            "file_path": file_path,
            "extracted_at": datetime.now().isoformat(),
            "success": True
        }
        
        self._save_to_cache(cache_key, "sections", result)
        return result
    
    def extract_references(self, file_path: str) -> Dict[str, any]:
        """Extract bibliography/references from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted references
        """
        cache_key = self._get_cache_key(file_path)
        cached = self._get_cached_result(cache_key, "references")
        if cached:
            return cached
        
        sections_data = self.extract_sections(file_path)
        references_text = sections_data["sections"].get("references", "")
        
        references = []
        
        if references_text:
            # Split by common reference patterns
            ref_patterns = [
                r'\[\d+\]',  # [1], [2], etc.
                r'\d+\.',    # 1., 2., etc.
                r'•',        # Bullet points
            ]
            
            for pattern in ref_patterns:
                splits = re.split(pattern, references_text)
                if len(splits) > 2:  # Found references
                    for ref in splits[1:]:  # Skip first empty split
                        ref = ref.strip()
                        if len(ref) > 20:  # Filter out too short entries
                            references.append(ref[:500])  # Limit length
                    break
        
        result = {
            "references": references[:50],  # Limit to 50 references
            "count": len(references),
            "file_path": file_path,
            "extracted_at": datetime.now().isoformat(),
            "success": True
        }
        
        self._save_to_cache(cache_key, "references", result)
        return result