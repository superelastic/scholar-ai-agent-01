"""Export manager for saving analysis results to files.

This module provides functionality to export analysis results in various formats
with proper file management and error handling.
"""

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from utils.formatters import FormatterFactory, FormatterError

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Base exception for export operations."""
    pass


class ExportManager:
    """Manages exporting analysis results to various formats and locations."""
    
    def __init__(self, base_export_dir: str = "./exports"):
        """Initialize the export manager.
        
        Args:
            base_export_dir: Base directory for exports
        """
        self.base_export_dir = Path(base_export_dir)
        self.base_export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ExportManager with base directory: {self.base_export_dir}")
    
    def export_analysis_results(
        self, 
        analysis_data: Dict[str, Any],
        formats: List[str],
        export_name: Optional[str] = None,
        create_bundle: bool = True
    ) -> Dict[str, Any]:
        """Export analysis results in multiple formats.
        
        Args:
            analysis_data: Analysis results data to export
            formats: List of formats to export ('json', 'markdown', 'html', 'txt')
            export_name: Optional custom name for export (defaults to paper title)
            create_bundle: Whether to create a ZIP bundle of all exports
            
        Returns:
            Export information including file paths and metadata
        """
        try:
            # Generate export session info
            export_session = self._create_export_session(analysis_data, export_name)
            session_dir = export_session['directory']
            
            # Export in each requested format
            export_files = {}
            successful_exports = []
            failed_exports = []
            
            for format_type in formats:
                try:
                    file_path = self._export_single_format(
                        analysis_data, 
                        format_type, 
                        session_dir,
                        export_session['base_filename']
                    )
                    export_files[format_type] = str(file_path)
                    successful_exports.append(format_type)
                    logger.info(f"Successfully exported {format_type.upper()} to {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to export {format_type}: {e}")
                    failed_exports.append({'format': format_type, 'error': str(e)})
            
            # Create metadata file
            metadata_file = self._create_metadata_file(export_session, analysis_data, export_files)
            
            # Create bundle if requested
            bundle_file = None
            if create_bundle and successful_exports:
                bundle_file = self._create_export_bundle(
                    session_dir, 
                    export_session['base_filename'],
                    list(export_files.values()) + [str(metadata_file)]
                )
            
            export_info = {
                'success': len(formats) == 0 or bool(successful_exports),  # Success if no formats requested or some succeeded
                'export_session': export_session,
                'exported_formats': successful_exports,
                'failed_formats': failed_exports,
                'files': export_files,
                'metadata_file': str(metadata_file),
                'bundle_file': str(bundle_file) if bundle_file else None,
                'summary': {
                    'total_formats_requested': len(formats),
                    'successful_exports': len(successful_exports),
                    'failed_exports': len(failed_exports),
                    'export_directory': str(session_dir)
                }
            }
            
            logger.info(f"Export completed: {len(successful_exports)}/{len(formats)} formats successful")
            return export_info
            
        except Exception as e:
            error_msg = f"Export operation failed: {e}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def export_single_file(
        self,
        analysis_data: Dict[str, Any],
        format_type: str,
        file_path: Optional[str] = None
    ) -> str:
        """Export analysis results to a single file.
        
        Args:
            analysis_data: Analysis results data
            format_type: Format to export
            file_path: Optional custom file path
            
        Returns:
            Path to exported file
        """
        try:
            if not file_path:
                # Generate default file path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                paper_title = analysis_data.get('paper_metadata', {}).get('title', 'analysis')
                safe_title = self._make_safe_filename(paper_title)
                
                formatter = FormatterFactory.create_formatter(format_type)
                extension = formatter.get_file_extension()
                
                filename = f"scholar_ai_{safe_title}_{timestamp}{extension}"
                file_path = self.base_export_dir / filename
            else:
                file_path = Path(file_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export the file
            actual_path = self._export_single_format(
                analysis_data,
                format_type,
                file_path.parent,
                file_path.stem
            )
            
            logger.info(f"Exported single file: {actual_path}")
            return str(actual_path)
            
        except Exception as e:
            error_msg = f"Single file export failed: {e}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def list_exports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent exports.
        
        Args:
            limit: Maximum number of exports to return
            
        Returns:
            List of export information
        """
        try:
            exports = []
            
            # Scan export directory for metadata files
            if self.base_export_dir.exists():
                for metadata_file in self.base_export_dir.glob("**/export_metadata.json"):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        export_info = {
                            'export_name': metadata.get('export_name'),
                            'paper_title': metadata.get('paper_metadata', {}).get('title'),
                            'created_at': metadata.get('created_at'),
                            'formats': metadata.get('formats', []),
                            'directory': str(metadata_file.parent),
                            'bundle_file': metadata.get('bundle_file')
                        }
                        exports.append(export_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read export metadata from {metadata_file}: {e}")
                
                # Sort by creation time (newest first)
                exports.sort(key=lambda x: x.get('created_at') or '', reverse=True)
            
            return exports[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list exports: {e}")
            return []
    
    def cleanup_old_exports(self, keep_days: int = 30) -> Dict[str, Any]:
        """Clean up old export files.
        
        Args:
            keep_days: Number of days to keep exports
            
        Returns:
            Cleanup summary
        """
        try:
            cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
            
            deleted_dirs = []
            deleted_files = []
            errors = []
            
            if self.base_export_dir.exists():
                for item in self.base_export_dir.iterdir():
                    try:
                        if item.stat().st_mtime < cutoff_date:
                            if item.is_dir():
                                shutil.rmtree(item)
                                deleted_dirs.append(str(item))
                                logger.info(f"Deleted old export directory: {item}")
                            else:
                                item.unlink()
                                deleted_files.append(str(item))
                                logger.info(f"Deleted old export file: {item}")
                    except Exception as e:
                        errors.append({'path': str(item), 'error': str(e)})
                        logger.warning(f"Failed to delete {item}: {e}")
            
            summary = {
                'deleted_directories': len(deleted_dirs),
                'deleted_files': len(deleted_files),
                'errors': len(errors),
                'cutoff_days': keep_days,
                'details': {
                    'deleted_dirs': deleted_dirs,
                    'deleted_files': deleted_files,
                    'errors': errors
                }
            }
            
            logger.info(f"Cleanup completed: {len(deleted_dirs)} dirs, {len(deleted_files)} files deleted")
            return summary
            
        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def _create_export_session(self, analysis_data: Dict[str, Any], export_name: Optional[str]) -> Dict[str, Any]:
        """Create a new export session with directory and metadata.
        
        Args:
            analysis_data: Analysis results data
            export_name: Optional custom export name
            
        Returns:
            Export session information
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not export_name:
            paper_metadata = analysis_data.get('paper_metadata', {})
            if isinstance(paper_metadata, dict):
                paper_title = paper_metadata.get('title', 'analysis')
            else:
                paper_title = 'analysis'
            export_name = self._make_safe_filename(paper_title)
        else:
            export_name = self._make_safe_filename(export_name)
        
        session_name = f"{export_name}_{timestamp}"
        session_dir = self.base_export_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'session_name': session_name,
            'export_name': export_name,
            'timestamp': timestamp,
            'directory': session_dir,
            'base_filename': export_name,
            'created_at': datetime.now().isoformat()
        }
    
    def _export_single_format(
        self,
        analysis_data: Dict[str, Any],
        format_type: str,
        output_dir: Path,
        base_filename: str
    ) -> Path:
        """Export data in a single format.
        
        Args:
            analysis_data: Data to export
            format_type: Format type
            output_dir: Output directory
            base_filename: Base filename (without extension)
            
        Returns:
            Path to exported file
        """
        try:
            # Create formatter and get content
            formatter = FormatterFactory.create_formatter(format_type)
            formatted_content = formatter.format(analysis_data)
            
            # Generate filename
            extension = formatter.get_file_extension()
            filename = f"{base_filename}{extension}"
            file_path = output_dir / filename
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            return file_path
            
        except FormatterError as e:
            raise ExportError(f"Formatting failed for {format_type}: {e}")
        except IOError as e:
            raise ExportError(f"File write failed for {format_type}: {e}")
    
    def _create_metadata_file(
        self,
        export_session: Dict[str, Any],
        analysis_data: Dict[str, Any],
        export_files: Dict[str, str]
    ) -> Path:
        """Create metadata file for the export session.
        
        Args:
            export_session: Export session info
            analysis_data: Original analysis data
            export_files: Dictionary of exported files by format
            
        Returns:
            Path to metadata file
        """
        # Safely extract data with type checking
        paper_metadata = analysis_data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        
        paper_analysis = analysis_data.get('paper_analysis', {})
        if not isinstance(paper_analysis, dict):
            paper_analysis = {}
        
        citations = analysis_data.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        research_directions = analysis_data.get('research_directions', {})
        if not isinstance(research_directions, dict):
            research_directions = {}
        
        processing_metadata = analysis_data.get('processing_metadata', {})
        if not isinstance(processing_metadata, dict):
            processing_metadata = {}
        
        metadata = {
            'export_metadata': {
                'export_name': export_session['export_name'],
                'created_at': export_session['created_at'],
                'formats': list(export_files.keys()),
                'files': export_files
            },
            'paper_metadata': paper_metadata,
            'analysis_summary': {
                'key_concepts_count': len(paper_analysis.get('key_concepts', [])),
                'citations_count': len(citations),
                'research_directions_count': len(research_directions.get('suggestions', [])),
                'processing_time': processing_metadata.get('total_processing_time')
            },
            'export_info': {
                'total_files': len(export_files),
                'export_directory': str(export_session['directory']),
                'created_by': 'Scholar AI Agent System'
            }
        }
        
        metadata_file = export_session['directory'] / 'export_metadata.json'
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_file
    
    def _create_export_bundle(
        self,
        session_dir: Path,
        base_filename: str,
        file_paths: List[str]
    ) -> Optional[Path]:
        """Create a ZIP bundle of exported files.
        
        Args:
            session_dir: Session directory
            base_filename: Base filename for bundle
            file_paths: List of file paths to include
            
        Returns:
            Path to bundle file or None if failed
        """
        try:
            bundle_filename = f"{base_filename}_export_bundle.zip"
            bundle_path = session_dir / bundle_filename
            
            with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in file_paths:
                    file_path = Path(file_path)
                    if file_path.exists():
                        # Use just the filename in the archive
                        archive_name = file_path.name
                        zf.write(file_path, archive_name)
            
            logger.info(f"Created export bundle: {bundle_path}")
            return bundle_path
            
        except Exception as e:
            logger.error(f"Failed to create export bundle: {e}")
            return None
    
    def _make_safe_filename(self, filename: str, max_length: int = 50) -> str:
        """Convert a string to a safe filename.
        
        Args:
            filename: Original filename
            max_length: Maximum filename length
            
        Returns:
            Safe filename string
        """
        if not filename:
            return "untitled"
        
        # Remove/replace problematic characters
        safe_chars = []
        for char in filename.lower():
            if char.isalnum():
                safe_chars.append(char)
            elif char in ' -_':
                safe_chars.append('_')
        
        safe_filename = ''.join(safe_chars)
        
        # Remove multiple underscores and trim
        safe_filename = '_'.join(filter(None, safe_filename.split('_')))
        
        # Limit length
        if len(safe_filename) > max_length:
            safe_filename = safe_filename[:max_length].rstrip('_')
        
        return safe_filename or "untitled"


def export_analysis_results(
    analysis_data: Dict[str, Any],
    formats: List[str] = None,
    output_dir: str = "./exports",
    export_name: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to export analysis results.
    
    Args:
        analysis_data: Analysis results to export
        formats: List of formats to export (defaults to all)
        output_dir: Output directory
        export_name: Optional custom export name
        
    Returns:
        Export information
    """
    if formats is None:
        formats = ['json', 'markdown', 'html', 'txt']
    
    export_manager = ExportManager(output_dir)
    return export_manager.export_analysis_results(
        analysis_data, 
        formats, 
        export_name,
        create_bundle=True
    )