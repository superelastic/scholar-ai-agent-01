"""Result formatters for different output types.

This module provides formatters for presenting analysis results in various formats
including JSON, Markdown, HTML, and plain text.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class FormatterError(Exception):
    """Base exception for formatter errors."""
    pass


class BaseFormatter(ABC):
    """Base class for all result formatters."""
    
    @abstractmethod
    def format(self, data: Dict[str, Any]) -> str:
        """Format the analysis results.
        
        Args:
            data: Analysis results data
            
        Returns:
            Formatted string
        """
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this format.
        
        Returns:
            File extension (e.g., '.json', '.md')
        """
        pass


class JsonFormatter(BaseFormatter):
    """Formatter for JSON output."""
    
    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """Initialize JSON formatter.
        
        Args:
            indent: JSON indentation level
            sort_keys: Whether to sort keys
        """
        self.indent = indent
        self.sort_keys = sort_keys
    
    def format(self, data: Dict[str, Any]) -> str:
        """Format data as JSON.
        
        Args:
            data: Analysis results data
            
        Returns:
            JSON formatted string
        """
        try:
            # Create clean structure for JSON output
            json_data = {
                "analysis_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "format": "json",
                    "version": "1.0"
                },
                "paper_analysis": self._format_paper_analysis(data),
                "citations": self._format_citations(data),
                "research_directions": self._format_research_directions(data),
                "processing_info": self._format_processing_info(data)
            }
            
            return json.dumps(json_data, indent=self.indent, sort_keys=self.sort_keys, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"JSON formatting failed: {e}")
            raise FormatterError(f"Failed to format as JSON: {e}")
    
    def get_file_extension(self) -> str:
        """Get JSON file extension."""
        return '.json'
    
    def _format_paper_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format paper analysis section."""
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        
        paper_analysis = data.get('paper_analysis', {})
        if not isinstance(paper_analysis, dict):
            paper_analysis = {}
        
        return {
            "metadata": {
                "title": paper_metadata.get('title', ''),
                "authors": paper_metadata.get('authors', []),
                "year": paper_metadata.get('year'),
                "doi": paper_metadata.get('doi', ''),
                "abstract": paper_metadata.get('abstract', '')
            },
            "analysis": {
                "key_concepts": paper_analysis.get('key_concepts', []),
                "methodology": paper_analysis.get('methodology', ''),
                "findings": paper_analysis.get('findings', []),
                "contributions": paper_analysis.get('contributions', []),
                "limitations": paper_analysis.get('limitations', [])
            }
        }
    
    def _format_citations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format citations section."""
        citations = data.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        formatted_citations = data.get('formatted_citations', {})
        if not isinstance(formatted_citations, dict):
            formatted_citations = {}
        
        return {
            "summary": {
                "total_found": len(citations),
                "displayed": min(len(citations), 20)
            },
            "papers": [
                {
                    "title": cite.get('title', ''),
                    "authors": cite.get('authors', []),
                    "year": cite.get('year'),
                    "venue": cite.get('venue', ''),
                    "citations": cite.get('citations', 0),
                    "relevance_score": cite.get('relevance_score', 0.0),
                    "url": cite.get('url', ''),
                    "snippet": cite.get('snippet', '')
                }
                for cite in citations[:20]  # Top 20 citations
            ]
        }
    
    def _format_research_directions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format research directions section."""
        research_directions = data.get('research_directions', {})
        if not isinstance(research_directions, dict):
            research_directions = {}
        
        suggestions = research_directions.get('suggestions', [])
        if not isinstance(suggestions, list):
            suggestions = []
        
        return {
            "suggestions": [
                {
                    "title": suggestion.get('title', ''),
                    "description": suggestion.get('description', ''),
                    "rationale": suggestion.get('rationale', ''),
                    "confidence": suggestion.get('confidence', 0.0),
                    "difficulty": suggestion.get('difficulty', 'medium'),
                    "time_horizon": suggestion.get('time_horizon', 'medium-term'),
                    "required_expertise": suggestion.get('required_expertise', []),
                    "potential_impact": suggestion.get('potential_impact', 'moderate')
                }
                for suggestion in suggestions
            ],
            "summary": {
                "total_suggestions": len(suggestions),
                "avg_confidence": sum(s.get('confidence', 0) for s in suggestions) / max(len(suggestions), 1)
            }
        }
    
    def _format_processing_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format processing information."""
        processing_metadata = data.get('processing_metadata', {})
        if not isinstance(processing_metadata, dict):
            processing_metadata = {}
        
        return {
            "pdf_path": processing_metadata.get('pdf_path', ''),
            "processed_at": processing_metadata.get('processed_at', ''),
            "processing_time_seconds": processing_metadata.get('total_processing_time', 0),
            "status": "completed"
        }


class MarkdownFormatter(BaseFormatter):
    """Formatter for Markdown output."""
    
    def format(self, data: Dict[str, Any]) -> str:
        """Format data as Markdown.
        
        Args:
            data: Analysis results data
            
        Returns:
            Markdown formatted string
        """
        try:
            sections = []
            
            # Header
            sections.append(self._format_header(data))
            
            # Paper Analysis
            sections.append(self._format_paper_analysis_md(data))
            
            # Citations
            sections.append(self._format_citations_md(data))
            
            # Research Directions
            sections.append(self._format_research_directions_md(data))
            
            # Footer
            sections.append(self._format_footer(data))
            
            return '\n\n'.join(sections)
            
        except Exception as e:
            logger.error(f"Markdown formatting failed: {e}")
            raise FormatterError(f"Failed to format as Markdown: {e}")
    
    def get_file_extension(self) -> str:
        """Get Markdown file extension."""
        return '.md'
    
    def _format_header(self, data: Dict[str, Any]) -> str:
        """Format Markdown header."""
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        title = paper_metadata.get('title', 'Academic Paper Analysis')
        
        header = [
            f"# Academic Analysis Report",
            f"## {title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add paper metadata if available
        if paper_metadata.get('authors'):
            authors = ', '.join(paper_metadata['authors'])
            header.append(f"**Authors:** {authors}")
        
        if paper_metadata.get('year'):
            header.append(f"**Year:** {paper_metadata['year']}")
        
        if paper_metadata.get('doi'):
            header.append(f"**DOI:** {paper_metadata['doi']}")
        
        return '\n'.join(header)
    
    def _format_paper_analysis_md(self, data: Dict[str, Any]) -> str:
        """Format paper analysis as Markdown."""
        paper_analysis = data.get('paper_analysis', {})
        if not isinstance(paper_analysis, dict):
            paper_analysis = {}
        
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        
        sections = ["## Paper Analysis"]
        
        # Abstract
        if paper_metadata.get('abstract'):
            sections.extend([
                "### Abstract",
                paper_metadata['abstract']
            ])
        
        # Key Concepts
        key_concepts = paper_analysis.get('key_concepts', [])
        if key_concepts:
            sections.extend([
                "### Key Concepts",
                ""
            ])
            for concept in key_concepts:
                sections.append(f"- **{concept}**")
        
        # Methodology
        methodology = paper_analysis.get('methodology', '')
        if methodology:
            sections.extend([
                "",
                "### Methodology",
                methodology
            ])
        
        # Key Findings
        findings = paper_analysis.get('findings', [])
        if findings:
            sections.extend([
                "",
                "### Key Findings",
                ""
            ])
            for i, finding in enumerate(findings, 1):
                sections.append(f"{i}. {finding}")
        
        # Contributions
        contributions = paper_analysis.get('contributions', [])
        if contributions:
            sections.extend([
                "",
                "### Contributions",
                ""
            ])
            for contrib in contributions:
                sections.append(f"- {contrib}")
        
        return '\n'.join(sections)
    
    def _format_citations_md(self, data: Dict[str, Any]) -> str:
        """Format citations as Markdown."""
        citations = data.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        sections = [
            f"## Citations Analysis",
            f"",
            f"Found **{len(citations)}** papers that cite this work."
        ]
        
        if not citations:
            sections.append("\nNo citations were found or accessible.")
            return '\n'.join(sections)
        
        sections.extend([
            "",
            "### Recent Citing Papers",
            ""
        ])
        
        # Show top 10 citations
        for i, citation in enumerate(citations[:10], 1):
            title = citation.get('title', 'Untitled')
            authors = citation.get('authors', [])
            year = citation.get('year', 'Unknown')
            venue = citation.get('venue', '')
            relevance = citation.get('relevance_score', 0.0)
            url = citation.get('url', '')
            
            sections.append(f"#### {i}. {title}")
            
            if authors:
                authors_str = ', '.join(authors[:3])  # Show first 3 authors
                if len(authors) > 3:
                    authors_str += f" et al."
                sections.append(f"**Authors:** {authors_str}")
            
            sections.append(f"**Year:** {year}")
            
            if venue:
                sections.append(f"**Venue:** {venue}")
            
            sections.append(f"**Relevance Score:** {relevance:.2f}/1.0")
            
            if url:
                sections.append(f"**Link:** [View Paper]({url})")
            
            # Add snippet if available
            snippet = citation.get('snippet', '')
            if snippet:
                sections.extend([
                    "",
                    f"*{snippet}*"
                ])
            
            sections.append("")
        
        return '\n'.join(sections)
    
    def _format_research_directions_md(self, data: Dict[str, Any]) -> str:
        """Format research directions as Markdown."""
        research_directions = data.get('research_directions', {})
        if not isinstance(research_directions, dict):
            research_directions = {}
        
        suggestions = research_directions.get('suggestions', [])
        if not isinstance(suggestions, list):
            suggestions = []
        
        sections = [
            "## Future Research Directions",
            "",
            f"Based on the analysis of the paper and its citations, here are **{len(suggestions)}** potential research directions:"
        ]
        
        if not suggestions:
            sections.append("\nNo research directions could be generated.")
            return '\n'.join(sections)
        
        sections.append("")
        
        for i, suggestion in enumerate(suggestions, 1):
            title = suggestion.get('title', f'Research Direction {i}')
            description = suggestion.get('description', '')
            rationale = suggestion.get('rationale', '')
            confidence = suggestion.get('confidence', 0.0)
            difficulty = suggestion.get('difficulty', 'medium')
            time_horizon = suggestion.get('time_horizon', 'medium-term')
            impact = suggestion.get('potential_impact', 'moderate')
            
            sections.extend([
                f"### {i}. {title}",
                "",
                f"**Confidence Score:** {confidence:.2f}/1.0",
                f"**Difficulty:** {difficulty.title()}",
                f"**Time Horizon:** {time_horizon.replace('_', ' ').title()}",
                f"**Potential Impact:** {impact.title()}",
                ""
            ])
            
            if description:
                sections.extend([
                    "**Description:**",
                    description,
                    ""
                ])
            
            if rationale:
                sections.extend([
                    "**Rationale:**",
                    rationale,
                    ""
                ])
            
            # Add required expertise if available
            expertise = suggestion.get('required_expertise', [])
            if expertise:
                expertise_str = ', '.join(expertise)
                sections.extend([
                    f"**Required Expertise:** {expertise_str}",
                    ""
                ])
        
        return '\n'.join(sections)
    
    def _format_footer(self, data: Dict[str, Any]) -> str:
        """Format Markdown footer."""
        processing_metadata = data.get('processing_metadata', {})
        if not isinstance(processing_metadata, dict):
            processing_metadata = {}
        processing_time = processing_metadata.get('total_processing_time', 0)
        
        footer = [
            "---",
            "",
            "## Analysis Information",
            "",
            f"**Analysis completed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if processing_time:
            footer.append(f"**Total processing time:** {processing_time:.1f} seconds")
        
        footer.extend([
            "",
            "*This analysis was generated by the Scholar AI Agent system.*"
        ])
        
        return '\n'.join(footer)


class HtmlFormatter(BaseFormatter):
    """Formatter for HTML output."""
    
    def format(self, data: Dict[str, Any]) -> str:
        """Format data as HTML.
        
        Args:
            data: Analysis results data
            
        Returns:
            HTML formatted string
        """
        try:
            html_parts = [
                self._get_html_header(data),
                self._format_paper_analysis_html(data),
                self._format_citations_html(data),
                self._format_research_directions_html(data),
                self._get_html_footer(data)
            ]
            
            return '\n'.join(html_parts)
            
        except Exception as e:
            logger.error(f"HTML formatting failed: {e}")
            raise FormatterError(f"Failed to format as HTML: {e}")
    
    def get_file_extension(self) -> str:
        """Get HTML file extension."""
        return '.html'
    
    def _get_html_header(self, data: Dict[str, Any]) -> str:
        """Generate HTML header."""
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        title = paper_metadata.get('title', 'Academic Paper Analysis')
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: #f8f9fa; padding: 25px; margin: 20px 0; border-radius: 8px; 
                    border-left: 4px solid #667eea; }}
        .citation {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; 
                     border: 1px solid #e9ecef; }}
        .suggestion {{ background: white; padding: 20px; margin: 15px 0; border-radius: 5px; 
                       border: 1px solid #e9ecef; border-left: 4px solid #28a745; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; 
                  font-weight: bold; }}
        .confidence-high {{ background: #d4edda; color: #155724; }}
        .confidence-medium {{ background: #fff3cd; color: #856404; }}
        .confidence-low {{ background: #f8d7da; color: #721c24; }}
        .metadata {{ color: #6c757d; font-size: 0.9em; }}
        h1, h2, h3 {{ color: #495057; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; 
                   color: #6c757d; border-top: 1px solid #e9ecef; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Academic Analysis Report</h1>
        <h2>{self._escape_html(title)}</h2>
        <div class="metadata">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</div>
    </div>"""
    
    def _format_paper_analysis_html(self, data: Dict[str, Any]) -> str:
        """Format paper analysis as HTML."""
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        
        paper_analysis = data.get('paper_analysis', {})
        if not isinstance(paper_analysis, dict):
            paper_analysis = {}
        
        html = ['<div class="section">', '<h2>Paper Analysis</h2>']
        
        # Paper metadata
        if paper_metadata.get('authors'):
            authors = ', '.join(paper_metadata['authors'])
            html.append(f'<p><strong>Authors:</strong> {self._escape_html(authors)}</p>')
        
        if paper_metadata.get('year'):
            html.append(f'<p><strong>Year:</strong> {paper_metadata["year"]}</p>')
        
        if paper_metadata.get('doi'):
            html.append(f'<p><strong>DOI:</strong> {self._escape_html(paper_metadata["doi"])}</p>')
        
        # Abstract
        if paper_metadata.get('abstract'):
            html.extend([
                '<h3>Abstract</h3>',
                f'<p>{self._escape_html(paper_metadata["abstract"])}</p>'
            ])
        
        # Key concepts
        key_concepts = paper_analysis.get('key_concepts', [])
        if key_concepts:
            html.append('<h3>Key Concepts</h3>')
            html.append('<ul>')
            for concept in key_concepts:
                html.append(f'<li><strong>{self._escape_html(concept)}</strong></li>')
            html.append('</ul>')
        
        # Methodology
        methodology = paper_analysis.get('methodology', '')
        if methodology:
            html.extend([
                '<h3>Methodology</h3>',
                f'<p>{self._escape_html(methodology)}</p>'
            ])
        
        # Findings
        findings = paper_analysis.get('findings', [])
        if findings:
            html.append('<h3>Key Findings</h3>')
            html.append('<ol>')
            for finding in findings:
                html.append(f'<li>{self._escape_html(finding)}</li>')
            html.append('</ol>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _format_citations_html(self, data: Dict[str, Any]) -> str:
        """Format citations as HTML."""
        citations = data.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        html = [
            '<div class="section">',
            '<h2>Citations Analysis</h2>',
            f'<p>Found <strong>{len(citations)}</strong> papers that cite this work.</p>'
        ]
        
        if citations:
            html.append('<h3>Recent Citing Papers</h3>')
            
            for i, citation in enumerate(citations[:10], 1):
                title = citation.get('title', 'Untitled')
                authors = citation.get('authors', [])
                year = citation.get('year', 'Unknown')
                venue = citation.get('venue', '')
                relevance = citation.get('relevance_score', 0.0)
                url = citation.get('url', '')
                snippet = citation.get('snippet', '')
                
                # Determine confidence badge
                if relevance >= 0.7:
                    badge_class = 'confidence-high'
                elif relevance >= 0.4:
                    badge_class = 'confidence-medium'
                else:
                    badge_class = 'confidence-low'
                
                html.append('<div class="citation">')
                html.append(f'<h4>{i}. {self._escape_html(title)}</h4>')
                
                if authors:
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += ' et al.'
                    html.append(f'<p><strong>Authors:</strong> {self._escape_html(authors_str)}</p>')
                
                html.append(f'<p><strong>Year:</strong> {year}</p>')
                
                if venue:
                    html.append(f'<p><strong>Venue:</strong> {self._escape_html(venue)}</p>')
                
                html.append(f'<p><strong>Relevance:</strong> <span class="badge {badge_class}">{relevance:.2f}</span></p>')
                
                if url:
                    html.append(f'<p><a href="{url}" target="_blank">View Paper</a></p>')
                
                if snippet:
                    html.append(f'<p><em>{self._escape_html(snippet)}</em></p>')
                
                html.append('</div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _format_research_directions_html(self, data: Dict[str, Any]) -> str:
        """Format research directions as HTML."""
        research_directions = data.get('research_directions', {})
        if not isinstance(research_directions, dict):
            research_directions = {}
        
        suggestions = research_directions.get('suggestions', [])
        if not isinstance(suggestions, list):
            suggestions = []
        
        html = [
            '<div class="section">',
            '<h2>Future Research Directions</h2>',
            f'<p>Based on the analysis, here are <strong>{len(suggestions)}</strong> potential research directions:</p>'
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            title = suggestion.get('title', f'Research Direction {i}')
            description = suggestion.get('description', '')
            rationale = suggestion.get('rationale', '')
            confidence = suggestion.get('confidence', 0.0)
            difficulty = suggestion.get('difficulty', 'medium')
            time_horizon = suggestion.get('time_horizon', 'medium-term')
            impact = suggestion.get('potential_impact', 'moderate')
            
            # Confidence badge
            if confidence >= 0.7:
                badge_class = 'confidence-high'
            elif confidence >= 0.4:
                badge_class = 'confidence-medium'
            else:
                badge_class = 'confidence-low'
            
            html.append('<div class="suggestion">')
            html.append(f'<h3>{i}. {self._escape_html(title)}</h3>')
            html.append(f'<p><strong>Confidence:</strong> <span class="badge {badge_class}">{confidence:.2f}</span></p>')
            html.append(f'<p><strong>Difficulty:</strong> {difficulty.title()}</p>')
            html.append(f'<p><strong>Time Horizon:</strong> {time_horizon.replace("_", " ").title()}</p>')
            html.append(f'<p><strong>Potential Impact:</strong> {impact.title()}</p>')
            
            if description:
                html.append(f'<p><strong>Description:</strong> {self._escape_html(description)}</p>')
            
            if rationale:
                html.append(f'<p><strong>Rationale:</strong> {self._escape_html(rationale)}</p>')
            
            expertise = suggestion.get('required_expertise', [])
            if expertise:
                expertise_str = ', '.join(expertise)
                html.append(f'<p><strong>Required Expertise:</strong> {self._escape_html(expertise_str)}</p>')
            
            html.append('</div>')
        
        html.append('</div>')
        return '\n'.join(html)
    
    def _get_html_footer(self, data: Dict[str, Any]) -> str:
        """Generate HTML footer."""
        processing_metadata = data.get('processing_metadata', {})
        if not isinstance(processing_metadata, dict):
            processing_metadata = {}
        processing_time = processing_metadata.get('total_processing_time', 0)
        
        footer = ['<div class="footer">']
        
        if processing_time:
            footer.append(f'<p>Analysis completed in {processing_time:.1f} seconds</p>')
        
        footer.extend([
            '<p><em>This analysis was generated by the Scholar AI Agent system.</em></p>',
            '</div>',
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(footer)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text


class PlainTextFormatter(BaseFormatter):
    """Formatter for plain text output."""
    
    def format(self, data: Dict[str, Any]) -> str:
        """Format data as plain text.
        
        Args:
            data: Analysis results data
            
        Returns:
            Plain text formatted string
        """
        try:
            sections = []
            
            # Header
            sections.append(self._format_header_txt(data))
            
            # Paper Analysis
            sections.append(self._format_paper_analysis_txt(data))
            
            # Citations
            sections.append(self._format_citations_txt(data))
            
            # Research Directions
            sections.append(self._format_research_directions_txt(data))
            
            # Footer
            sections.append(self._format_footer_txt(data))
            
            return '\n\n'.join(sections)
            
        except Exception as e:
            logger.error(f"Plain text formatting failed: {e}")
            raise FormatterError(f"Failed to format as plain text: {e}")
    
    def get_file_extension(self) -> str:
        """Get plain text file extension."""
        return '.txt'
    
    def _format_header_txt(self, data: Dict[str, Any]) -> str:
        """Format plain text header."""
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        title = paper_metadata.get('title', 'Academic Paper Analysis')
        
        header = [
            "=" * 80,
            "ACADEMIC ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Paper: {title}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if paper_metadata.get('authors'):
            authors = ', '.join(paper_metadata['authors'])
            header.append(f"Authors: {authors}")
        
        if paper_metadata.get('year'):
            header.append(f"Year: {paper_metadata['year']}")
        
        return '\n'.join(header)
    
    def _format_paper_analysis_txt(self, data: Dict[str, Any]) -> str:
        """Format paper analysis as plain text."""
        paper_analysis = data.get('paper_analysis', {})
        if not isinstance(paper_analysis, dict):
            paper_analysis = {}
        
        paper_metadata = data.get('paper_metadata', {})
        if not isinstance(paper_metadata, dict):
            paper_metadata = {}
        
        sections = [
            "-" * 80,
            "PAPER ANALYSIS",
            "-" * 80
        ]
        
        # Abstract
        if paper_metadata.get('abstract'):
            sections.extend([
                "",
                "ABSTRACT:",
                paper_metadata['abstract']
            ])
        
        # Key Concepts
        key_concepts = paper_analysis.get('key_concepts', [])
        if key_concepts:
            sections.extend([
                "",
                "KEY CONCEPTS:",
            ])
            for concept in key_concepts:
                sections.append(f"  • {concept}")
        
        # Methodology
        methodology = paper_analysis.get('methodology', '')
        if methodology:
            sections.extend([
                "",
                "METHODOLOGY:",
                methodology
            ])
        
        # Findings
        findings = paper_analysis.get('findings', [])
        if findings:
            sections.extend([
                "",
                "KEY FINDINGS:",
            ])
            for i, finding in enumerate(findings, 1):
                sections.append(f"  {i}. {finding}")
        
        return '\n'.join(sections)
    
    def _format_citations_txt(self, data: Dict[str, Any]) -> str:
        """Format citations as plain text."""
        citations = data.get('citations', [])
        if not isinstance(citations, list):
            citations = []
        
        sections = [
            "-" * 80,
            "CITATIONS ANALYSIS",
            "-" * 80,
            "",
            f"Found {len(citations)} papers that cite this work."
        ]
        
        if citations:
            sections.extend([
                "",
                "RECENT CITING PAPERS:",
                ""
            ])
            
            for i, citation in enumerate(citations[:10], 1):
                title = citation.get('title', 'Untitled')
                authors = citation.get('authors', [])
                year = citation.get('year', 'Unknown')
                relevance = citation.get('relevance_score', 0.0)
                
                sections.append(f"{i}. {title}")
                
                if authors:
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += ' et al.'
                    sections.append(f"   Authors: {authors_str}")
                
                sections.append(f"   Year: {year}")
                sections.append(f"   Relevance: {relevance:.2f}/1.0")
                sections.append("")
        
        return '\n'.join(sections)
    
    def _format_research_directions_txt(self, data: Dict[str, Any]) -> str:
        """Format research directions as plain text."""
        research_directions = data.get('research_directions', {})
        if not isinstance(research_directions, dict):
            research_directions = {}
        
        suggestions = research_directions.get('suggestions', [])
        if not isinstance(suggestions, list):
            suggestions = []
        
        sections = [
            "-" * 80,
            "FUTURE RESEARCH DIRECTIONS",
            "-" * 80,
            "",
            f"Based on the analysis, here are {len(suggestions)} potential research directions:"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            title = suggestion.get('title', f'Research Direction {i}')
            description = suggestion.get('description', '')
            confidence = suggestion.get('confidence', 0.0)
            difficulty = suggestion.get('difficulty', 'medium')
            
            sections.extend([
                "",
                f"{i}. {title}",
                f"   Confidence: {confidence:.2f}/1.0",
                f"   Difficulty: {difficulty.title()}"
            ])
            
            if description:
                # Wrap long descriptions
                wrapped_desc = self._wrap_text(description, width=75, indent="   ")
                sections.extend([
                    "   Description:",
                    wrapped_desc
                ])
        
        return '\n'.join(sections)
    
    def _format_footer_txt(self, data: Dict[str, Any]) -> str:
        """Format plain text footer."""
        processing_metadata = data.get('processing_metadata', {})
        if not isinstance(processing_metadata, dict):
            processing_metadata = {}
        processing_time = processing_metadata.get('total_processing_time', 0)
        
        footer = [
            "-" * 80,
            "ANALYSIS INFORMATION",
            "-" * 80,
            "",
            f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        if processing_time:
            footer.append(f"Total processing time: {processing_time:.1f} seconds")
        
        footer.extend([
            "",
            "This analysis was generated by the Scholar AI Agent system."
        ])
        
        return '\n'.join(footer)
    
    def _wrap_text(self, text: str, width: int = 80, indent: str = "") -> str:
        """Wrap text to specified width with indentation."""
        words = text.split()
        lines = []
        current_line = indent
        
        for word in words:
            if len(current_line + word) <= width:
                current_line += word + " "
            else:
                lines.append(current_line.rstrip())
                current_line = indent + word + " "
        
        if current_line.strip():
            lines.append(current_line.rstrip())
        
        return '\n'.join(lines)


class FormatterFactory:
    """Factory for creating formatters."""
    
    _formatters = {
        'json': JsonFormatter,
        'markdown': MarkdownFormatter,
        'md': MarkdownFormatter,
        'html': HtmlFormatter,
        'txt': PlainTextFormatter,
        'text': PlainTextFormatter
    }
    
    @classmethod
    def create_formatter(cls, format_type: str) -> BaseFormatter:
        """Create a formatter for the specified format.
        
        Args:
            format_type: Format type (json, markdown, html, txt)
            
        Returns:
            Formatter instance
            
        Raises:
            FormatterError: If format type is not supported
        """
        format_type = format_type.lower()
        
        if format_type not in cls._formatters:
            available = ', '.join(cls._formatters.keys())
            raise FormatterError(f"Unsupported format '{format_type}'. Available: {available}")
        
        return cls._formatters[format_type]()
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported format types.
        
        Returns:
            List of supported format types
        """
        return list(cls._formatters.keys())
    
    @classmethod
    def format_data(cls, data: Dict[str, Any], format_type: str) -> str:
        """Format data using the specified formatter.
        
        Args:
            data: Data to format
            format_type: Format type
            
        Returns:
            Formatted string
        """
        formatter = cls.create_formatter(format_type)
        return formatter.format(data)