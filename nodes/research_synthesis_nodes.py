"""Research Synthesis Agent nodes for academic research direction generation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from utils import LLMAnalysisUtility

logger = logging.getLogger(__name__)


class ResearchSynthesisNodeError(Exception):
    """Custom exception for Research Synthesis node errors."""
    pass


class AsyncNode:
    """Base class for async nodes."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the node.
        
        Args:
            context: Execution context containing store and input data
            
        Returns:
            Processing result with success status
        """
        raise NotImplementedError


class PaperSynthesisNode(AsyncNode):
    """Node for synthesizing seminal paper and citation data."""

    def __init__(self, llm_utility: LLMAnalysisUtility):
        """Initialize the paper synthesis node.
        
        Args:
            llm_utility: LLM utility for analysis
        """
        self.llm_utility = llm_utility

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize paper analysis and citation data.
        
        Args:
            context: Execution context
            
        Returns:
            Synthesized analysis and success status
        """
        store = context['store']
        input_data = context['input_data']
        
        try:
            # Extract input data
            paper_metadata = input_data.get('paper_metadata', {})
            paper_analysis = input_data.get('paper_analysis', {})
            citation_data = input_data.get('citation_data', [])
            
            if not paper_metadata and not paper_analysis:
                raise ResearchSynthesisNodeError("No paper metadata or analysis provided")
            
            store['paper_metadata'] = paper_metadata
            store['paper_analysis'] = paper_analysis
            store['citation_data'] = citation_data
            store['status'] = 'synthesizing_data'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Synthesizing data for paper: {paper_metadata.get('title', 'Unknown')}")
            logger.info(f"Processing {len(citation_data)} citations")
            
            # Create comprehensive synthesis of the paper and its citations
            synthesis_data = self._create_synthesis_overview(
                paper_metadata, paper_analysis, citation_data
            )
            
            # Use LLM to enhance the synthesis with deeper insights
            try:
                enhanced_synthesis = await self._enhance_with_llm_analysis(
                    synthesis_data, paper_metadata, citation_data
                )
                synthesis_data.update(enhanced_synthesis)
                logger.info("Enhanced synthesis with LLM analysis")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}, using basic synthesis")
            
            store['comprehensive_synthesis'] = synthesis_data
            store['status'] = 'synthesis_complete'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'synthesis': synthesis_data
            }
            
        except Exception as e:
            error_msg = f"Paper synthesis failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'synthesis_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _create_synthesis_overview(self, paper_metadata: Dict[str, Any], 
                                 paper_analysis: Dict[str, Any], 
                                 citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create basic synthesis overview from available data.
        
        Args:
            paper_metadata: Original paper metadata
            paper_analysis: Analysis results from coordinator
            citation_data: Citation information
            
        Returns:
            Comprehensive synthesis data
        """
        synthesis = {
            'seminal_paper': {
                'title': paper_metadata.get('title', ''),
                'year': paper_metadata.get('year'),
                'authors': paper_metadata.get('authors', []),
                'key_concepts': paper_analysis.get('key_concepts', []),
                'methodology': paper_analysis.get('methodology', ''),
                'main_findings': paper_analysis.get('findings', []),
                'significance': paper_analysis.get('significance', '')
            },
            'citation_landscape': {
                'total_citations': len(citation_data),
                'citation_years': self._extract_citation_years(citation_data),
                'citing_venues': self._extract_venues(citation_data),
                'citation_themes': self._extract_citation_themes(citation_data),
                'high_impact_citations': self._identify_high_impact_citations(citation_data)
            },
            'research_evolution': {
                'temporal_trends': self._analyze_temporal_trends(citation_data),
                'methodological_evolution': self._analyze_methodology_evolution(citation_data),
                'application_domains': self._extract_application_domains(citation_data)
            },
            'synthesis_metadata': {
                'synthesized_at': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(paper_metadata, paper_analysis, citation_data)
            }
        }
        
        return synthesis

    def _extract_citation_years(self, citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract year distribution from citations."""
        years = [c.get('year') for c in citation_data if c.get('year')]
        if not years:
            return {'min_year': None, 'max_year': None, 'year_distribution': {}}
        
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            'min_year': min(years),
            'max_year': max(years),
            'year_distribution': year_counts,
            'citation_growth': self._calculate_citation_growth(year_counts)
        }

    def _extract_venues(self, citation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract venue information from citations."""
        venue_counts = {}
        for citation in citation_data:
            venue = citation.get('venue', '').strip()
            if venue:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        # Return top venues
        sorted_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'venue': venue, 'count': count} for venue, count in sorted_venues[:10]]

    def _extract_citation_themes(self, citation_data: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from citation titles and abstracts."""
        themes = []
        common_terms = {}
        
        for citation in citation_data:
            title = citation.get('title', '').lower()
            abstract = citation.get('abstract', '').lower()
            
            # Extract key terms (simple approach)
            text = f"{title} {abstract}"
            words = re.findall(r'\b[a-z]{4,}\b', text)  # Words with 4+ characters
            
            for word in words:
                if word not in ['paper', 'study', 'research', 'method', 'approach', 'analysis']:
                    common_terms[word] = common_terms.get(word, 0) + 1
        
        # Get most common themes
        sorted_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)
        themes = [term for term, count in sorted_terms[:15] if count >= 2]
        
        return themes

    def _identify_high_impact_citations(self, citation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify high-impact citations based on citation count and relevance."""
        high_impact = []
        
        for citation in citation_data:
            citation_count = citation.get('citation_count', 0)
            relevance_score = citation.get('relevance_score', 0)
            
            # High impact criteria: high citations OR high relevance
            if citation_count >= 100 or relevance_score >= 0.8:
                high_impact.append({
                    'title': citation.get('title', ''),
                    'year': citation.get('year'),
                    'citation_count': citation_count,
                    'relevance_score': relevance_score,
                    'venue': citation.get('venue', '')
                })
        
        return sorted(high_impact, key=lambda x: x['citation_count'], reverse=True)[:5]

    def _analyze_temporal_trends(self, citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how research has evolved over time."""
        if not citation_data:
            return {'trend_analysis': 'Insufficient data for temporal analysis'}
        
        # Group citations by time periods
        early_period = []
        recent_period = []
        current_year = datetime.now().year
        
        for citation in citation_data:
            year = citation.get('year')
            if year:
                if year <= current_year - 3:
                    early_period.append(citation)
                else:
                    recent_period.append(citation)
        
        return {
            'early_period_count': len(early_period),
            'recent_period_count': len(recent_period),
            'research_acceleration': len(recent_period) > len(early_period),
            'period_themes': {
                'early': self._extract_citation_themes(early_period)[:5],
                'recent': self._extract_citation_themes(recent_period)[:5]
            }
        }

    def _analyze_methodology_evolution(self, citation_data: List[Dict[str, Any]]) -> List[str]:
        """Analyze evolution in methodologies."""
        methodologies = []
        
        for citation in citation_data:
            title = citation.get('title', '').lower()
            abstract = citation.get('abstract', '').lower()
            
            # Look for methodology indicators
            method_terms = [
                'deep learning', 'machine learning', 'neural network', 'transformer',
                'reinforcement learning', 'supervised learning', 'unsupervised learning',
                'attention mechanism', 'convolutional', 'recurrent', 'optimization',
                'statistical', 'probabilistic', 'bayesian', 'ensemble'
            ]
            
            for term in method_terms:
                if term in title or term in abstract:
                    methodologies.append(term)
        
        # Count methodology mentions
        method_counts = {}
        for method in methodologies:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Return top methodologies
        sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        return [method for method, count in sorted_methods[:8]]

    def _extract_application_domains(self, citation_data: List[Dict[str, Any]]) -> List[str]:
        """Extract application domains from citations."""
        domains = []
        
        domain_terms = [
            'natural language processing', 'computer vision', 'speech recognition',
            'machine translation', 'sentiment analysis', 'question answering',
            'information retrieval', 'recommender systems', 'robotics',
            'autonomous driving', 'medical imaging', 'drug discovery',
            'finance', 'healthcare', 'education', 'social media'
        ]
        
        domain_counts = {}
        
        for citation in citation_data:
            title = citation.get('title', '').lower()
            abstract = citation.get('abstract', '').lower()
            text = f"{title} {abstract}"
            
            for domain in domain_terms:
                if domain in text:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Return top domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, count in sorted_domains[:6]]

    def _calculate_citation_growth(self, year_counts: Dict[int, int]) -> str:
        """Calculate citation growth trend."""
        if len(year_counts) < 2:
            return "insufficient_data"
        
        years = sorted(year_counts.keys())
        recent_years = years[-3:] if len(years) >= 3 else years
        early_years = years[:-3] if len(years) >= 3 else []
        
        if not early_years:
            return "increasing"
        
        recent_avg = sum(year_counts[y] for y in recent_years) / len(recent_years)
        early_avg = sum(year_counts[y] for y in early_years) / len(early_years)
        
        if recent_avg > early_avg * 1.2:
            return "accelerating"
        elif recent_avg > early_avg:
            return "increasing"
        elif recent_avg < early_avg * 0.8:
            return "declining"
        else:
            return "stable"

    def _assess_data_quality(self, paper_metadata: Dict[str, Any], 
                           paper_analysis: Dict[str, Any], 
                           citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality and completeness of synthesis data."""
        quality = {
            'paper_metadata_completeness': 0,
            'paper_analysis_completeness': 0,
            'citation_data_quality': 0,
            'overall_score': 0
        }
        
        # Assess paper metadata
        metadata_fields = ['title', 'authors', 'year']
        metadata_score = sum(1 for field in metadata_fields if paper_metadata.get(field))
        quality['paper_metadata_completeness'] = metadata_score / len(metadata_fields)
        
        # Assess paper analysis
        analysis_fields = ['key_concepts', 'methodology', 'findings']
        analysis_score = sum(1 for field in analysis_fields if paper_analysis.get(field))
        quality['paper_analysis_completeness'] = analysis_score / len(analysis_fields)
        
        # Assess citation data
        if citation_data:
            complete_citations = sum(1 for c in citation_data 
                                   if c.get('title') and c.get('year'))
            quality['citation_data_quality'] = complete_citations / len(citation_data)
        
        # Overall score
        quality['overall_score'] = (
            quality['paper_metadata_completeness'] * 0.3 +
            quality['paper_analysis_completeness'] * 0.4 +
            quality['citation_data_quality'] * 0.3
        )
        
        return quality

    async def _enhance_with_llm_analysis(self, synthesis_data: Dict[str, Any],
                                       paper_metadata: Dict[str, Any],
                                       citation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance synthesis with LLM-powered deep analysis.
        
        Args:
            synthesis_data: Basic synthesis data
            paper_metadata: Original paper metadata
            citation_data: Citation information
            
        Returns:
            Enhanced synthesis insights
        """
        # Prepare context for LLM analysis
        context = {
            'seminal_paper': {
                'title': paper_metadata.get('title', ''),
                'year': paper_metadata.get('year'),
                'key_concepts': synthesis_data['seminal_paper'].get('key_concepts', [])
            },
            'citation_summary': {
                'total_citations': len(citation_data),
                'citation_years': synthesis_data['citation_landscape']['citation_years'],
                'top_themes': synthesis_data['citation_landscape']['citation_themes'][:5],
                'methodologies': synthesis_data['research_evolution']['methodological_evolution'][:5]
            }
        }
        
        # Use LLM to analyze research impact and evolution
        result = await self.llm_utility.synthesize_research_directions(
            paper_data=context,
            citations=citation_data[:10],  # Use top 10 citations for context
            analysis_type='comprehensive_synthesis'
        )
        
        if result.get('success'):
            return {
                'llm_insights': result.get('insights', {}),
                'research_impact_analysis': result.get('impact_analysis', {}),
                'field_evolution_summary': result.get('evolution_summary', {})
            }
        
        return {}


class TrendAnalysisNode(AsyncNode):
    """Node for identifying research trends and gaps."""

    def __init__(self, llm_utility: LLMAnalysisUtility):
        """Initialize the trend analysis node.
        
        Args:
            llm_utility: LLM utility for analysis
        """
        self.llm_utility = llm_utility

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends and identify research gaps.
        
        Args:
            context: Execution context
            
        Returns:
            Identified trends and gaps with success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            synthesis_data = store.get('comprehensive_synthesis', {})
            if not synthesis_data:
                raise ResearchSynthesisNodeError("No synthesis data available for trend analysis")
            
            store['status'] = 'analyzing_trends'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info("Analyzing research trends and identifying gaps...")
            
            # Analyze trends from synthesis data
            trends = self._analyze_research_trends(synthesis_data)
            
            # Identify research gaps
            gaps = self._identify_research_gaps(synthesis_data, trends)
            
            # Use LLM for enhanced trend analysis if available
            try:
                enhanced_analysis = await self._enhance_trend_analysis_with_llm(
                    synthesis_data, trends, gaps
                )
                trends.update(enhanced_analysis.get('enhanced_trends', {}))
                gaps.extend(enhanced_analysis.get('additional_gaps', []))
                logger.info("Enhanced trend analysis with LLM insights")
            except Exception as e:
                logger.warning(f"LLM trend enhancement failed: {e}")
            
            store['identified_trends'] = trends
            store['research_gaps'] = gaps
            store['synthesis_stats']['trends_identified'] = len(trends.get('major_trends', []))
            store['synthesis_stats']['gaps_found'] = len(gaps)
            store['status'] = 'trend_analysis_complete'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Identified {len(trends.get('major_trends', []))} major trends and {len(gaps)} research gaps")
            
            return {
                'success': True,
                'trends': trends,
                'gaps': gaps
            }
            
        except Exception as e:
            error_msg = f"Trend analysis failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'trend_analysis_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _analyze_research_trends(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research trends from synthesis data.
        
        Args:
            synthesis_data: Comprehensive synthesis data
            
        Returns:
            Identified trends and patterns
        """
        trends = {
            'major_trends': [],
            'methodological_trends': [],
            'application_trends': [],
            'temporal_patterns': {},
            'trend_confidence': {}
        }
        
        # Analyze methodological trends
        methodologies = synthesis_data.get('research_evolution', {}).get('methodological_evolution', [])
        if methodologies:
            trends['methodological_trends'] = [
                {
                    'trend': f"Increasing adoption of {method}",
                    'evidence': f"Mentioned in multiple citing papers",
                    'strength': 'moderate' if methodologies.index(method) < len(methodologies)//2 else 'weak'
                }
                for method in methodologies[:3]
            ]
        
        # Analyze application domain trends
        domains = synthesis_data.get('research_evolution', {}).get('application_domains', [])
        if domains:
            trends['application_trends'] = [
                {
                    'trend': f"Expansion into {domain}",
                    'evidence': "Citations show application diversity",
                    'strength': 'strong' if domains.index(domain) == 0 else 'moderate'
                }
                for domain in domains[:3]
            ]
        
        # Analyze temporal patterns
        temporal_data = synthesis_data.get('research_evolution', {}).get('temporal_trends', {})
        if temporal_data.get('research_acceleration'):
            trends['major_trends'].append({
                'trend': "Accelerating research interest",
                'evidence': f"Recent period has {temporal_data.get('recent_period_count', 0)} citations vs {temporal_data.get('early_period_count', 0)} earlier",
                'strength': 'strong',
                'type': 'growth'
            })
        
        # Analyze citation growth
        citation_landscape = synthesis_data.get('citation_landscape', {})
        years_data = citation_landscape.get('citation_years', {})
        growth_trend = years_data.get('citation_growth', 'stable')
        
        if growth_trend == 'accelerating':
            trends['major_trends'].append({
                'trend': "Exponential impact growth",
                'evidence': "Citation rate is accelerating over time",
                'strength': 'strong',
                'type': 'impact'
            })
        elif growth_trend == 'increasing':
            trends['major_trends'].append({
                'trend': "Sustained research impact",
                'evidence': "Steady citation growth over time",
                'strength': 'moderate',
                'type': 'impact'
            })
        
        # Analyze venue diversity
        venues = citation_landscape.get('citing_venues', [])
        if len(venues) >= 5:
            trends['major_trends'].append({
                'trend': "Cross-disciplinary influence",
                'evidence': f"Papers cited in {len(venues)} different venues",
                'strength': 'moderate',
                'type': 'interdisciplinary'
            })
        
        # Add confidence scores
        for trend_type in ['major_trends', 'methodological_trends', 'application_trends']:
            if trends[trend_type]:
                total_strength = sum(1 if t.get('strength') == 'strong' else 0.5 if t.get('strength') == 'moderate' else 0.25 
                                   for t in trends[trend_type])
                trends['trend_confidence'][trend_type] = min(total_strength / len(trends[trend_type]), 1.0)
        
        return trends

    def _identify_research_gaps(self, synthesis_data: Dict[str, Any], 
                               trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential research gaps from analysis.
        
        Args:
            synthesis_data: Comprehensive synthesis data
            trends: Identified trends
            
        Returns:
            List of potential research gaps
        """
        gaps = []
        
        # Look for methodological gaps
        methodologies = synthesis_data.get('research_evolution', {}).get('methodological_evolution', [])
        modern_methods = ['transformer', 'attention mechanism', 'deep learning']
        
        missing_modern = [method for method in modern_methods if method not in methodologies]
        for method in missing_modern:
            gaps.append({
                'type': 'methodological',
                'description': f"Limited exploration of {method} approaches",
                'evidence': f"{method} not prominent in citing literature",
                'opportunity_level': 'high',
                'research_direction': f"Apply {method} to problems in this domain"
            })
        
        # Look for application domain gaps
        domains = synthesis_data.get('research_evolution', {}).get('application_domains', [])
        
        # Common domains that might be underexplored
        potential_domains = ['healthcare', 'education', 'finance', 'robotics', 'autonomous driving']
        missing_domains = [domain for domain in potential_domains if domain not in domains]
        
        for domain in missing_domains[:2]:  # Limit to top 2
            gaps.append({
                'type': 'application',
                'description': f"Underexplored applications in {domain}",
                'evidence': f"{domain} applications not well represented",
                'opportunity_level': 'medium',
                'research_direction': f"Investigate applications and adaptations for {domain}"
            })
        
        # Look for theoretical gaps
        seminal_paper = synthesis_data.get('seminal_paper', {})
        key_concepts = seminal_paper.get('key_concepts', [])
        
        # Check for theoretical development opportunities
        if 'theoretical' not in ' '.join(key_concepts).lower():
            gaps.append({
                'type': 'theoretical',
                'description': "Limited theoretical foundations development",
                'evidence': "Few citations focus on theoretical aspects",
                'opportunity_level': 'medium',
                'research_direction': "Develop stronger theoretical foundations and mathematical frameworks"
            })
        
        # Look for temporal gaps
        temporal_trends = synthesis_data.get('research_evolution', {}).get('temporal_trends', {})
        recent_themes = temporal_trends.get('period_themes', {}).get('recent', [])
        early_themes = temporal_trends.get('period_themes', {}).get('early', [])
        
        # Identify themes that were explored early but not recently
        abandoned_themes = [theme for theme in early_themes if theme not in recent_themes]
        for theme in abandoned_themes[:2]:
            gaps.append({
                'type': 'temporal',
                'description': f"Underexplored modern approaches to {theme}",
                'evidence': f"'{theme}' was explored early but not in recent work",
                'opportunity_level': 'medium',
                'research_direction': f"Revisit {theme} with modern techniques and perspectives"
            })
        
        # Look for interdisciplinary gaps
        venues = synthesis_data.get('citation_landscape', {}).get('citing_venues', [])
        venue_names = [v.get('venue', '').lower() for v in venues]
        
        # Check for missing interdisciplinary connections
        disciplines = ['biology', 'psychology', 'physics', 'sociology', 'economics']
        missing_disciplines = []
        
        for discipline in disciplines:
            if not any(discipline in venue for venue in venue_names):
                missing_disciplines.append(discipline)
        
        if missing_disciplines:
            for discipline in missing_disciplines[:2]:
                gaps.append({
                    'type': 'interdisciplinary',
                    'description': f"Limited connections to {discipline}",
                    'evidence': f"No citations from {discipline} venues",
                    'opportunity_level': 'low',
                    'research_direction': f"Explore connections and applications in {discipline}"
                })
        
        return gaps

    async def _enhance_trend_analysis_with_llm(self, synthesis_data: Dict[str, Any],
                                             trends: Dict[str, Any],
                                             gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance trend analysis using LLM insights.
        
        Args:
            synthesis_data: Synthesis data
            trends: Basic trends analysis
            gaps: Identified gaps
            
        Returns:
            Enhanced analysis from LLM
        """
        # Prepare context for LLM analysis
        context = {
            'research_trends': trends.get('major_trends', [])[:3],
            'methodological_evolution': synthesis_data.get('research_evolution', {}).get('methodological_evolution', [])[:5],
            'application_domains': synthesis_data.get('research_evolution', {}).get('application_domains', [])[:5],
            'identified_gaps': gaps[:3]
        }
        
        # Use LLM for enhanced trend analysis
        result = await self.llm_utility.synthesize_research_directions(
            paper_data=synthesis_data.get('seminal_paper', {}),
            citations=[],  # Focus on trend analysis rather than specific citations
            analysis_type='trend_analysis',
            context=context
        )
        
        if result.get('success'):
            return {
                'enhanced_trends': result.get('enhanced_trends', {}),
                'additional_gaps': result.get('additional_gaps', []),
                'trend_insights': result.get('trend_insights', {})
            }
        
        return {}


class DirectionGeneratorNode(AsyncNode):
    """Node for generating novel research directions."""

    def __init__(self, llm_utility: LLMAnalysisUtility):
        """Initialize the direction generator node.
        
        Args:
            llm_utility: LLM utility for generation
        """
        self.llm_utility = llm_utility

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate novel research directions.
        
        Args:
            context: Execution context
            
        Returns:
            Generated research directions with success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            trends = store.get('identified_trends', {})
            gaps = store.get('research_gaps', [])
            synthesis_data = store.get('comprehensive_synthesis', {})
            
            if not trends and not gaps and not synthesis_data:
                raise ResearchSynthesisNodeError("No trends, gaps, or synthesis data available for direction generation")
            
            # If no trends or gaps but we have synthesis data, create basic directions
            if not trends and not gaps and synthesis_data:
                logger.info("No trends or gaps available, generating basic directions from synthesis data")
                basic_directions = self._generate_basic_directions_from_synthesis(synthesis_data)
                store['suggested_directions'] = basic_directions
                store['synthesis_stats']['suggestions_generated'] = len(basic_directions)
                store['status'] = 'directions_generated'
                store['last_updated'] = datetime.now().isoformat()
                
                return {
                    'success': True,
                    'directions': basic_directions
                }
            
            store['status'] = 'generating_directions'
            store['last_updated'] = datetime.now().isoformat()
            
            max_suggestions = config.get('max_suggestions', 5)
            min_confidence = config.get('min_confidence', 0.7)
            
            logger.info(f"Generating up to {max_suggestions} research directions...")
            
            # Generate directions from gaps
            gap_directions = self._generate_directions_from_gaps(gaps, synthesis_data)
            
            # Generate directions from trends
            trend_directions = self._generate_directions_from_trends(trends, synthesis_data)
            
            # Combine and prioritize directions
            all_directions = gap_directions + trend_directions
            
            # Use LLM to enhance and validate directions
            try:
                enhanced_directions = await self._enhance_directions_with_llm(
                    all_directions, synthesis_data, trends, gaps
                )
                all_directions.extend(enhanced_directions)
                logger.info("Enhanced directions with LLM generation")
            except Exception as e:
                logger.warning(f"LLM direction enhancement failed: {e}")
            
            # Filter and rank directions
            final_directions = self._filter_and_rank_directions(
                all_directions, min_confidence, max_suggestions
            )
            
            # Add confidence scores and metadata
            for i, direction in enumerate(final_directions):
                direction['rank'] = i + 1
                direction['generated_at'] = datetime.now().isoformat()
                if 'confidence' not in direction:
                    direction['confidence'] = self._calculate_confidence(direction, gaps, trends)
            
            store['suggested_directions'] = final_directions
            store['synthesis_stats']['suggestions_generated'] = len(final_directions)
            store['synthesis_stats']['high_confidence_suggestions'] = sum(
                1 for d in final_directions if d.get('confidence', 0) >= 0.8
            )
            store['status'] = 'directions_generated'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Generated {len(final_directions)} research directions")
            
            return {
                'success': True,
                'directions': final_directions
            }
            
        except Exception as e:
            error_msg = f"Direction generation failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'direction_generation_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_directions_from_gaps(self, gaps: List[Dict[str, Any]], 
                                     synthesis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate research directions from identified gaps.
        
        Args:
            gaps: Identified research gaps
            synthesis_data: Synthesis data for context
            
        Returns:
            List of research directions
        """
        directions = []
        seminal_paper = synthesis_data.get('seminal_paper', {})
        
        for gap in gaps:
            direction = {
                'title': f"Address {gap['description']}",
                'description': gap.get('research_direction', ''),
                'rationale': f"This direction addresses a {gap['type']} gap: {gap['description']}",
                'approach': self._suggest_approach_for_gap(gap, seminal_paper),
                'impact_potential': gap.get('opportunity_level', 'medium'),
                'research_type': gap['type'],
                'source': 'gap_analysis',
                'gap_evidence': gap.get('evidence', ''),
                'feasibility': self._assess_feasibility(gap),
                'novelty_score': self._calculate_novelty_score(gap),
                'confidence': self._calculate_gap_confidence(gap)
            }
            directions.append(direction)
        
        return directions

    def _generate_directions_from_trends(self, trends: Dict[str, Any], 
                                       synthesis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate research directions from identified trends.
        
        Args:
            trends: Identified trends
            synthesis_data: Synthesis data for context
            
        Returns:
            List of research directions
        """
        directions = []
        seminal_paper = synthesis_data.get('seminal_paper', {})
        
        # Generate directions from major trends
        for trend in trends.get('major_trends', []):
            direction = {
                'title': f"Leverage {trend['trend']} for Novel Applications",
                'description': f"Explore new applications building on the {trend['trend'].lower()}",
                'rationale': f"This direction capitalizes on a {trend['strength']} trend: {trend['trend']}",
                'approach': self._suggest_approach_for_trend(trend, seminal_paper),
                'impact_potential': 'high' if trend['strength'] == 'strong' else 'medium',
                'research_type': trend.get('type', 'extension'),
                'source': 'trend_analysis',
                'trend_evidence': trend.get('evidence', ''),
                'feasibility': 'high',  # Trends are generally feasible
                'novelty_score': 0.7,  # Moderate novelty for trend-based directions
                'confidence': 0.8 if trend['strength'] == 'strong' else 0.6
            }
            directions.append(direction)
        
        return directions

    def _suggest_approach_for_gap(self, gap: Dict[str, Any], 
                                seminal_paper: Dict[str, Any]) -> str:
        """Suggest research approach for addressing a gap.
        
        Args:
            gap: Research gap information
            seminal_paper: Original paper context
            
        Returns:
            Suggested research approach
        """
        gap_type = gap['type']
        approaches = {
            'methodological': f"Develop and validate new {gap.get('research_direction', 'approaches')} with systematic comparison to existing methods",
            'application': f"Conduct domain-specific adaptation studies, validate in real-world scenarios, and measure domain-specific performance metrics",
            'theoretical': f"Develop mathematical frameworks, prove theoretical properties, and establish formal foundations with empirical validation",
            'temporal': f"Apply modern techniques to revisit classic problems, conduct comparative studies across time periods",
            'interdisciplinary': f"Establish cross-disciplinary collaborations, adapt methods to new domain constraints, validate across disciplinary boundaries"
        }
        
        base_approach = approaches.get(gap_type, "Systematic investigation with empirical validation")
        
        # Add context from seminal paper
        key_concepts = seminal_paper.get('key_concepts', [])
        if key_concepts:
            base_approach += f". Build upon the foundational concepts of {', '.join(key_concepts[:2])}"
        
        return base_approach

    def _suggest_approach_for_trend(self, trend: Dict[str, Any], 
                                  seminal_paper: Dict[str, Any]) -> str:
        """Suggest research approach for leveraging a trend.
        
        Args:
            trend: Trend information
            seminal_paper: Original paper context
            
        Returns:
            Suggested research approach
        """
        trend_type = trend.get('type', 'general')
        approaches = {
            'growth': "Conduct scaling studies, investigate emerging applications, and develop next-generation methods",
            'impact': "Focus on high-impact applications, develop practical implementations, and measure real-world effectiveness",
            'interdisciplinary': "Explore cross-domain applications, establish new collaborations, and validate in multiple contexts"
        }
        
        base_approach = approaches.get(trend_type, "Systematic extension and validation of emerging approaches")
        
        # Add context from seminal paper
        methodology = seminal_paper.get('methodology', '')
        if methodology:
            base_approach += f". Extend the core methodology of {methodology}"
        
        return base_approach

    def _filter_and_rank_directions(self, directions: List[Dict[str, Any]], 
                                  min_confidence: float, 
                                  max_suggestions: int) -> List[Dict[str, Any]]:
        """Filter and rank research directions.
        
        Args:
            directions: List of all generated directions
            min_confidence: Minimum confidence threshold
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            Filtered and ranked directions
        """
        # Filter by confidence
        filtered = [d for d in directions if d.get('confidence', 0) >= min_confidence]
        
        # If too few high-confidence directions, lower threshold slightly
        if len(filtered) < max_suggestions // 2:
            filtered = [d for d in directions if d.get('confidence', 0) >= min_confidence - 0.1]
        
        # Rank by composite score
        def composite_score(direction):
            confidence = direction.get('confidence', 0)
            novelty = direction.get('novelty_score', 0)
            feasibility = 1.0 if direction.get('feasibility') == 'high' else 0.7 if direction.get('feasibility') == 'medium' else 0.4
            impact = 1.0 if direction.get('impact_potential') == 'high' else 0.7 if direction.get('impact_potential') == 'medium' else 0.4
            
            return (confidence * 0.3 + novelty * 0.3 + feasibility * 0.2 + impact * 0.2)
        
        filtered.sort(key=composite_score, reverse=True)
        
        return filtered[:max_suggestions]

    def _assess_feasibility(self, gap: Dict[str, Any]) -> str:
        """Assess feasibility of addressing a research gap.
        
        Args:
            gap: Research gap information
            
        Returns:
            Feasibility assessment (high/medium/low)
        """
        gap_type = gap['type']
        opportunity_level = gap.get('opportunity_level', 'medium')
        
        # Methodological gaps are generally feasible
        if gap_type == 'methodological':
            return 'high'
        
        # Application gaps depend on opportunity level
        if gap_type == 'application':
            return 'high' if opportunity_level == 'high' else 'medium'
        
        # Theoretical gaps are challenging but important
        if gap_type == 'theoretical':
            return 'medium'
        
        # Others are generally feasible
        return 'medium'

    def _calculate_novelty_score(self, gap: Dict[str, Any]) -> float:
        """Calculate novelty score for a gap-based direction.
        
        Args:
            gap: Research gap information
            
        Returns:
            Novelty score (0-1)
        """
        gap_type = gap['type']
        opportunity_level = gap.get('opportunity_level', 'medium')
        
        # Base scores by gap type
        base_scores = {
            'methodological': 0.8,  # High novelty for new methods
            'theoretical': 0.9,     # Very high novelty for theory
            'application': 0.6,     # Moderate novelty for applications
            'temporal': 0.7,        # Good novelty for revisiting topics
            'interdisciplinary': 0.8  # High novelty for cross-domain work
        }
        
        base_score = base_scores.get(gap_type, 0.6)
        
        # Adjust by opportunity level
        if opportunity_level == 'high':
            return min(base_score + 0.1, 1.0)
        elif opportunity_level == 'low':
            return max(base_score - 0.1, 0.0)
        
        return base_score

    def _calculate_gap_confidence(self, gap: Dict[str, Any]) -> float:
        """Calculate confidence score for a gap-based direction.
        
        Args:
            gap: Research gap information
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence by gap type
        type_confidence = {
            'methodological': 0.8,
            'application': 0.7,
            'theoretical': 0.6,
            'temporal': 0.7,
            'interdisciplinary': 0.6
        }
        
        base_confidence = type_confidence.get(gap['type'], 0.6)
        
        # Adjust by opportunity level
        opportunity_level = gap.get('opportunity_level', 'medium')
        if opportunity_level == 'high':
            return min(base_confidence + 0.1, 1.0)
        elif opportunity_level == 'low':
            return max(base_confidence - 0.2, 0.1)
        
        return base_confidence

    def _generate_basic_directions_from_synthesis(self, synthesis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic research directions from synthesis data when trends/gaps are unavailable.
        
        Args:
            synthesis_data: Comprehensive synthesis data
            
        Returns:
            List of basic research directions
        """
        directions = []
        seminal_paper = synthesis_data.get('seminal_paper', {})
        citation_landscape = synthesis_data.get('citation_landscape', {})
        research_evolution = synthesis_data.get('research_evolution', {})
        
        # Direction 1: Extension of core methodology
        if seminal_paper.get('key_concepts'):
            directions.append({
                'title': f"Advanced {seminal_paper['key_concepts'][0].title()} Architectures",
                'description': f"Develop next-generation approaches building on {seminal_paper['key_concepts'][0]}",
                'rationale': f"Build upon the foundational work of {seminal_paper.get('title', 'the seminal paper')}",
                'approach': "Systematic extension of core methodologies with modern techniques",
                'impact_potential': 'medium',
                'research_type': 'methodological',
                'source': 'synthesis_fallback',
                'feasibility': 'high',
                'novelty_score': 0.6,
                'confidence': 0.65,
                'rank': 1,
                'generated_at': datetime.now().isoformat()
            })
        
        # Direction 2: Cross-domain applications
        if research_evolution.get('application_domains'):
            domains = research_evolution['application_domains']
            directions.append({
                'title': f"Cross-Domain Applications Beyond {domains[0].title()}",
                'description': f"Explore applications in underexplored domains beyond current focus areas",
                'rationale': f"Expand the impact beyond established domains like {domains[0]}",
                'approach': "Domain adaptation studies with cross-validation",
                'impact_potential': 'medium',
                'research_type': 'application',
                'source': 'synthesis_fallback',
                'feasibility': 'medium',
                'novelty_score': 0.55,
                'confidence': 0.6,
                'rank': 2,
                'generated_at': datetime.now().isoformat()
            })
        
        # Direction 3: Scalability and efficiency
        if citation_landscape.get('total_citations', 0) > 5:  # Popular work likely has efficiency concerns
            directions.append({
                'title': "Efficiency and Scalability Improvements",
                'description': "Develop more efficient variants while maintaining performance",
                'rationale': "High citation count suggests need for practical, scalable implementations",
                'approach': "Computational optimization with performance benchmarking",
                'impact_potential': 'high',
                'research_type': 'methodological',
                'source': 'synthesis_fallback',
                'feasibility': 'high',
                'novelty_score': 0.5,
                'confidence': 0.7,
                'rank': 3,
                'generated_at': datetime.now().isoformat()
            })
        
        return directions[:3]  # Return top 3 basic directions

    def _calculate_confidence(self, direction: Dict[str, Any], 
                            gaps: List[Dict[str, Any]], 
                            trends: Dict[str, Any]) -> float:
        """Calculate overall confidence for a research direction.
        
        Args:
            direction: Research direction
            gaps: All identified gaps
            trends: All identified trends
            
        Returns:
            Confidence score (0-1)
        """
        if direction.get('confidence'):
            return direction['confidence']
        
        # Default confidence calculation
        source = direction.get('source', 'unknown')
        research_type = direction.get('research_type', 'general')
        
        if source == 'gap_analysis':
            return self._calculate_gap_confidence({'type': research_type})
        elif source == 'trend_analysis':
            return 0.7  # Moderate confidence for trend-based directions
        
        return 0.6  # Default confidence

    async def _enhance_directions_with_llm(self, directions: List[Dict[str, Any]],
                                         synthesis_data: Dict[str, Any],
                                         trends: Dict[str, Any],
                                         gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate additional research directions using LLM.
        
        Args:
            directions: Existing directions
            synthesis_data: Synthesis data
            trends: Identified trends
            gaps: Identified gaps
            
        Returns:
            Additional LLM-generated directions
        """
        # Prepare context for LLM
        context = {
            'existing_directions': [d['title'] for d in directions[:3]],
            'seminal_paper': synthesis_data.get('seminal_paper', {}),
            'major_trends': trends.get('major_trends', [])[:3],
            'key_gaps': gaps[:3]
        }
        
        # Use LLM to generate novel directions
        result = await self.llm_utility.synthesize_research_directions(
            paper_data=context['seminal_paper'],
            citations=[],
            analysis_type='novel_directions',
            context=context
        )
        
        if result.get('success') and result.get('suggestions'):
            llm_directions = []
            for suggestion in result['suggestions'][:2]:  # Limit to 2 additional
                direction = {
                    'title': suggestion.get('title', ''),
                    'description': suggestion.get('description', ''),
                    'rationale': suggestion.get('rationale', ''),
                    'approach': suggestion.get('approach', ''),
                    'impact_potential': suggestion.get('impact_potential', 'medium'),
                    'research_type': 'novel',
                    'source': 'llm_generation',
                    'feasibility': suggestion.get('feasibility', 'medium'),
                    'novelty_score': suggestion.get('novelty_score', 0.8),
                    'confidence': suggestion.get('confidence', 0.7)
                }
                llm_directions.append(direction)
            
            return llm_directions
        
        return []


class SuggestionFormatterNode(AsyncNode):
    """Node for formatting research suggestions for presentation."""

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format research suggestions for presentation.
        
        Args:
            context: Execution context
            
        Returns:
            Formatted suggestions with success status
        """
        store = context['store']
        config = context.get('config', {})
        
        try:
            directions = store.get('suggested_directions', [])
            synthesis_data = store.get('comprehensive_synthesis', {})
            trends = store.get('identified_trends', {})
            gaps = store.get('research_gaps', [])
            
            # Ensure trends is a dict (it might be empty list if trend analysis failed)
            if not isinstance(trends, dict):
                trends = {}
            
            store['status'] = 'formatting_suggestions'
            store['last_updated'] = datetime.now().isoformat()
            
            # Create comprehensive formatted output
            formatted = {
                'research_suggestions': [],
                'synthesis_summary': self._create_synthesis_summary(synthesis_data, trends, gaps),
                'methodology_overview': self._create_methodology_overview(directions),
                'impact_assessment': self._create_impact_assessment(directions),
                'implementation_roadmap': self._create_implementation_roadmap(directions),
                'metadata': {
                    'agent': 'AcademicNewResearchAgent',
                    'generated_at': datetime.now().isoformat(),
                    'synthesis_quality': synthesis_data.get('synthesis_metadata', {}).get('data_quality', {}),
                    'confidence_distribution': self._analyze_confidence_distribution(directions)
                }
            }
            
            # Format individual suggestions
            for direction in directions:
                formatted_suggestion = {
                    'rank': direction.get('rank', 0),
                    'title': direction.get('title', ''),
                    'description': direction.get('description', ''),
                    'rationale': direction.get('rationale', ''),
                    'approach': direction.get('approach', ''),
                    'confidence': direction.get('confidence', 0),
                    'impact_potential': direction.get('impact_potential', 'medium'),
                    'feasibility': direction.get('feasibility', 'medium'),
                    'novelty_score': direction.get('novelty_score', 0),
                    'research_type': direction.get('research_type', 'general'),
                    'source': direction.get('source', 'analysis'),
                    'estimated_timeline': self._estimate_timeline(direction),
                    'required_resources': self._estimate_resources(direction),
                    'potential_collaborations': self._suggest_collaborations(direction),
                    'risk_factors': self._identify_risks(direction),
                    'success_metrics': self._define_success_metrics(direction)
                }
                formatted['research_suggestions'].append(formatted_suggestion)
            
            store['formatted_suggestions'] = formatted
            store['status'] = 'suggestions_formatted'
            store['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Formatted {len(directions)} research suggestions with comprehensive metadata")
            
            return {
                'success': True,
                'formatted': formatted
            }
            
        except Exception as e:
            error_msg = f"Suggestion formatting failed: {str(e)}"
            logger.error(error_msg)
            store['errors'].append(error_msg)
            store['status'] = 'formatting_error'
            store['last_updated'] = datetime.now().isoformat()
            
            return {
                'success': False,
                'error': error_msg
            }

    def _create_synthesis_summary(self, synthesis_data: Dict[str, Any],
                                trends: Dict[str, Any],
                                gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the synthesis process.
        
        Args:
            synthesis_data: Synthesis data
            trends: Identified trends
            gaps: Identified gaps
            
        Returns:
            Synthesis summary
        """
        seminal_paper = synthesis_data.get('seminal_paper', {})
        citation_landscape = synthesis_data.get('citation_landscape', {})
        
        return {
            'seminal_paper_overview': {
                'title': seminal_paper.get('title', ''),
                'year': seminal_paper.get('year'),
                'key_contributions': seminal_paper.get('main_findings', [])[:3],
                'impact_summary': f"Cited by {citation_landscape.get('total_citations', 0)} papers"
            },
            'research_landscape': {
                'citation_span': citation_landscape.get('citation_years', {}),
                'field_diversity': len(citation_landscape.get('citing_venues', [])),
                'major_themes': citation_landscape.get('citation_themes', [])[:5]
            },
            'trend_overview': {
                'major_trends_count': len(trends.get('major_trends', [])),
                'strongest_trend': trends.get('major_trends', [{}])[0].get('trend', 'N/A') if trends.get('major_trends') else 'N/A',
                'methodological_evolution': trends.get('methodological_trends', [])[:3]
            },
            'gap_analysis': {
                'total_gaps_identified': len(gaps),
                'gap_types': list(set(gap['type'] for gap in gaps)),
                'high_opportunity_gaps': [gap for gap in gaps if gap.get('opportunity_level') == 'high']
            }
        }

    def _create_methodology_overview(self, directions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create overview of methodological approaches in suggestions.
        
        Args:
            directions: Research directions
            
        Returns:
            Methodology overview
        """
        research_types = {}
        sources = {}
        
        for direction in directions:
            research_type = direction.get('research_type', 'general')
            source = direction.get('source', 'analysis')
            
            research_types[research_type] = research_types.get(research_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'research_type_distribution': research_types,
            'suggestion_sources': sources,
            'primary_approaches': [d.get('approach', '')[:100] + '...' for d in directions[:3]],
            'methodological_diversity': len(set(d.get('research_type') for d in directions))
        }

    def _create_impact_assessment(self, directions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create impact assessment for research directions.
        
        Args:
            directions: Research directions
            
        Returns:
            Impact assessment
        """
        high_impact = [d for d in directions if d.get('impact_potential') == 'high']
        medium_impact = [d for d in directions if d.get('impact_potential') == 'medium']
        
        avg_confidence = sum(d.get('confidence', 0) for d in directions) / len(directions) if directions else 0
        avg_novelty = sum(d.get('novelty_score', 0) for d in directions) / len(directions) if directions else 0
        
        return {
            'impact_distribution': {
                'high_impact_count': len(high_impact),
                'medium_impact_count': len(medium_impact),
                'low_impact_count': len(directions) - len(high_impact) - len(medium_impact)
            },
            'quality_metrics': {
                'average_confidence': round(avg_confidence, 2),
                'average_novelty': round(avg_novelty, 2),
                'high_confidence_suggestions': len([d for d in directions if d.get('confidence', 0) >= 0.8])
            },
            'strategic_recommendations': self._generate_strategic_recommendations(directions)
        }

    def _create_implementation_roadmap(self, directions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation roadmap for research directions.
        
        Args:
            directions: Research directions
            
        Returns:
            Implementation roadmap
        """
        # Categorize by timeline
        short_term = []  # <= 1 year
        medium_term = []  # 1-3 years
        long_term = []  # > 3 years
        
        for direction in directions:
            timeline = self._estimate_timeline(direction)
            if 'months' in timeline or '1 year' in timeline:
                short_term.append(direction)
            elif '2 year' in timeline or '3 year' in timeline:
                medium_term.append(direction)
            else:
                long_term.append(direction)
        
        return {
            'implementation_phases': {
                'short_term': [{'title': d['title'], 'timeline': self._estimate_timeline(d)} for d in short_term],
                'medium_term': [{'title': d['title'], 'timeline': self._estimate_timeline(d)} for d in medium_term],
                'long_term': [{'title': d['title'], 'timeline': self._estimate_timeline(d)} for d in long_term]
            },
            'priority_order': [d['title'] for d in sorted(directions, key=lambda x: x.get('confidence', 0), reverse=True)],
            'resource_requirements': {
                'computational': sum(1 for d in directions if 'computational' in d.get('approach', '').lower()),
                'theoretical': sum(1 for d in directions if d.get('research_type') == 'theoretical'),
                'experimental': sum(1 for d in directions if 'experimental' in d.get('approach', '').lower())
            }
        }

    def _analyze_confidence_distribution(self, directions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence distribution across directions.
        
        Args:
            directions: Research directions
            
        Returns:
            Confidence distribution analysis
        """
        if not directions:
            return {'distribution': 'No directions available'}
        
        confidences = [d.get('confidence', 0) for d in directions]
        
        return {
            'high_confidence': len([c for c in confidences if c >= 0.8]),
            'medium_confidence': len([c for c in confidences if 0.6 <= c < 0.8]),
            'low_confidence': len([c for c in confidences if c < 0.6]),
            'average': round(sum(confidences) / len(confidences), 2),
            'range': f"{min(confidences):.2f} - {max(confidences):.2f}"
        }

    def _estimate_timeline(self, direction: Dict[str, Any]) -> str:
        """Estimate implementation timeline for a research direction.
        
        Args:
            direction: Research direction
            
        Returns:
            Estimated timeline
        """
        research_type = direction.get('research_type', 'general')
        feasibility = direction.get('feasibility', 'medium')
        
        # Base timelines by research type
        base_timelines = {
            'methodological': '12-18 months',
            'application': '6-12 months',
            'theoretical': '18-24 months',
            'temporal': '9-15 months',
            'interdisciplinary': '15-24 months',
            'novel': '18-30 months'
        }
        
        base_timeline = base_timelines.get(research_type, '12-18 months')
        
        # Adjust by feasibility
        if feasibility == 'high':
            return base_timeline
        elif feasibility == 'low':
            # Extend timeline for low feasibility
            if 'months' in base_timeline:
                return base_timeline.replace('months', 'months (extended timeline)')
            return base_timeline + ' (extended)'
        
        return base_timeline

    def _estimate_resources(self, direction: Dict[str, Any]) -> List[str]:
        """Estimate required resources for a research direction.
        
        Args:
            direction: Research direction
            
        Returns:
            List of required resources
        """
        resources = []
        research_type = direction.get('research_type', 'general')
        approach = direction.get('approach', '').lower()
        
        # Add resources based on research type
        if research_type == 'methodological':
            resources.extend(['Computational resources', 'Development expertise', 'Benchmark datasets'])
        elif research_type == 'application':
            resources.extend(['Domain expertise', 'Application data', 'User testing capabilities'])
        elif research_type == 'theoretical':
            resources.extend(['Mathematical expertise', 'Proof verification tools', 'Literature access'])
        elif research_type == 'interdisciplinary':
            resources.extend(['Cross-domain collaboration', 'Multi-disciplinary expertise'])
        
        # Add resources based on approach
        if 'experimental' in approach:
            resources.append('Experimental setup')
        if 'comparative' in approach:
            resources.append('Multiple baseline implementations')
        if 'validation' in approach:
            resources.append('Validation datasets')
        
        return list(set(resources))  # Remove duplicates

    def _suggest_collaborations(self, direction: Dict[str, Any]) -> List[str]:
        """Suggest potential collaborations for a research direction.
        
        Args:
            direction: Research direction
            
        Returns:
            List of suggested collaborations
        """
        collaborations = []
        research_type = direction.get('research_type', 'general')
        
        collaboration_map = {
            'methodological': ['ML research groups', 'Algorithm developers', 'Software engineers'],
            'application': ['Domain experts', 'Industry partners', 'User experience researchers'],
            'theoretical': ['Mathematics departments', 'Computer science theorists', 'Logic experts'],
            'interdisciplinary': ['Cross-domain researchers', 'Interdisciplinary centers', 'Industry partners'],
            'temporal': ['Historical research groups', 'Longitudinal study experts']
        }
        
        return collaboration_map.get(research_type, ['Relevant research groups', 'Domain experts'])

    def _identify_risks(self, direction: Dict[str, Any]) -> List[str]:
        """Identify potential risks for a research direction.
        
        Args:
            direction: Research direction
            
        Returns:
            List of risk factors
        """
        risks = []
        feasibility = direction.get('feasibility', 'medium')
        research_type = direction.get('research_type', 'general')
        confidence = direction.get('confidence', 0)
        
        # Feasibility-based risks
        if feasibility == 'low':
            risks.append('High implementation complexity')
        
        # Confidence-based risks
        if confidence < 0.6:
            risks.append('Uncertain research outcomes')
        
        # Research type-based risks
        type_risks = {
            'theoretical': ['Proof complexity', 'Limited empirical validation'],
            'application': ['Domain-specific constraints', 'Real-world applicability'],
            'methodological': ['Scalability issues', 'Baseline comparison challenges'],
            'interdisciplinary': ['Communication barriers', 'Conflicting domain requirements']
        }
        
        risks.extend(type_risks.get(research_type, []))
        
        return risks

    def _define_success_metrics(self, direction: Dict[str, Any]) -> List[str]:
        """Define success metrics for a research direction.
        
        Args:
            direction: Research direction
            
        Returns:
            List of success metrics
        """
        metrics = []
        research_type = direction.get('research_type', 'general')
        impact_potential = direction.get('impact_potential', 'medium')
        
        # Base metrics by research type
        type_metrics = {
            'methodological': ['Performance improvement over baselines', 'Computational efficiency', 'Generalizability'],
            'application': ['Domain-specific performance', 'User satisfaction', 'Real-world deployment success'],
            'theoretical': ['Theoretical soundness', 'Mathematical rigor', 'Proof completeness'],
            'interdisciplinary': ['Cross-domain validation', 'Collaboration success', 'Knowledge transfer'],
            'temporal': ['Historical accuracy', 'Modern applicability', 'Temporal generalization']
        }
        
        metrics.extend(type_metrics.get(research_type, ['Research impact', 'Publication success']))
        
        # Add impact-based metrics
        if impact_potential == 'high':
            metrics.extend(['Citation impact', 'Industry adoption', 'Follow-up research'])
        
        return metrics

    def _generate_strategic_recommendations(self, directions: List[Dict[str, Any]]) -> List[str]:
        """Generate strategic recommendations based on research directions.
        
        Args:
            directions: Research directions
            
        Returns:
            List of strategic recommendations
        """
        recommendations = []
        
        if not directions:
            return ["No research directions available for strategic analysis"]
        
        # Analyze overall portfolio
        high_confidence = [d for d in directions if d.get('confidence', 0) >= 0.8]
        novel_directions = [d for d in directions if d.get('novelty_score', 0) >= 0.8]
        
        if high_confidence:
            recommendations.append(f"Prioritize {len(high_confidence)} high-confidence directions for immediate implementation")
        
        if novel_directions:
            recommendations.append(f"Pursue {len(novel_directions)} highly novel directions for breakthrough potential")
        
        # Resource recommendations
        research_types = [d.get('research_type', 'general') for d in directions]
        if research_types.count('methodological') > 1:
            recommendations.append("Consider coordinated methodological development across multiple directions")
        
        if research_types.count('application') > 1:
            recommendations.append("Explore synergies between application-focused directions")
        
        # Timeline recommendations
        short_term_count = sum(1 for d in directions if 'months' in self._estimate_timeline(d))
        if short_term_count >= 2:
            recommendations.append("Balance short-term and long-term research objectives")
        
        return recommendations[:5]  # Limit to top 5 recommendations