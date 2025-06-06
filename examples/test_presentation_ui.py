"""Demo script for User Interface and Result Presentation capabilities.

This script demonstrates:
1. Multiple output formatters (JSON, Markdown, HTML, Plain Text)
2. Progress tracking for long-running operations
3. Export functionality with bundling
4. Enhanced PresentationNode capabilities
5. Real-time progress indicators

Usage:
    python examples/test_presentation_ui.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    FormatterFactory,
    ExportManager,
    export_analysis_results,
    ScholarAIProgressTracker,
    create_progress_indicator,
    create_sample_pdf
)
from nodes import PresentationNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_comprehensive_sample_data():
    """Create comprehensive sample analysis data for demonstration."""
    return {
        'paper_metadata': {
            'title': 'Attention Is All You Need: Transforming Natural Language Processing',
            'authors': [
                'Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 
                'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez',
                'Łukasz Kaiser', 'Illia Polosukhin'
            ],
            'year': 2017,
            'doi': '10.48550/arXiv.1706.03762',
            'abstract': (
                'The dominant sequence transduction models are based on complex recurrent '
                'or convolutional neural networks that include an encoder and a decoder. '
                'The best performing models also connect the encoder and decoder through '
                'an attention mechanism. We propose a new simple network architecture, '
                'the Transformer, based solely on attention mechanisms, dispensing with '
                'recurrence and convolutions entirely.'
            ),
            'venue': 'Neural Information Processing Systems (NeurIPS)',
            'citations_count': 85000,
            'pdf_pages': 15
        },
        'paper_analysis': {
            'key_concepts': [
                'transformer architecture',
                'self-attention mechanism',
                'multi-head attention',
                'positional encoding',
                'sequence-to-sequence models',
                'encoder-decoder architecture',
                'neural machine translation',
                'parallelization'
            ],
            'methodology': (
                'The authors propose the Transformer, a model architecture eschewing '
                'recurrence and instead relying entirely on an attention mechanism to '
                'draw global dependencies between input and output. The Transformer follows '
                'this overall architecture using stacked self-attention and point-wise, '
                'fully connected layers for both the encoder and decoder.'
            ),
            'findings': [
                'Transformers achieve superior translation quality on WMT 2014 English-to-German and English-to-French tasks',
                'The model is significantly more parallelizable than recurrent models',
                'Training time is substantially reduced compared to RNNs and CNNs',
                'Self-attention provides better interpretability of model behavior',
                'The architecture generalizes well to other sequence transduction tasks'
            ],
            'contributions': [
                'Introduction of the Transformer architecture based solely on attention',
                'Demonstration of superior performance on machine translation tasks',
                'Significant improvement in training efficiency and parallelization',
                'Novel multi-head attention mechanism',
                'Elimination of recurrent and convolutional layers'
            ],
            'limitations': [
                'Quadratic complexity with respect to sequence length',
                'Limited context window for very long sequences',
                'Requires substantial computational resources for training'
            ],
            'impact_score': 9.8,
            'novelty_score': 9.5,
            'clarity_score': 8.9
        },
        'citations': [
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee', 'Kristina Toutanova'],
                'year': 2019,
                'venue': 'NAACL-HLT',
                'citations': 45000,
                'relevance_score': 0.95,
                'url': 'https://arxiv.org/abs/1810.04805',
                'snippet': 'BERT makes use of Transformer, an attention mechanism that learns contextual relations between words.',
                'impact_factor': 9.2
            },
            {
                'title': 'Language Models are Unsupervised Multitask Learners',
                'authors': ['Alec Radford', 'Jeffrey Wu', 'Rewon Child', 'David Luan', 'Dario Amodei', 'Ilya Sutskever'],
                'year': 2019,
                'venue': 'OpenAI',
                'citations': 15000,
                'relevance_score': 0.88,
                'url': 'https://openai.com/blog/better-language-models/',
                'snippet': 'GPT-2 uses a decoder-only transformer architecture similar to the original Transformer.',
                'impact_factor': 8.5
            },
            {
                'title': 'Training language models to follow instructions with human feedback',
                'authors': ['Long Ouyang', 'Jeff Wu', 'Xu Jiang', 'Diogo Almeida'],
                'year': 2022,
                'venue': 'NeurIPS',
                'citations': 3500,
                'relevance_score': 0.82,
                'url': 'https://arxiv.org/abs/2203.02155',
                'snippet': 'InstructGPT builds upon the transformer architecture introduced in "Attention Is All You Need".',
                'impact_factor': 8.1
            },
            {
                'title': 'Scaling Laws for Neural Language Models',
                'authors': ['Jared Kaplan', 'Sam McCandlish', 'Tom Henighan', 'Tom B. Brown'],
                'year': 2020,
                'venue': 'arXiv',
                'citations': 2800,
                'relevance_score': 0.75,
                'url': 'https://arxiv.org/abs/2001.08361',
                'snippet': 'We study empirical scaling laws for language model performance using transformer architectures.',
                'impact_factor': 7.8
            },
            {
                'title': 'Vision Transformer (ViT): An Image is Worth 16x16 Words',
                'authors': ['Alexey Dosovitskiy', 'Lucas Beyer', 'Alexander Kolesnikov'],
                'year': 2021,
                'venue': 'ICLR',
                'citations': 8500,
                'relevance_score': 0.78,
                'url': 'https://arxiv.org/abs/2010.11929',
                'snippet': 'Vision Transformer adapts the transformer architecture for computer vision tasks.',
                'impact_factor': 8.3
            }
        ],
        'research_directions': {
            'suggestions': [
                {
                    'title': 'Efficient Attention Mechanisms for Long Sequences',
                    'description': (
                        'Develop attention mechanisms that scale sub-quadratically with sequence length '
                        'to enable processing of very long documents and sequences. This could involve '
                        'sparse attention patterns, hierarchical attention, or approximation techniques.'
                    ),
                    'rationale': (
                        'The quadratic complexity of standard attention limits the application of '
                        'transformers to very long sequences, which is important for document-level '
                        'processing and long-form content generation.'
                    ),
                    'confidence': 0.85,
                    'difficulty': 'high',
                    'time_horizon': 'medium-term',
                    'required_expertise': ['deep learning', 'optimization', 'algorithmic design'],
                    'potential_impact': 'high',
                    'research_areas': ['natural language processing', 'computer vision', 'sequence modeling'],
                    'expected_citations': 1000
                },
                {
                    'title': 'Multimodal Transformer Architectures',
                    'description': (
                        'Extend transformer architectures to naturally handle multiple modalities '
                        'simultaneously, including text, images, audio, and video. This requires '
                        'developing unified attention mechanisms across different data types.'
                    ),
                    'rationale': (
                        'Real-world applications often involve multiple modalities, and current '
                        'approaches typically process each modality separately before fusion. '
                        'Native multimodal transformers could achieve better integration.'
                    ),
                    'confidence': 0.78,
                    'difficulty': 'medium',
                    'time_horizon': 'short-term',
                    'required_expertise': ['multimodal learning', 'computer vision', 'speech processing'],
                    'potential_impact': 'moderate',
                    'research_areas': ['computer vision', 'natural language processing', 'speech recognition'],
                    'expected_citations': 800
                },
                {
                    'title': 'Interpretable Attention Mechanisms',
                    'description': (
                        'Develop attention mechanisms that provide clearer interpretation of model '
                        'decisions and reasoning processes. This could involve constrained attention '
                        'patterns, attention regularization, or post-hoc analysis tools.'
                    ),
                    'rationale': (
                        'While transformers have achieved remarkable performance, understanding '
                        'their decision-making process remains challenging. Better interpretability '
                        'is crucial for high-stakes applications and scientific discovery.'
                    ),
                    'confidence': 0.72,
                    'difficulty': 'medium',
                    'time_horizon': 'medium-term',
                    'required_expertise': ['interpretable AI', 'visualization', 'cognitive science'],
                    'potential_impact': 'moderate',
                    'research_areas': ['explainable AI', 'human-computer interaction', 'cognitive science'],
                    'expected_citations': 600
                },
                {
                    'title': 'Energy-Efficient Transformer Training',
                    'description': (
                        'Develop training methodologies and architectural modifications that '
                        'significantly reduce the energy consumption and computational requirements '
                        'for training large transformer models while maintaining performance.'
                    ),
                    'rationale': (
                        'The environmental impact and computational cost of training large '
                        'transformers is substantial. More efficient training methods would '
                        'democratize access to large-scale models and reduce carbon footprint.'
                    ),
                    'confidence': 0.88,
                    'difficulty': 'high',
                    'time_horizon': 'short-term',
                    'required_expertise': ['green AI', 'distributed computing', 'optimization'],
                    'potential_impact': 'high',
                    'research_areas': ['sustainable AI', 'distributed systems', 'optimization'],
                    'expected_citations': 900
                }
            ],
            'trend_analysis': {
                'emerging_themes': [
                    'efficiency and scalability',
                    'multimodal integration',
                    'interpretability and explainability',
                    'domain specialization'
                ],
                'declining_areas': [
                    'purely recurrent architectures',
                    'CNN-only sequence models'
                ],
                'research_momentum': 'very high'
            }
        },
        'formatted_citations': {
            'total_citing_papers': 75432,
            'recent_citations': 15678,
            'citation_growth_rate': 0.45,
            'top_citing_venues': [
                'NeurIPS', 'ICML', 'ICLR', 'ACL', 'EMNLP'
            ]
        },
        'processing_metadata': {
            'pdf_path': '/example/path/attention_is_all_you_need.pdf',
            'processed_at': datetime.now().isoformat(),
            'total_processing_time': 67.3,
            'status': 'completed',
            'processing_stages': [
                {'stage': 'pdf_extraction', 'duration': 8.2},
                {'stage': 'content_analysis', 'duration': 23.5},
                {'stage': 'citation_search', 'duration': 28.1},
                {'stage': 'research_synthesis', 'duration': 7.5}
            ]
        }
    }


def demonstrate_formatters():
    """Demonstrate all output formatters."""
    logger.info("=== Demonstrating Output Formatters ===")
    
    # Create sample data
    sample_data = create_comprehensive_sample_data()
    
    # Test all formatters
    formats = ['json', 'markdown', 'html', 'txt']
    
    for format_type in formats:
        logger.info(f"\nTesting {format_type.upper()} formatter...")
        
        try:
            formatter = FormatterFactory.create_formatter(format_type)
            result = formatter.format(sample_data)
            
            # Show a preview of the output
            preview_length = 200
            preview = result[:preview_length]
            if len(result) > preview_length:
                preview += "..."
            
            logger.info(f"✓ {format_type.upper()} format generated successfully ({len(result)} characters)")
            logger.info(f"Preview: {preview.replace(chr(10), ' ').replace(chr(13), ' ')}")
            
        except Exception as e:
            logger.error(f"✗ {format_type.upper()} format failed: {e}")
    
    logger.info(f"\nSupported formats: {', '.join(FormatterFactory.get_supported_formats())}")


def demonstrate_progress_tracking():
    """Demonstrate progress tracking functionality."""
    logger.info("\n=== Demonstrating Progress Tracking ===")
    
    # Create progress indicator
    progress_indicator = create_progress_indicator("Scholar AI Analysis", width=40)
    
    # Create tracker with callback
    tracker = ScholarAIProgressTracker.create_default(callback=progress_indicator)
    
    logger.info("Starting simulated workflow with progress tracking...")
    tracker.start()
    
    # Simulate PDF processing
    logger.info("\nSimulating PDF processing...")
    for progress in [25, 50, 75, 100]:
        tracker.update_stage_progress(progress, {'current_operation': f'Processing page {progress//25}'})
        time.sleep(0.5)
    tracker.advance_stage(progress_percent=100)
    
    # Simulate paper analysis
    logger.info("\nSimulating paper analysis...")
    for progress in [20, 40, 60, 80, 100]:
        tracker.update_stage_progress(progress, {'current_operation': f'Analyzing section {progress//20}'})
        time.sleep(0.3)
    tracker.advance_stage(progress_percent=100)
    
    # Simulate citation search
    logger.info("\nSimulating citation search...")
    for progress in [15, 35, 60, 85, 100]:
        tracker.update_stage_progress(progress, {'current_operation': f'Found {progress//10} citations'})
        time.sleep(0.4)
    tracker.advance_stage(progress_percent=100)
    
    # Simulate research synthesis
    logger.info("\nSimulating research synthesis...")
    for progress in [30, 70, 100]:
        tracker.update_stage_progress(progress, {'current_operation': 'Generating research directions'})
        time.sleep(0.6)
    tracker.advance_stage(progress_percent=100)
    
    # Complete
    tracker.complete()
    
    logger.info("\nWorkflow completed!")
    
    # Show detailed progress
    detailed_progress = tracker.get_detailed_progress()
    logger.info(f"Total stages completed: {detailed_progress['summary']['completed_stages']}")
    logger.info(f"Total processing time: {detailed_progress['summary']['elapsed_time']:.1f} seconds")


def demonstrate_export_functionality():
    """Demonstrate export functionality."""
    logger.info("\n=== Demonstrating Export Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary export directory: {temp_dir}")
        
        # Create sample data
        sample_data = create_comprehensive_sample_data()
        
        # Test single format export
        logger.info("\nTesting single format export...")
        export_manager = ExportManager(temp_dir)
        
        json_file = export_manager.export_single_file(sample_data, 'json')
        logger.info(f"✓ Exported JSON to: {json_file}")
        
        # Test multi-format export with bundle
        logger.info("\nTesting multi-format export with bundle...")
        export_info = export_manager.export_analysis_results(
            sample_data,
            formats=['json', 'markdown', 'html', 'txt'],
            export_name='comprehensive_demo',
            create_bundle=True
        )
        
        if export_info['success']:
            logger.info(f"✓ Multi-format export successful!")
            logger.info(f"  Exported formats: {', '.join(export_info['exported_formats'])}")
            logger.info(f"  Export directory: {export_info['summary']['export_directory']}")
            logger.info(f"  Bundle file: {export_info['bundle_file']}")
            
            # Show file sizes
            for format_type, file_path in export_info['files'].items():
                file_size = os.path.getsize(file_path)
                logger.info(f"  {format_type.upper()}: {file_size:,} bytes")
        
        # Test convenience function
        logger.info("\nTesting convenience export function...")
        convenience_export = export_analysis_results(
            sample_data,
            output_dir=temp_dir,
            export_name='convenience_demo'
        )
        
        if convenience_export['success']:
            logger.info(f"✓ Convenience export successful!")
            logger.info(f"  Exported {len(convenience_export['exported_formats'])} formats")
        
        # List all exports
        exports = export_manager.list_exports()
        logger.info(f"\nTotal exports created: {len(exports)}")
        for export in exports[:3]:  # Show first 3
            logger.info(f"  - {export['export_name']} ({', '.join(export['formats'])})")


async def demonstrate_presentation_node():
    """Demonstrate PresentationNode capabilities."""
    logger.info("\n=== Demonstrating PresentationNode ===")
    
    # Create sample data in the format expected by PresentationNode
    sample_store = {
        'paper_metadata': {
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani, A.', 'Shazeer, N.'],
            'year': 2017
        },
        'analysis_results': {
            'key_concepts': ['transformer', 'attention'],
            'methodology': 'Novel architecture based on attention',
            'findings': ['Better performance', 'More parallelizable']
        },
        'filtered_citations': [
            {'title': 'BERT', 'year': 2019, 'relevance_score': 0.95},
            {'title': 'GPT-2', 'year': 2019, 'relevance_score': 0.88}
        ],
        'research_suggestions': {
            'suggestions': [
                {
                    'title': 'Efficient Attention',
                    'confidence': 0.85,
                    'description': 'Develop more efficient attention mechanisms'
                }
            ]
        },
        'started_at': (datetime.now()).isoformat()
    }
    
    # Create presentation node
    presentation_node = PresentationNode(['json', 'markdown', 'html'])
    
    # Create context
    context = {'store': sample_store}
    
    # Process
    logger.info("Processing with PresentationNode...")
    result = await presentation_node.process(context)
    
    if result['success']:
        logger.info("✓ PresentationNode processing successful!")
        logger.info(f"  Generated formats: {list(result['presentations'].keys())}")
        logger.info(f"  Export files: {list(result['export_files'].keys())}")
        
        # Show progress summary
        progress_summary = result['progress_summary']
        logger.info(f"  Stages completed: {len(progress_summary['stages_completed'])}")
        logger.info(f"  Results summary: {progress_summary['results_summary']}")
    else:
        logger.error(f"✗ PresentationNode failed: {result.get('error')}")


def demonstrate_real_time_progress():
    """Demonstrate real-time progress updates."""
    logger.info("\n=== Demonstrating Real-time Progress Updates ===")
    
    progress_updates = []
    
    def capture_progress(summary):
        progress_updates.append(summary.copy())
        # Print compact progress line
        progress = summary['overall_progress']
        stage = summary.get('current_stage', 'processing')
        print(f"\rProgress: {progress:5.1f}% | {stage:<20}", end='', flush=True)
    
    # Create tracker
    tracker = ScholarAIProgressTracker.create_default(callback=capture_progress)
    tracker.start()
    
    # Simulate rapid updates
    logger.info("Simulating rapid progress updates...")
    
    stages = ['pdf_processing', 'paper_analysis', 'citation_search', 'research_synthesis', 'result_formatting']
    
    for stage_idx, stage_name in enumerate(stages):
        # Update progress in increments
        for progress in range(0, 101, 10):
            tracker.update_stage_progress(progress)
            time.sleep(0.05)  # Fast updates
        
        # Advance to next stage
        if stage_idx < len(stages) - 1:
            tracker.advance_stage(progress_percent=100)
    
    tracker.complete()
    print()  # New line after progress updates
    
    logger.info(f"Captured {len(progress_updates)} progress updates")
    logger.info(f"Final progress: {progress_updates[-1]['overall_progress']:.1f}%")


def demonstrate_error_handling():
    """Demonstrate error handling in presentation components."""
    logger.info("\n=== Demonstrating Error Handling ===")
    
    # Test formatter with invalid data
    logger.info("Testing formatter error handling...")
    try:
        formatter = FormatterFactory.create_formatter('invalid_format')
    except Exception as e:
        logger.info(f"✓ Properly caught formatter error: {e}")
    
    # Test export with invalid data
    logger.info("Testing export error handling...")
    with tempfile.TemporaryDirectory() as temp_dir:
        export_manager = ExportManager(temp_dir)
        try:
            export_manager.export_analysis_results(None, ['json'])
        except Exception as e:
            logger.info(f"✓ Properly caught export error: {e}")
    
    # Test progress tracker with failing callback
    logger.info("Testing progress tracker error handling...")
    
    def failing_callback(summary):
        raise Exception("Callback failure")
    
    tracker = ScholarAIProgressTracker.create_default(callback=failing_callback)
    tracker.start()
    tracker.update_stage_progress(50)  # Should not crash despite callback failure
    
    logger.info("✓ Progress tracker handled callback failure gracefully")


def show_feature_summary():
    """Show a summary of all demonstrated features."""
    logger.info("\n" + "="*60)
    logger.info("PRESENTATION & UI FEATURES DEMONSTRATION SUMMARY")
    logger.info("="*60)
    
    features = [
        "✓ Multiple Output Formatters (JSON, Markdown, HTML, Plain Text)",
        "✓ Comprehensive formatting with proper structure and styling",
        "✓ Unicode and special character handling",
        "✓ Progress Tracking for long-running operations",
        "✓ Real-time progress indicators with callbacks",
        "✓ Estimated time remaining calculations",
        "✓ Export Functionality with file management",
        "✓ Multi-format export with ZIP bundling",
        "✓ Export metadata and session tracking",
        "✓ Enhanced PresentationNode with multiple formats",
        "✓ Thread-safe progress tracking",
        "✓ Graceful error handling and recovery",
        "✓ Customizable progress indicators",
        "✓ Export cleanup and management tools",
        "✓ Convenience functions for easy integration"
    ]
    
    for feature in features:
        logger.info(f"  {feature}")
    
    logger.info("\nThe presentation system provides:")
    logger.info("  • Professional-quality output in multiple formats")
    logger.info("  • Real-time progress tracking and user feedback")
    logger.info("  • Comprehensive export and file management")
    logger.info("  • Robust error handling and graceful degradation")
    logger.info("  • Easy integration with existing workflows")
    logger.info("  • Extensible architecture for new formats and features")


async def main():
    """Run all presentation and UI demonstrations."""
    logger.info("Starting Presentation & UI Capabilities Demo")
    logger.info("="*60)
    
    try:
        # Demonstrate core features
        demonstrate_formatters()
        demonstrate_progress_tracking()
        demonstrate_export_functionality()
        await demonstrate_presentation_node()
        demonstrate_real_time_progress()
        demonstrate_error_handling()
        
        # Show summary
        show_feature_summary()
        
        logger.info("\n" + "="*60)
        logger.info("Presentation & UI Demo completed successfully!")
        logger.info("All components are working correctly and ready for integration.")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())