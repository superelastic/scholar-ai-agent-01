"""Helper utilities for testing."""

import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def create_sample_pdf(filename: str = None) -> str:
    """Create a sample academic PDF for testing.
    
    Args:
        filename: Optional filename. If not provided, creates a temp file.
        
    Returns:
        Path to the created PDF file
    """
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix='.pdf', prefix='sample_paper_')
        os.close(fd)
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title and authors
    c.setFont("Times-Bold", 16)
    c.drawCentredString(width/2, height - inch, "Deep Learning for Natural Language Processing:")
    c.drawCentredString(width/2, height - 1.3*inch, "A Comprehensive Survey")
    
    c.setFont("Times-Roman", 12)
    c.drawCentredString(width/2, height - 1.8*inch, "John Doe¹, Jane Smith², Robert Johnson¹")
    c.setFont("Times-Roman", 10)
    c.drawCentredString(width/2, height - 2.1*inch, "¹University of Example, ²Institute of Technology")
    c.drawCentredString(width/2, height - 2.3*inch, f"{datetime.now().year}")
    
    # Abstract
    c.setFont("Times-Bold", 12)
    c.drawString(inch, height - 3*inch, "Abstract")
    
    c.setFont("Times-Roman", 11)
    abstract_text = """This paper presents a comprehensive survey of deep learning techniques applied to natural language 
processing (NLP). We review the evolution from traditional methods to modern transformer-based architectures, 
discussing key innovations such as attention mechanisms, pre-training strategies, and transfer learning. 
Our analysis covers major breakthroughs including BERT, GPT, and their variants, examining their impact 
on various NLP tasks including text classification, named entity recognition, and machine translation."""
    
    y_position = height - 3.3*inch
    for line in abstract_text.split('\n'):
        c.drawString(inch, y_position, line.strip())
        y_position -= 0.2*inch
    
    # Keywords
    c.setFont("Times-Bold", 11)
    c.drawString(inch, y_position - 0.2*inch, "Keywords:")
    c.setFont("Times-Roman", 11)
    c.drawString(2*inch, y_position - 0.2*inch, "deep learning, natural language processing, transformers, BERT, GPT")
    
    # Introduction
    c.setFont("Times-Bold", 12)
    y_position -= 0.6*inch
    c.drawString(inch, y_position, "1. Introduction")
    
    c.setFont("Times-Roman", 11)
    intro_text = """Natural language processing has undergone a revolutionary transformation with the advent of deep learning. 
Traditional rule-based and statistical methods have given way to neural approaches that can automatically 
learn complex patterns from large-scale data. This paradigm shift has led to unprecedented improvements 
in performance across virtually all NLP tasks."""
    
    y_position -= 0.3*inch
    for line in intro_text.split('\n'):
        c.drawString(inch, y_position, line.strip())
        y_position -= 0.2*inch
    
    # Add new page for more content
    c.showPage()
    
    # Methodology
    c.setFont("Times-Bold", 12)
    c.drawString(inch, height - inch, "2. Methodology")
    
    c.setFont("Times-Roman", 11)
    method_text = """We conducted a systematic review of deep learning literature in NLP from 2013 to 2023. 
Our analysis includes: (1) Architecture evolution from RNNs to Transformers, (2) Pre-training objectives 
and strategies, (3) Fine-tuning approaches for downstream tasks, and (4) Evaluation metrics and benchmarks."""
    
    y_position = height - 1.3*inch
    for line in method_text.split('\n'):
        c.drawString(inch, y_position, line.strip())
        y_position -= 0.2*inch
    
    # Results
    c.setFont("Times-Bold", 12)
    y_position -= 0.4*inch
    c.drawString(inch, y_position, "3. Results")
    
    c.setFont("Times-Roman", 11)
    results_text = """Our analysis reveals several key trends: (1) Transformer-based models consistently outperform 
previous architectures, (2) Pre-training on large unlabeled corpora significantly improves downstream 
performance, (3) Model size and training data scale show strong correlation with performance gains."""
    
    y_position -= 0.3*inch
    for line in results_text.split('\n'):
        c.drawString(inch, y_position, line.strip())
        y_position -= 0.2*inch
    
    # Conclusion
    c.setFont("Times-Bold", 12)
    y_position -= 0.4*inch
    c.drawString(inch, y_position, "4. Conclusion")
    
    c.setFont("Times-Roman", 11)
    conclusion_text = """Deep learning has fundamentally transformed NLP, with transformer architectures establishing 
new state-of-the-art results across all major tasks. Future research directions include improving 
computational efficiency, handling low-resource languages, and developing more interpretable models."""
    
    y_position -= 0.3*inch
    for line in conclusion_text.split('\n'):
        c.drawString(inch, y_position, line.strip())
        y_position -= 0.2*inch
    
    # References
    c.setFont("Times-Bold", 12)
    y_position -= 0.6*inch
    c.drawString(inch, y_position, "References")
    
    c.setFont("Times-Roman", 10)
    references = [
        "[1] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.",
        "[2] Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL.",
        "[3] Brown, T., et al. (2020). Language models are few-shot learners. NeurIPS.",
        "[4] Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv.",
        "[5] Raffel, C., et al. (2020). Exploring the limits of transfer learning with T5. JMLR."
    ]
    
    y_position -= 0.3*inch
    for ref in references:
        c.drawString(inch, y_position, ref)
        y_position -= 0.15*inch
    
    # Save the PDF
    c.save()
    
    return filename