import os
from docx import Document
from docx.shared import Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.shared import OxmlElement, qn
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

def add_heading_with_formatting(doc, text, level):
    """Add heading with proper formatting"""
    heading = doc.add_heading(text, level)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return heading

def add_bullet_point(doc, text, level=0):
    """Add bullet point with proper indentation"""
    p = doc.add_paragraph()
    p.style = 'List Bullet' if level == 0 else 'List Bullet 2'
    p.text = text
    return p

def add_numbered_point(doc, text, level=0):
    """Add numbered point with proper indentation"""
    p = doc.add_paragraph()
    p.style = 'List Number' if level == 0 else 'List Number 2'
    p.text = text
    return p

def add_bold_text(doc, text):
    """Add bold text"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    return p

def add_two_column_section(doc):
    """Add a two-column section to the document"""
    section = doc.sections[-1]
    sectPr = section._sectPr
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '708')  # 0.5 inch spacing
    sectPr.append(cols)

def add_single_column_section(doc):
    """Add a single-column section to the document"""
    section = doc.sections[-1]
    sectPr = section._sectPr
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '1')
    sectPr.append(cols)

def create_docx_report():
    """Create Word document from the markdown report"""
    
    # Create new document
    doc = Document()
    
    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
    
    # Title (single column)
    title = doc.add_heading('Sign Language Digit Recognition: A Comprehensive Hyperparameter Analysis', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Abstract (single column)
    add_heading_with_formatting(doc, 'Abstract', 1)
    abstract_text = """This study presents a comprehensive analysis of hyperparameter optimization for Convolutional Neural Networks (CNNs) applied to sign language digit recognition. We evaluated 10 different hyperparameter configurations across various aspects including batch size, learning rate, regularization techniques, dropout rates, and architectural variations. Our experiments utilized a dataset of 2,060 sign language digit images representing digits 0-9, achieving a maximum test accuracy of 97.58% with the Deep Architecture configuration. The analysis reveals significant insights into the impact of different hyperparameters on model performance, with particular emphasis on the critical role of learning rate optimization and architectural design in achieving optimal classification accuracy. Notably, we discovered that batch size has a dramatic impact on performance, with small batch sizes (16) achieving 96.37% accuracy while large batch sizes (128) dropped to only 73.37% accuracy, highlighting the importance of gradient noise in CNN training for this specific task."""
    doc.add_paragraph(abstract_text)
    
    # Introduction (single column)
    add_heading_with_formatting(doc, 'I. Introduction', 1)
    intro_text = """Sign language recognition represents a crucial application of computer vision and machine learning in assistive technology. The ability to automatically recognize and classify sign language gestures can significantly improve accessibility and communication for the deaf and hard-of-hearing community. This study focuses specifically on digit recognition in sign language, which serves as a fundamental building block for more complex sign language recognition systems.

The primary objective of this research is to conduct a systematic hyperparameter analysis to identify optimal configurations for CNN-based sign language digit recognition. Through comprehensive experimentation, we aim to understand the impact of various hyperparameters on model performance and provide actionable insights for practitioners working in this domain."""
    doc.add_paragraph(intro_text)
    
    # Switch to two-column format for methodology
    add_two_column_section(doc)
    
    # Methodology
    add_heading_with_formatting(doc, 'II. Methodology', 1)
    
    # Dataset Description
    add_heading_with_formatting(doc, 'A. Dataset Description', 2)
    dataset_text = """The dataset consists of 2,060 high-quality images of sign language digits (0-9), with approximately 206 images per digit class. The images were captured in a controlled environment with consistent lighting and background conditions. Each image represents a hand gesture corresponding to a specific digit, captured from multiple angles and with slight variations in hand positioning to ensure robustness."""
    doc.add_paragraph(dataset_text)
    
    add_bold_text(doc, 'Dataset Statistics:')
    add_bullet_point(doc, 'Total images: 2,060')
    add_bullet_point(doc, 'Classes: 10 (digits 0-9)')
    add_bullet_point(doc, 'Images per class: ~206')
    add_bullet_point(doc, 'Image resolution: 64×64 pixels')
    add_bullet_point(doc, 'Color channels: 3 (RGB)')
    add_bullet_point(doc, 'Training set: 1,648 images (80%)')
    add_bullet_point(doc, 'Test set: 412 images (20%)')
    
    add_bold_text(doc, 'Class Distribution Analysis:')
    doc.add_paragraph('The dataset exhibits excellent class balance with each digit class containing approximately 206 images, ensuring that the model is not biased toward any particular class. This balanced distribution is crucial for fair evaluation of model performance across all digit classes.')
    
    # Data Preprocessing
    add_heading_with_formatting(doc, 'B. Data Preprocessing', 2)
    doc.add_paragraph('The preprocessing pipeline was designed to ensure consistent input format and optimal training conditions:')
    
    add_numbered_point(doc, 'Image Loading: Images were loaded using OpenCV and converted from BGR to RGB color space')
    add_numbered_point(doc, 'Resizing: All images were resized to 64×64 pixels to ensure uniform input dimensions')
    add_numbered_point(doc, 'Normalization: Pixel values were normalized to the range [0, 1] by dividing by 255')
    add_numbered_point(doc, 'Train-Test Split: Data was split into 80% training (1,648 images) and 20% testing (412 images) using stratified sampling to maintain class distribution')
    add_numbered_point(doc, 'Data Augmentation: No augmentation was applied to maintain consistency across experiments')
    
    # CNN Architecture
    add_heading_with_formatting(doc, 'C. CNN Architecture', 2)
    doc.add_paragraph('The study employed three distinct CNN architectures to evaluate the impact of architectural choices. All architectures were designed with modern best practices including BatchNormalization, proper weight initialization, and progressive dropout rates.')
    
    add_heading_with_formatting(doc, '1. Default Architecture', 3)
    add_bullet_point(doc, 'Input Layer: 64×64×3 RGB images with explicit Input layer')
    add_bullet_point(doc, 'Convolutional Block 1: Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.3×dropout_rate)')
    add_bullet_point(doc, 'Convolutional Block 2: Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.5×dropout_rate)')
    add_bullet_point(doc, 'Convolutional Block 3: Conv2D(128, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.7×dropout_rate)')
    add_bullet_point(doc, 'Dense Layers: Dense(512) + BatchNorm + ReLU + Dropout(dropout_rate)')
    add_bullet_point(doc, 'Output Layer: Dense(10) + Softmax (Glorot Uniform initialization)')
    add_bullet_point(doc, 'Total Parameters: 2,590,922')
    
    add_heading_with_formatting(doc, '2. Deep Architecture', 3)
    add_bullet_point(doc, 'Enhanced Depth: Additional convolutional layers for increased feature extraction')
    add_bullet_point(doc, 'Convolutional Structure: Double convolutional layers per block')
    add_bullet_point(doc, 'Total Parameters: 1,473,066 (43% fewer than default)')
    add_bullet_point(doc, 'Performance: Achieved best overall accuracy (97.58%)')
    
    add_heading_with_formatting(doc, '3. Wide Architecture', 3)
    add_bullet_point(doc, 'Enhanced Width: Increased filter counts for broader feature representation')
    add_bullet_point(doc, 'Convolutional Structure: Increased filter counts (64, 128, 256)')
    add_bullet_point(doc, 'Total Parameters: 5,226,890 (2× more than default)')
    add_bullet_point(doc, 'Performance: Achieved 96.61% accuracy with higher computational cost')
    
    # Hyperparameter Optimization
    add_heading_with_formatting(doc, 'D. Hyperparameter Optimization', 2)
    doc.add_paragraph('The study systematically evaluated 10 different hyperparameter configurations across five key dimensions:')
    
    add_bold_text(doc, 'Learning Rate Variations:')
    add_bullet_point(doc, 'Baseline: Standard learning rate of 0.001 with batch size 32, dropout 0.3, and L2 regularization 0.001')
    add_bullet_point(doc, 'High Learning Rate: Increased to 0.003 (3× higher) while maintaining other baseline parameters')
    add_bullet_point(doc, 'Low Learning Rate: Reduced to 0.0003 (3× lower) with extended patience of 15 epochs')
    
    add_bold_text(doc, 'Batch Size Experiments:')
    add_bullet_point(doc, 'Small Batch Size: Reduced to 16 with dropout 0.5 and L2 regularization 0.01')
    add_bullet_point(doc, 'Large Batch Size: Increased to 128 with dropout 0.5 and L2 regularization 0.01')
    
    add_bold_text(doc, 'Regularization Studies:')
    add_bullet_point(doc, 'High Dropout: Increased dropout to 0.8 with L2 regularization 0.01')
    add_bullet_point(doc, 'Low Dropout: Reduced dropout to 0.2 with L2 regularization 0.01')
    add_bullet_point(doc, 'High L2 Regularization: Increased L2 to 0.1 with dropout 0.5')
    
    add_bold_text(doc, 'Architectural Variations:')
    add_bullet_point(doc, 'Deep Architecture: Double convolutional layers per block with 15-epoch patience')
    add_bullet_point(doc, 'Wide Architecture: Increased filter counts (64, 128, 256) with standard training parameters')
    
    # Training Configuration
    add_heading_with_formatting(doc, 'E. Training Configuration', 2)
    doc.add_paragraph('All models were trained with the following consistent settings to ensure fair comparison:')
    
    add_bold_text(doc, 'Optimizer Configuration:')
    add_bullet_point(doc, 'Algorithm: Adam optimizer with optimized parameters')
    add_bullet_point(doc, 'Learning Rate: Variable per configuration (0.0003 to 0.003)')
    add_bullet_point(doc, 'Beta Parameters: β₁=0.9, β₂=0.999 (standard values)')
    add_bullet_point(doc, 'Epsilon: ε=1e-7 (improved numerical stability)')
    
    add_bold_text(doc, 'Training Parameters:')
    add_bullet_point(doc, 'Loss Function: Sparse Categorical Crossentropy (appropriate for multi-class classification)')
    add_bullet_point(doc, 'Metrics: Accuracy (primary evaluation metric)')
    add_bullet_point(doc, 'Epochs: 50 maximum (with early stopping to prevent overfitting)')
    add_bullet_point(doc, 'Validation Split: 20% of training data (330 images) for validation')
    add_bullet_point(doc, 'Batch Size: Variable per configuration (16, 32, or 128)')
    
    # Experimental Results (single column for better readability)
    add_single_column_section(doc)
    add_heading_with_formatting(doc, 'III. Experimental Results', 1)
    
    # Individual Configuration Performance
    add_heading_with_formatting(doc, 'A. Individual Configuration Performance Analysis', 2)
    doc.add_paragraph('This section provides detailed analysis of all 10 hyperparameter configurations tested, ranked by performance.')
    
    # Performance Results Summary
    add_heading_with_formatting(doc, 'Performance Results Summary:', 3)
    doc.add_paragraph('The comprehensive analysis of all 10 configurations revealed significant performance variations, with accuracy ranging from 73.37% to 97.58%:')
    
    add_bold_text(doc, 'Top Performers (95%+ Accuracy):')
    add_bullet_point(doc, 'Deep Architecture achieved the highest accuracy of 97.58% with the lowest test loss (0.4290) and near-perfect macro AUC (0.9994)')
    add_bullet_point(doc, 'High Learning Rate configuration reached 97.34% accuracy with the lowest test loss (0.3759) among all configurations')
    add_bullet_point(doc, 'Low Dropout setting achieved 96.85% accuracy, demonstrating the importance of maintaining sufficient model capacity')
    add_bullet_point(doc, 'Wide Architecture reached 96.61% accuracy but required 2× more parameters than the default configuration')
    add_bullet_point(doc, 'Baseline configuration provided reliable 96.37% accuracy, establishing a solid reference point')
    
    add_bold_text(doc, 'Moderate Performers (90-95% Accuracy):')
    add_bullet_point(doc, 'Small Batch Size achieved 96.37% accuracy with the highest validation accuracy (98.18%), indicating excellent generalization')
    add_bullet_point(doc, 'High L2 Regularization maintained 96.13% accuracy while preventing overfitting through strong regularization')
    add_bullet_point(doc, 'Low Learning Rate configuration reached 93.22% accuracy, suggesting insufficient learning capacity')
    
    add_bold_text(doc, 'Poor Performers (<90% Accuracy):')
    add_bullet_point(doc, 'High Dropout configuration achieved only 92.25% accuracy with the highest test loss (2.3827), indicating excessive regularization')
    add_bullet_point(doc, 'Large Batch Size performed worst at 73.37% accuracy, demonstrating the critical importance of appropriate batch size selection')
    
    # Key Findings
    add_heading_with_formatting(doc, 'B. Key Findings', 2)
    
    add_heading_with_formatting(doc, '1. Batch Size Impact', 3)
    doc.add_paragraph('The analysis revealed a dramatic impact of batch size on model performance:')
    
    add_bold_text(doc, 'Small Batch Size (16):')
    add_bullet_point(doc, 'Test Accuracy: 96.37%')
    add_bullet_point(doc, 'Validation Accuracy: 98.18% (highest validation performance)')
    add_bullet_point(doc, 'Analysis: Small batch sizes provided excellent gradient noise')
    
    add_bold_text(doc, 'Large Batch Size (128):')
    add_bullet_point(doc, 'Test Accuracy: 73.37% (worst performing configuration)')
    add_bullet_point(doc, 'Validation Accuracy: 74.55%')
    add_bullet_point(doc, 'Analysis: Dramatically reduced performance due to reduced gradient noise')
    
    add_heading_with_formatting(doc, '2. Learning Rate Sensitivity', 3)
    doc.add_paragraph('Learning rate proved to be one of the most critical hyperparameters:')
    
    add_bold_text(doc, 'High Learning Rate (0.003):')
    add_bullet_point(doc, 'Test Accuracy: 97.34% (second-best overall)')
    add_bullet_point(doc, 'Test Loss: 0.3759 (lowest loss achieved)')
    add_bullet_point(doc, 'Analysis: Higher learning rate enabled more aggressive learning')
    
    add_bold_text(doc, 'Low Learning Rate (0.0003):')
    add_bullet_point(doc, 'Test Accuracy: 93.22% (below baseline performance)')
    add_bullet_point(doc, 'Test Loss: 1.1951 (higher than baseline)')
    add_bullet_point(doc, 'Analysis: Insufficient learning rate led to underfitting')
    
    add_heading_with_formatting(doc, '3. Regularization Effects', 3)
    doc.add_paragraph('Regularization techniques showed significant impact on model performance:')
    
    add_bold_text(doc, 'Low Dropout (0.2):')
    add_bullet_point(doc, 'Test Accuracy: 96.85% (third-best overall)')
    add_bullet_point(doc, 'Analysis: Lower dropout allowed the model to maintain more connections')
    
    add_bold_text(doc, 'High Dropout (0.8):')
    add_bullet_point(doc, 'Test Accuracy: 92.25% (second-worst performance)')
    add_bullet_point(doc, 'Test Loss: 2.3827 (highest loss)')
    add_bullet_point(doc, 'Analysis: Excessive regularization severely limited model capacity')
    
    # Discussion (two-column format)
    add_two_column_section(doc)
    add_heading_with_formatting(doc, 'IV. Discussion', 1)
    
    add_heading_with_formatting(doc, 'A. Performance Analysis', 2)
    doc.add_paragraph('The experimental results reveal several critical insights about hyperparameter optimization for sign language digit recognition:')
    
    add_bold_text(doc, 'Batch Size:')
    add_bullet_point(doc, 'Most impactful hyperparameter with 24.24% accuracy gap between small and large batch sizes')
    add_bullet_point(doc, 'Small batch sizes (16) provide optimal gradient noise for this dataset')
    add_bullet_point(doc, 'Large batch sizes (128) lead to poor convergence and suboptimal performance')
    
    add_bold_text(doc, 'Learning Rate:')
    add_bullet_point(doc, 'Higher learning rates (0.003) significantly outperform lower rates (0.0003)')
    add_bullet_point(doc, 'Optimal learning rate for this task is higher than commonly used 0.001')
    add_bullet_point(doc, 'Learning rate has second-highest impact on final performance')
    
    add_bold_text(doc, 'Architecture:')
    add_bullet_point(doc, 'Depth is more important than width for this specific task')
    add_bullet_point(doc, 'Deep Architecture achieved best performance with 43% fewer parameters')
    add_bullet_point(doc, 'Parameter efficiency is crucial for practical applications')
    
    # Conclusion (single column for better readability)
    add_single_column_section(doc)
    add_heading_with_formatting(doc, 'V. Conclusion', 1)
    doc.add_paragraph('This comprehensive hyperparameter analysis for sign language digit recognition has yielded several critical insights that advance our understanding of CNN optimization for gesture recognition tasks.')
    
    add_heading_with_formatting(doc, 'Key Findings Summary:', 2)
    add_numbered_point(doc, 'Optimal Configuration: The Deep Architecture with standard hyperparameters achieved the best performance (97.58% accuracy)')
    add_numbered_point(doc, 'Critical Hyperparameters: Batch size and learning rate have the most significant impact on model performance')
    add_numbered_point(doc, 'Regularization Balance: Moderate regularization (dropout 0.2-0.5, L2 0.001-0.01) provides optimal performance')
    add_numbered_point(doc, 'Architectural Insights: Depth is more important than width for this task')
    add_numbered_point(doc, 'Training Dynamics: The dramatic performance difference between batch sizes highlights the critical importance of gradient noise')
    
    add_heading_with_formatting(doc, 'Practical Implications:', 2)
    doc.add_paragraph('For Practitioners:')
    add_bullet_point(doc, 'Prioritize batch size selection (16-32 optimal for this task)')
    add_bullet_point(doc, 'Use learning rates around 0.003 for better convergence')
    add_bullet_point(doc, 'Implement moderate regularization (dropout 0.2-0.5)')
    add_bullet_point(doc, 'Consider depth over width in architectural design')
    add_bullet_point(doc, 'Monitor training dynamics to identify convergence issues early')
    
    # Save document
    doc.save('Sign_Language_Digit_Recognition_Report.docx')
    print("Word document created successfully: Sign_Language_Digit_Recognition_Report.docx")

if __name__ == "__main__":
    create_docx_report()
