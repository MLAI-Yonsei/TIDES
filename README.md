# TIDES: Technical Information Discovery and Extraction System

## Abstract
Addressing the challenges in QA for specific technical domains requires identifying relevant portions of extensive documents and generating answers based on this focused content. Traditional pre-trained LLMs often struggle with domain-specific terminology, while finef-tuned LLMs demand substantial computational resources. To overcome these limitations, we propose TIDES , Technical Information Distillation and Extraction System.

TIDES is a training-free approach that combines traditional TF-IDF techniques with prompt-based LLMs in a hybrid process, effectively addressing complex technical questions. It uses TF-IDF to identify and prioritize domain-specific terms that are less common in other documents and LLMs to refine the candidate pool by focusing on the most relevant segments in documents through multiple stages. Our approach enhances the precision and efficiency of QA systems in technical contexts without extensive LLM retraining.



![Figure1](figure1.png)

*Figure 1: Overview of the TIDES workflow. The process begins with TF-IDF analysis to identify relevant content in technical documents. Non-relevant documents are discarded, and the remaining documents are segmented into paragraphs. These paragraphs undergo associative selection to further filter out non-relevant content. In the rationale generation phase, key evidence is extracted from the relevant paragraphs. Finally, systematic composition integrates the extracted evidence into a coherent and concise answer to the technical question.*
