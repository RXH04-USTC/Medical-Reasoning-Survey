# Medical Reasoning with Large Language Models: A Systematic Review and Evaluation

This repository contains the source code and evaluation framework for the paper **"Medical Reasoning with Large Language Models: A Systematic Review and Evaluation"** 
## Overview

This work presents a comprehensive review of medical reasoning with Large Language Models (LLMs). We:

1. **Systematize medical reasoning approaches** into 7 major technical routes spanning training-based and training-free methods
2. **Conduct unified cross-benchmark evaluation** of representative medical reasoning models under consistent experimental settings
3. **Introduce MR-Bench**, a clinically grounded benchmark derived from real-world hospital data (MIMIC-IV)
4. **Expose critical gaps** between exam-level performance and authentic clinical decision-making tasks

## Key Contributions

### 1. Comprehensive Taxonomy of Medical Reasoning Methods

**Training-Based Approaches:**
- Continued Pretraining (Domain-adaptive pretraining)
- Supervised Fine-Tuning (Instruction tuning with Chain-of-Thought)
- Reinforcement Learning (Policy-gradient and preference-based methods)

**Training-Free Approaches:**
- Prompt Engineering (Clinical CoT, diagnostic reasoning prompts)
- Test-Time Strategies (Self-consistency, ensemble refinement, inference-time scaling)
- Retrieval-Augmented Generation (Evidence grounding and adaptive retrieval)
- Agentic Reasoning Pipelines (Structured workflows and multi-agent collaboration)

### 2. Cross-Benchmark Evaluation

We evaluate representative medical LLMs across 8 diverse benchmarks:
- **[MedQA](https://huggingface.co/datasets/bigbio/med_qa)** 
- **[MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa)** 
- **[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)** 
- **[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa)** 
- **[JMED](https://huggingface.co/datasets/jdh-algo/JMED)** 
- **[ReDis-QA](https://huggingface.co/datasets/guan-wang/ReDis-QA)** 
- **[MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQAL)** 
- **[MMLU-Pro (Medical Subset)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)** 

### 3. MR-Bench: Real-World Clinical Benchmark

MR-Bench operationalizes medical reasoning as safety-critical clinical decision-making with:
- **1,000 authentic clinical cases per task** from MIMIC-IV
- **Temporal generalization** across 4 distinct time periods
- **Two decision tasks:**
  - Medication Imputation: Infer appropriate medications given partial prescription information
  - Procedure Selection: Translate clinical evidence into appropriate interventional decisions
- **Clinically grounded distractors** with drug-drug interaction risks and contraindications




**Note**: This paper is currently under review. Citation information will be updated upon publication.



## Future Directions

1. **Clinically Grounded Evaluation**: Move beyond static benchmarks toward dynamic, decision-driven evaluation
2. **Active Interaction**: Enable models to actively acquire targeted evidence and manage uncertainty
3. **Trustworthy Reasoning**: Develop systems that ground reasoning in authoritative evidence and manage uncertainty
4. **Safety and Ethics**: Address hallucination, bias, and ensure alignment with clinical practice



## Disclaimer

This repository is for research purposes only. The code and models should not be used for clinical decision-making without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical advice.
