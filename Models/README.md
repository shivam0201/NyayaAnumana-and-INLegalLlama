# INLegalLlama and Transformer Classification Models

This repository hosts the models and resources developed as part of the INLegalLlama project for legal judgment prediction & explanation tasks and the Transformer Classification Models for legal judgment prediction task.

## 1. INLegalLlama Models
The INLegalLlama project integrates Large Language Models (LLMs) with the NyayaAnumana and PREDEX datasets to advance legal judgment prediction and explanation capabilities. This involves two key phases:

### 1.1 Continued PreTraining (CPT)
**Objective**: Enhance the base LLama2 model by continuing pretraining on a subset of the NyayaAnumana dataset to instill domain-specific knowledge and improve comprehension of legal texts.<br/>
**Model Access**: Models produced in this phase can be accessed here.

### 1.2 Supervised FineTuning (SFT)
**Objective**: Fine-tune the CPT-trained model on the PREDEX dataset for the respective downstream tasks:
 - Judgment Prediction Only
 - Judgment Prediction + Rationale Explanation
<br/>
**Model Variants**:
 - Prediction-Only Model: Optimized for predicting legal case outcomes.
 - Prediction + Explanation Model: Optimized for predicting outcomes and providing detailed rationale explanations.
<br/>
**Model Access**:
 - Prediction-Only Model: Access here.
 - Prediction + Explanation Model: Access here.

## 2. Transformer Classification Models
We have trained various transformer-based models (InLegalBERT, InCaseLaw, XLNet) for two distinct legal judgment classification tasks:

### 2.1 Binary Classification Task
**Objective**: Classify cases as either:
   - Accepted (Label 1 / Class 1)
   - Rejected (Label 0 / Class 0)
### 2.2 Ternary Classification Task
**Objective**: Classify cases into one of three categories:
   - Rejected (Label 0 / Class 0): All judgments in the case are rejected.
   - Accepted (Label 1 / Class 1): All judgments in the case are accepted.
   - Multi-Label (Label 2 / Class 2): Mixed judgments, with some accepted and some rejected.

### 2.3 Model Access
 - InCaseLaw
 - InLegalBERT
 - XLNet

### 2.4 Folder Structure
Each model folder is organized into the following subdirectories:

 - **Binary**: Contains models fine-tuned for binary classification tasks. This is further divided into:<br/>
    - Single: Models trained exclusively on binary-labeled datasets, where:
        - Label 0: Rejected cases
        - Label 1: Accepted cases
   <br/>
    - Multi: Models trained on an enhanced binary dataset derived from the ternary classification dataset. In this setup, multi-label cases (Label 2) are merged with the Accepted label (Label 1). These models are experimental and optional.
 - **Ternary**: Contains models fine-tuned for ternary classification tasks, categorizing cases into:

    - Label 0: Rejected cases
    - Label 1: Accepted cases
    - Label 2: Multi-label cases (mixed outcomes, with some judgments accepted and others rejected).
