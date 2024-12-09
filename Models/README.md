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
**Model Variants**:
 - Prediction-Only Model: Optimized for predicting legal case outcomes.
 - Prediction + Explanation Model: Optimized for predicting outcomes and providing detailed rationale explanations.
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

 - **Binary**: Contains models fine-tuned for binary classification tasks. This is further divided into:
    - Single: Models trained exclusively on binary-labeled datasets, where:
        - Label 0: Rejected cases
        - Label 1: Accepted cases
    - Multi: Models trained on an enhanced binary dataset derived from the ternary classification dataset. In this setup, multi-label cases (Label 2) are merged with the Accepted label (Label 1). These models are experimental and optional.
 - **Ternary**: Contains models fine-tuned for ternary classification tasks, categorizing cases into:

    - Label 0: Rejected cases
    - Label 1: Accepted cases
    - Label 2: Multi-label cases (mixed outcomes, with some judgments accepted and others rejected).

















This sb-directory involves the models that we buuilt in this project.
1. INLegalLlama Model: The INLegalLlama integrates Large Language Models (LLMs) with the NyayaAnumana and PREDEX datasets to enhance legal judgment prediction and explanation capabilities. 
This encompasses two main phases for working with LLMs:
 1. Continued PreTraining
 2. Supervised FineTuning

So,tHis consists of models obtained at the end of each phase
 1. **Continued PreTraining**: We begin by continuing the pretraining of the base LLAMA2 model on a subset of the NyayaAnumana dataset. This phase aims to enrich the model's understanding of legal text and domain-specific knowledge.
   - Model can be accessed here:   
 2. **Supervised FineTuning**: The model produced from the CPT phase undergoes supervised fine-tuning on the PREDEX dataset. This step is focused on optimizing the model for predicting judgments and providing coherent explanations in downstream tasks.We have done 2 main variants for each downstream task one being prediction only and one being prediction along with a rationale exaplantion.
   - Models can be accessed here:
      - Prediction Only Model
      - Prediction + Explanation Model
2. Transformer Classification Models:We train various models (InLegalBERT, InCaseLawBERT, XLNet) for two types of classification tasks:

 - Binary Classification Task: Fine-tuning models to classify a given case as either accepted (label-1/class-1) or rejected (label-0/class-0).

 - Ternary Classification Task: Fine-tuning models to classify a given case into one of three classes:

    * Rejected (label-0/class-0): Cases where single or multiple judgments are present, all are rejected.
    * Accepted (label-1/class-1): Cases where single or multiple judgments are present, all are accepted.
    * Multi-label (label-2/class-2): Cases where multiple judgments are present, with some accepted and some rejected.
       
 - The corresponding Models can be accessed here:
   1. InCaseLaw
   2. InLegalBert
   3. XLNet
 - Each Model folder has 2 sub-folders namely binary and ternary,each represnting the type of classification task they are intended for.
 - Also, for binary classification tasks, you can use multi folder. These are intended for experimental purposes, merging Multi-label (label-2/class-2) cases into the Accepted label from the ternary classification task. You may choose to use or disregard these datasets based on your experimental needs.
