# Code Folder Structure
This repository contains the source codes for the project. The codes are organized into the following folders:
1. INLegalLlama
2. Inference
3. Prediction_classification
---
### 1. INLegalLlama
The INLegalLlama integrates Large Language Models (LLMs) with the NyayaAnumana and PREDEX datasets to enhance legal judgment prediction and explanation capabilities. 
This encompasses two main phases for working with LLMs:
1. Continued PreTraining
2. Supervised FineTuning
---
### 2. Inference
This folder is dedicated to evaluating the final models obtained from both the Continued Pre-Training and Supervised Fine-Tuning phases on inference tasks using two distinct datasets: PREDEX_TEST and ILDC Expert data.

---
### 3. Prediction_classification

This folder contains the code required to train various models (InLegalBERT, InCaseLawBERT, XLNet) for two types of classification tasks:

1. **Binary Classification Task**: Fine-tuning models to classify a given case as either accepted (label-1/class-1) or rejected (label-0/class-0).

2. **Ternary Classification Task**: Fine-tuning models to classify a given case into one of three classes:
   - *Rejected* (label-0/class-0): Cases where single or multiple judgments are present, all are rejected.
   - *Accepted* (label-1/class-1): Cases where single or multiple judgments are present, all are accepted.
   - *Multi-label* (label-2/class-2): Cases where multiple judgments are present, with some accepted and some rejected.
