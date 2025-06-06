# INLegalLlama
The INLegalLlama integrates Large Language Models (LLMs) with the NyayaAnumana and PREDEX datasets to enhance legal judgment prediction and explanation capabilities. 
This encompasses two main phases for working with LLMs:
1. Continued PreTraining
2. Supervised FineTuning
<br/>

1. **Continued PreTraining**: We begin by continuing the pretraining of the base LLAMA2 model on a subset of the NyayaAnumana dataset. This phase aims to enrich the model's understanding of legal text and domain-specific knowledge.
2. **Supervised FineTuning**: The model produced from the CPT phase undergoes supervised fine-tuning on the PREDEX dataset. This step is focused on optimizing the model for predicting judgments and providing coherent explanations in downstream tasks.

## Folder Structure:
1. **Continued_PreTraining:** This folder includes the code and dataset required for continued pre-training of the model. It encompasses all the resources necessary to run the pre-training phase.

2. **Supervised_FineTuning:** Includes the code and data for conducting supervised fine-tuning on our two downstream tasks: Prediction and Prediction with Explanation. This folder details the process for fine-tuning the models for specific tasks related to judgment prediction.

### Important Note:
Before running any code in the current folder, please ensure you:
1. Review Comments in the Code: Carefully follow the comments embedded within the code to understand the required modifications.
2. Complete All Necessary Fields: Make sure all placeholders and blanks in the code are filled in with the appropriate values, such as file paths and configuration settings(specifically those which are enclosed in "### ... ###" and text is capitalized).
3. Check the README Instructions: Consult the README file for any additional setup details or configuration instructions that may be necessary for successful execution.

By thoroughly reviewing the comments and the README, you will ensure that the code is correctly set up and ready to run. 
