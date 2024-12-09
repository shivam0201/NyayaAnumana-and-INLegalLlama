# Inference Evaluation on PREDEX TEST Dataset
The final models, derived from both the Continued PreTraining and Supervised Fine-Tuning phases of INLegalLlama, are assessed for their inference capabilities over a choosen dataset (PREDEX/ILDC).Ensure that you download the datasets using below links and specify the correct file paths in the code where the data is required.
- [PREDEX_TEST](https://huggingface.co/datasets/L-NLProc/PredEx/resolve/main/test.csv)
- ILDC Expert Data.The dataset can be obtained by filling this [google form](https://docs.google.com/forms/d/e/1FAIpQLSf2A90ZaeZ2zc29nlhGDm8PUISRWpjCLf1TIj1YqV1hmDipPw/viewform). After filling this form, you will get access to the dataset link in a week.Please note that data is free to use for academic research and commercial usage of data is not allowed.

## Inference Types
Four different types of inferences are performed:
1. **CPT Model Inferences:**
   - Inference for Prediction Only Task (over CPT model)
   - Inference for Prediction + Explanation Task (over CPT model)
2. **Task-Specific SFT Model Inferences:**
   - Inference for Prediction Only Task (over SFT model trained for only prediction) 
   - Inference for Prediction + Explanation Task (over SFT model trained for prediction & explanation)
   

## Folder Structure
To manage these inferences, the following sub-folders are organized as:
1. **Inference_Codes:**
   - Contains Python scripts to perform inferences using the final models on the PREDEX_TEST dataset.
   - Scripts are organized by downstream task:
     * Prediction Only: Includes CPT and SFT Inferences.
     * Prediction + Explanation: Includes CPT and SFT Inferences.
2. **Inference_evaluation_metrics_calculation:**
   - Contains scripts to calculate evaluation metrics based on the results in the Inference_Results folder.
   - Metrics include accuracy, precision,f1 score,etc for evaluating prediction part of Inference and ROUGE scores, BLEU scores, etc for evaluating explanation part of Inference.
   - Scripts are organized by downstream task:
      * Prediction Only: Includes CPT and SFT Inference Evaluations.
      * Prediction + Explanation: Includes CPT and SFT Inference Evaluations.


### Note: 
The results obtained from executing the inference scripts found in the Inference_Codes folder can be accessed in the "Results" folder present in the main directory.
