# Prediction_classification

This folder contains the code required to train various models (InLegalBERT, InCaseLawBERT, XLNet) for two types of classification tasks:

1. **Binary Classification Task**: Fine-tuning models to classify a given case as either accepted (label-1/class-1) or rejected (label-0/class-0).

2. **Ternary Classification Task**: Fine-tuning models to classify a given case into one of three classes:
   - *Rejected* (label-0/class-0): Cases where single or multiple judgments are present, all are rejected.
   - *Accepted* (label-1/class-1): Cases where single or multiple judgments are present, all are accepted.
   - *Multi-label* (label-2/class-2): Cases where multiple judgments are present, with some accepted and some rejected.

## Folder Structure

Each classification task has a dedicated folder:

1. **binary**: 
    - Contains a Python script for training models on the binary classification task. This script is integrated to support all models (InLegalBERT, InCaseLawBERT, XLNet).
    - **USAGE INSTRUCTIONS:**
	   1. The Python scripts within each task-specific folder are integrated to support all models (InLegalBERT, InCaseLawBERT, XLNet). Update the script with:
   		- `model_type` (highlighted as `###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###`)
   		    * Choose one from MODEL_CLASSES dictionary,which has keys as (InLegalBERT, InCaseLawBERT, xlnet) specified inside the code based on which model you want to use .
   		- `model_name` (highlighted as `###--> CHANGE WHAT MODEL PATH WANT HERE!!! <--###`)
		    * Give the model path based on which model you want to use
		   	1. InLegalBERT: "law-ai/InLegalBERT"
		   	2. InCaseLawBERT: "law-ai/InCaseLawBERT"
		   	3. XLNet: "xlnet/xlnet-base-cased"
	   2. Replace the CSV file names in `train_set`, `validation_set`, and `test_set` variables with the data files present in the NyayaAnumana folder, which was organized into train, dev and test .Choose appropriate csv from binary task of each of train,dev and test. You can find how to choose appropriate csv using README.md file in NyayaAnumana folder.
			  
2. **ternary**: 
    - Contains a Python script for training models on the ternary classification task. This script is integrated to support all models (InLegalBERT, InCaseLawBERT, XLNet).
    - **USAGE INSTRUCTIONS:**
	   1. The Python scripts within each task-specific folder are integrated to support all models (InLegalBERT, InCaseLawBERT, XLNet). Update the script with:
   		- `model_type` (highlighted as `###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###`)
   		    * Choose one from MODEL_CLASSES dictionary,which has keys as (InLegalBERT, InCaseLawBERT, xlnet) specified inside the code based on which model you want to use .
   		- `model_name` (highlighted as `###--> CHANGE WHAT MODEL PATH WANT HERE!!! <--###`)
		    * Give the model path based on which model you want to use
		   	1. InLegalBERT: "law-ai/InLegalBERT"
		   	2. InCaseLawBERT: "law-ai/InCaseLawBERT"
		   	3. XLNet: "xlnet/xlnet-base-cased"
	   2. Replace the CSV file names in `train_set`, `validation_set`, and `test_set` variables with the data files present in the NyayaAnumana folder, which was organized into train, dev and test .Choose appropriate csv from ternary of each of train,dev and test. You can find how to choose appropriate csv using README.md file in NyayaAnumana folder.

