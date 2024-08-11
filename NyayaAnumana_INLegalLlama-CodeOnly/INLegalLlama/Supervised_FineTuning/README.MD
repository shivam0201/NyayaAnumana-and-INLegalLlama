# Supervised FineTuning
Following the continued pre-training phase, we aim to perform supervised fine-tuning on the pre-trained model for two distinct downstream tasks:
1. **Binary Judgment Prediction**: Classify judgments as either Accepted or Rejected.
2. **Judgment Prediction with Explanation**: Predict the judgment and provide a corresponding reasoning or explanation.

To achieve this, the Supervised Fine-Tuning phase is divided into two specific sub-phases:

1. **Prediction_Only Supervised Fine-Tuning**: Focuses on predicting judgments as Accepted or Rejected.
2. **Prediction_Explanation Supervised Fine-Tuning**: Aims to predict judgments and generate a rationale or explanation for the predicted outcome.

## Folder Structure:
1. **Prediction_Explanation**:
   - Includes the code required to run the Supervised Fine-Tuning (SFT) for the Prediction & Explanation task.Please update datapaths of train and validation data based on the path in which these data files are downloaded.
2. **Prediction_Only**: 
   - Includes the code required to run the Supervised Fine-Tuning (SFT) for the Prediction task alone.Please update datapaths of train and validation data based on the path in which these data files are downloaded.
   
3. **Train_Val_Data_Preparation(Optional,see the Note point)** : 
This sub-folder contains custom Python scripts designed for two primary functions:
- **Data Splitting**: It includes scripts to divide the provided PREDEX_TRAIN dataset into training and validation sets.
- **Prompt Formatting**: The scripts also automate the formatting of prompt inputs for model training, streamlining the preparation of data for effective fine-tuning.

Please refer to the individual subfolders for detailed instructions and code for each fine-tuning phase.
### Note
The data links for performing Supervised Fine-Tuning (SFT) for the two downstream tasks—Prediction and Prediction+Explanation—are already mentioned in their respective sections. You may ignore this if you wish to proceed with the provided data. However, if you want to customize the number of tokens from the input (case description) and output (Official Reasoning) based on your `max_seq_length` parameter and available resources, you can use this data for that purpose.

