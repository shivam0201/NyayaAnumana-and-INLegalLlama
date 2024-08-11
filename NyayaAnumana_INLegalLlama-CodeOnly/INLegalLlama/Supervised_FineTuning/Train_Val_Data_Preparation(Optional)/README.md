# Dataset Preparation For Supervised Fine-Tuning
This folder contains data and custom Python scripts designed for generating training and validation datasets from the PREDEX TRAIN dataset, as well as for formatting input prompts specific to each downstream task (prediction only and prediction + explanation).

## Input Prompt Formatting
Input prompt formatting relies on instructions specified in the instruction_dataset.csv file. This file includes instructions/prompts for both "prediction only" and "prediction + explanation" tasks. The dataset preparation scripts randomly select prompt from this file according to the specific downstream task at hand and combine them with the input and output data to create training inputs for the model.

## Folder Structure:

1. **Prediction_Explanation**: Contains the Python script for preparing the training and validation data, as well as formatting the prompts required for model training on the Prediction+Explanation task.

2. **Prediction_Only**: Contains the Python script for preparing the training and validation data, as well as formatting the prompts required for model training on the Prediction-only task.

### Usage Instructions
Each subfolder within the current directory contains code for generating datasets specific to the corresponding downstream task (as indicated by the folder names). Before executing the scripts, ensure you:

- Update the paths for PREDEX_TRAIN.csv and instruction_dataset.csv in the code.
- *(Optional)* Modify the output filenames for the training and validation datasets if needed.<br/>
By following these steps, you will prepare the datasets necessary for training the model according to the specified downstream tasks.

