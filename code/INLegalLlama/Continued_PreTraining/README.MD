# Continued PreTraining
This sub-directory contains the code necessary to perform continued pre-training on the LLAMA2 base model using a subset of the NyayaAnumana dataset. The code is present in the code folder and the following instructions detail the data and setup required to execute the pre-training:

## Folder Structure:
1. **code:** This folder contains the scripts necessary for performing continued pre-training on the LLAMA2 base model. It includes all the code required to execute the pre-training process using the data provided in the data folder.

### 1. Training data: 
- This dataset includes 138,321 rows(a subset of the NyayaAnumana Dataset) composed of:
  * 100,000 randomly selected rows from the HighCourt dataset (1800 to 2019) in the NyayaAnumana Multi dataset.
  * All rows(38,321 rows) from the Supreme Court dataset (1800 to 2019) in the NyayaAnumana Multi dataset.
- **Download**: [Training Data](https://huggingface.co/datasets/L-NLProc/InLegalLlama-training-data/tree/main/For%20CPT)
### 2. Validation data: 
- This dataset includes 12,239 rows composed of:
  * 10,000 randomly selected rows from the HighCourt dataset (2020 to 2024) in the NyayaAnumana Multi dataset.
  * All rows from the Supreme Court dataset (2020 to 2024) in the NyayaAnumana Multi dataset.
- **Download**: [Validation Data](https://huggingface.co/datasets/L-NLProc/InLegalLlama-training-data/tree/main/For%20CPT)

To execute the continued pre-training, please follow the steps outlined in the provided notebook/python script. Ensure that you update the file paths in the code as specified in the notebook.

### Note:
For access to the complete NyayaAnumana dataset, which extends beyond the subset used for training (selected due to resource constraints), please refer to the NyayaAnumana folder in main directory.
