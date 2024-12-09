## Dataset links

1. **input_data**: The complete training dataset required for Supervised Fine-Tuning (SFT). This dataset will be split into training and validation sets in the respective task-specific folders.
	- [PREDEX_TRAIN.csv](https://huggingface.co/datasets/L-NLProc/PredEx)
2. **instruction_sets**: It provides the instructions to be used as prompts during the training of each downstream task (prediction only and prediction + explanation).
	-[instruction_dataset.csv](https://huggingface.co/datasets/L-NLProc/PredEx_Instruction_sets/blob/main/instruction_decision.csv)

### Usage Instructions
Each subfolder within the current directory contains code for generating datasets specific to the corresponding downstream task (as indicated by the folder names). Before executing the scripts, ensure you:

- Update the paths for PREDEX_TRAIN.csv and instruction_dataset.csv in the code.
- *(Optional)* Modify the output filenames for the training and validation datasets if needed.<br/>
By following these steps, you will prepare the datasets necessary for training the model according to the specified downstream tasks.

