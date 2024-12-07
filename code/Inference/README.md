# Inference Evaluation
The final models, derived from both the Continued PreTraining and Supervised Fine-Tuning phases of INLegalLlama, are assessed for their inference capabilities. Evaluation is performed using two distinct datasets:
- [PREDEX_TEST](https://paperswithcode.com/dataset/ildc)
- [ILDC Expert Data](https://paperswithcode.com/dataset/ildc)

These datasets are evaluated using the same evaluation code, which can be found in the relevant links given above. You can download the datasets from the provided links above. Once downloaded, update the dataset path in the code to point to the location of the dataset you wish to test.

How to Run Evaluation
1. Download the desired dataset (PREDEX_TEST or ILDC Expert Data) using the links provided.
2. Modify the dataset path in the evaluation scripts to match the location of the dataset you want to test.
3. Run the evaluation script to assess the model's performance on the selected dataset.
