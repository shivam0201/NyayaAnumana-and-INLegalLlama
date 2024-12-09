## Folder Structure
   - Contains Python scripts to perform inferences using the final models on the PREDEX_TEST dataset(present in [PREDEX_TEST.csv](https://huggingface.co/datasets/L-NLProc/PredEx/resolve/main/test.csv)).
   - Scripts are organized by downstream task:
      * **Prediction_Only**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction Only task) Inference.
      * **Prediction + Explanation**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction and Explanation task) Inference.
