## Folder Structure
   - Contains Python scripts to perform inferences using the final models on the ILDC Expert annotated dataset(present in [ILDC_54.csv](https://drive.google.com/file/d/1aY4s7f_5rA6MvdDek0q5Gqq3qsdbycXN/view)).
   - Scripts are organized by downstream task:
      * **Prediction_Only**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction Only task) Inference.
      * **Prediction + Explanation**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction and Explanation task) Inference.
