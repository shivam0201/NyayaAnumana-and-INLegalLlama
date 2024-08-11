# Inference_evaluation_metrics_calculation
   - Contains scripts to calculate evaluation metrics based on the results in the Inference_Results folder.
   - Metrics include accuracy, precision,f1 score,etc for evaluating prediction part of Inference and ROUGE scores, BLEU scores, etc for evaluating explanation part of Inference.
   - Scripts are organized by downstream task:
      * **Prediction_Only**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction Only task)  Inference Evaluations.
      * **Prediction + Explanation**: Includes **CPT**(inference over CPT trained model) and **SFT**(inference over SFT model trained for Prediction and Explanation task) Inference Evaluations.
   - Get paths to files obtained by executing Inference Codes over Data.The results of the inferences performed on the models trained in this project can be accessed via the following links:

1. CPT Model Inferences:
   - [Inference for Prediction Only Task (over CPT model)](https://drive.google.com/file/d/1vPC3pTp7Xn1oW9uSe44apwaQPNVTplEr/view?usp=sharing)
   - [Inference for Prediction + Explanation (Task over CPT model)](https://drive.google.com/file/d/1QCboaGuxPkpPLdCEe0dvCaSTP_CjayB0/view?usp=sharing)
2. Task-Specific SFT Model Inferences:
   - [Inference for Prediction Only Task (over SFT model trained for only prediction)](https://drive.google.com/file/d/1OH_U9LCNQrDVJBxvfGaX-I2p0wc_mPjD/view?usp=sharing)
   - [Inference for Prediction + Explanation Task (over SFT model trained for prediction & explanation)](https://drive.google.com/file/d/1iDf3yoWpuuVEtaKEqKTU8xO7HdlNaxGx/view?usp=sharing)
