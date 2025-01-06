# NyayaAnumana_INLegalLlama
### NyayaAnumana &amp; INLegalLlama: The Largest Indian Legal Judgment Prediction Dataset and Specialized Language Model for Enhanced Decision Analysis

The integration of artificial intelligence (AI) in legal judgment prediction (LJP) has the potential to transform the legal landscape, particularly in jurisdictions like India, where a significant backlog of cases burdens the legal system. This paper introduces _NyayaAnumana_, the largest and most diverse corpus of Indian legal cases compiled for LJP, encompassing a total of 7,02,945 preprocessed cases. _NyayaAnumana_, which combines the words **"Nyay" (judgment)** and **"Anuman" (prediction or inference)** respectively for most major Indian languages, includes a wide range of cases from the Supreme Court, High Courts, Tribunal Courts, District Courts, and Daily Orders and, thus, provides unparalleled diversity and coverage. Our dataset surpasses existing datasets like PredEx and ILDC, offering a comprehensive foundation for advanced AI research in the legal domain. 

In addition to the dataset, we present _INLegalLlama_, a domain-specific generative large language model (LLM) tailored to the intricacies of the Indian legal system. It is developed through a two-phase training approach over a base LLaMa model. First, Indian legal documents are injected using continual pretraining. Second, task-specific supervised finetuning is done. This method allows the model to achieve a deeper understanding of legal contexts. 

Our experiments demonstrate that incorporating diverse court data significantly boosts model accuracy, achieving approximately 90% F1-score in prediction tasks. _INLegalLlama_ not only improves prediction accuracy but also offers comprehensible explanations, addressing the need for explainability in AI-assisted legal decisions. 

![InLegalLlama](images/InLegalLlama.png)
![NyayaAnumana](images/NyayaAnumana.png)

## Citation

If you use this project in your research or other work, please cite it as follows:
## Citation
If you use our method or models, please cite [our paper]():
```
@article{nigam2024nyayaanumana,
  title={NyayaAnumana \& INLegalLlama: The Largest Indian Legal Judgment Prediction Dataset and Specialized Language Model for Enhanced Decision Analysis},
  author={Nigam, Shubham Kumar and Patnaik, Balaramamahanthi Deepak and Mishra, Shivam and Shallum, Noel and Ghosh, Kripabandhu and Bhattacharya, Arnab},
  journal={arXiv preprint arXiv:2412.08385},
  year={2024}
}

