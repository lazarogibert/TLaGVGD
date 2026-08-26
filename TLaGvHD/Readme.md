# Leveraging Transfer Learning for Predicting Acute Graft-versus-Host Disease in Stem Cell Transplantation

## Description
This repository contains the code and dataset information for our machine learning framework designed to avoid raw data sharing and predict acute Graft-versus-Host Disease (aGvHD) following allogeneic hematopoietic stem cell transplantation. Due to data privacy regulations and healthcare records heterogeneity, centralized data collection is often limited. Our framework addresses this by applying a transfer learning strategy across decentralized data environments. 

A multi-input neural network is used: a source model's encoder is trained locally using shared attributes, and its learned representations are frozen and transferred to enhance a target model on a different dataset. To demonstrate the robustness and generalizability of this approach, **the framework is evaluated bidirectionally** (training on Dataset A -> transferring to Dataset B, and vice versa). This strategy achieves strong predictive performance without sharing sensitive patient-level data.

## Key Contributions & Conclusion
Currently, there is no reliable method to predict whether a patient will develop aGvHD following transplantation. The acquisition of sufficiently large training datasets is often infeasible due to low incidence rates, privacy constraints, and healthcare record heterogeneity. This study addresses these limitations with the following contributions:

* **Institutions Collaboration:** Demonstrates that medical institutions can engage in transfer learning to build advanced predictive models by sharing *pre-trained weights* rather than highly-sensitive, raw patient data.
* **Handling Heterogeneous Data:** Proves that this transfer learning strategy is effective even when the source and target datasets do not share the exact same clinical features.
* **Robust Performance:** Yields strong discriminative outcomes (high AUROC) despite notable differences in patient diagnoses, donor types, and graft manipulation protocols between the CIBMTR, UCHW and BMTCH cohorts.
* **Clinical Trust via Explainability:** Incorporates extensive XAI analysis to identify key clinical attributes influencing predictions, supporting transparent automatic clinical decision-making. 
* **Future Directions:** Establishes a foundation for future work aimed at predicting specific grades of aGvHD and validating the framework across broader patient demographics.

## Dataset Information
This study utilizes three publicly available datasets. Shared attributes across both datasets were assigned uniform naming conventions and standardized for consistency. Depending on the experimental phase, each dataset acts as either the Source (base model) or the Target (transfer model).

* **UCHW Dataset:** Pediatric patients from the University Children's Hospital in Würzburg (Germany) who underwent HSCT between Jan 2005 and Dec 2015. 
    * **Filtering:** Autologous transplants were excluded (as they do not lead to aGvHD). To prevent data leakage, only pre-transplantation attributes were used.
    * **Final Cohort:** 126 patients (9 predictive features). 
    * **aGvHD Prevalence:** 32.8%.
* **BMTCH Dataset:** Pediatric patients admitted to Wrocław Medical University (Poland) from 2000 to 2008, receiving unmodified allogeneic HSCT from unrelated donors.
    * **Filtering:** Only pre-transplant attributes were selected. Cases with missing data were excluded.
    * **Final Cohort:** 169 patients (15 predictive features).
    * **aGvHD Prevalence:** 59.9%.

* **CIBMTR Dataset:** Comprises data on pediatric patients collected by the Center for International Blood and Marrow Transplant Research from 2008 to 2016, receiving their first allogeneic HSCT from haploidentical, and unrelated matched donors.
    * **Filtering:** Only pre-transplant attributes were selected. Cases with missing data were excluded.
    * **Final Cohort:** 583 patients (16 predictive features).
    * **aGvHD Prevalence:** 63.8%.

## Methodology 
The primary objective is predicting the occurrence of aGvHD grades II-IV within 100 days post-transplantation. 
1.  **Data Splitting:** Data was partitioned independently for each dataset using stratified splitting to preserve class distributions: ~70% Training, ~15% Validation, and 15% Test.
2.  **Modeling Framework:** * **Transfer Learning:** The pre-trained encoder from the base model is frozen. 
    * **Multi-Input Network:** A dual-branch neural network processes the existing shared features through the frozen encoder, while new institution-specific features are processed through a tunable dense network. The representations are concatenated and passed to a final classification head.
3.  **Hyperparameter Tuning:** `keras_tuner.RandomSearch` was utilized to optimize the network architecture (layers, units, dropout rates) and learning rate, maximizing the validation F1-score.
4.  **Post-hoc Explainability (XAI):** To ensure clinical transparency, multiple interpretability frameworks were applied to identify key factors associated with aGvHD development:
    * **SHAP (SHapley Additive exPlanations):** Used for global feature importance (Summary Plots) and local predictions (Force and Waterfall Plots).
    * **PDP (Partial Dependence Plots) & ALE (Accumulated Local Effects):** Utilized to isolate and visualize the marginal effect of individual clinical features on the predicted probability of aGvHD.
    * **DALEX:** Applied for secondary validation of global variable importance profiles.

## Materials & Methods: Assessment Metrics
* **AUROC (Area Under the Receiver Operating Characteristic Curve):** AUROC was selected as the primary assessment metric because it provides a robust measure of the model's discriminative ability across all classification thresholds. This is particularly justified given the varying class distributions between datasets (32.8%, 59.9% and 63.8%); AUROC remains unbiased by these prevalence differences. 
* **PR-AUC (Area Under the Precision-Recall Curve):** Given the imbalanced nature of the target variable, PR-AUC is utilized to accurately reflect the model's performance on the positive (minority) class.
* **F1-Score:** The F1-score is used to balance Precision and Recall. To accommodate varying operational thresholds, an optimal decision threshold is dynamically calculated on the validation set to maximize the F1-score before evaluating the test set.
* **Statistical Significance:** Dunnett's test was utilized to assess the significance of the improvements in AUROC against baseline models (p < 0.01).

## Code Information
The repository is provided as interactive Jupyter Notebooks (`.ipynb`) and is structured into two distinct pipelines to ensure complete reproducibility of the bidirectional experiments. 

**Pipeline 1: BMTCH (Source) -> UCHW (Target)**
* `Base_model_BMTCH_UCHW.ipynb`: Ingests the BMTCH dataset, tunes hyperparameters, constructs the base network with an `encoder` layer, and exports the artifacts.
* `Transfer_Learning_BMTCH_UCHW.ipynb`: Transfers learned representations from the BMTCH base model to the UCHW target dataset.
* `Explainability_Transfer_Learning_BMTCH_UCHW.ipynb`: Evaluates the transferred model on the test set and executes the XAI pipeline to export high-resolution vector graphics (.pdf).

**Pipeline 2: UCHW (Source) -> BMTCH (Target)**
* `Base_model_UCHW_BMTCH.ipynb`: Ingests the UCHW dataset, tunes hyperparameters, constructs the base network with an `encoder` layer, and exports the artifacts.
* `Transfer_Learning_UCHW_BMTCH.ipynb`: Transfers learned representations from the UCHW base model to the BMTCH target dataset.
* `Explainability_Transfer_Learning_UCHW_BMTCH.ipynb`: Evaluates the target model and generates XAI visualizations for this direction.

**Pipeline 2: UCHW (Source) -> BMTCH (Target)**
* `Base_model_BMTCH_CIBMTR.ipynb`: Ingests the BMTCH dataset, tunes hyperparameters, constructs the base network with an `encoder` layer, and exports the artifacts.
* `Transfer_Learning_BMTCH_CIBMTR.ipynb`: Transfers learned representations from the BMTCH base model to the CIBMTR target dataset.
* `Explainability_Transfer_Learning_BMTCH_CIBMTR.ipynb`: Evaluates the target model and generates XAI visualizations for this direction.


## Requirements
The code was developed in Python 3. The primary dependencies include:
* `tensorflow` (>= 2.x)
* `keras-tuner`
* `scikit-learn`
* `pandas`, `numpy`, `scipy`
* `joblib`, `dill`
* `openpyxl` (for dataset ingestion)
* `matplotlib`, `seaborn`, `plotly` (>= 5.24.1), `kaleido` (>= 0.2.1)
* **XAI Libraries:** `shap`, `PyALE`, `dalex`

## Usage Instructions (Google Colab)
This code is optimized to run seamlessly in Google Colaboratory. 

1. Open Google Colab and upload the desired notebook (`.ipynb` file).
2. Upload the datasets (`BMTCH.xlsx`, `UCHW.xlsx` or `CIBMTR.xlsx`) to the default `/content/sample_data/` directory via the left-hand file explorer in Colab.
3. Execute the cells within the notebooks sequentially. To reproduce a full pipeline, ensure you run the notebooks in the correct phase order:
   * **Phase 1:** Run the `Base_model...` notebook.
   * **Phase 2:** Run the `Transfer_Learning...` notebook.
   * **Phase 3:** Run the `Explainability...` notebook.
4. *Note: Ensure you download the generated `.pdf` figures and `.keras` model artifacts from the Colab file explorer to your local machine before the runtime disconnects.*

## Computing Infrastructure
* **Environment:** Google Colaboratory (Ubuntu-based cloud environment).
* **Hardware Compute:** Standard Colab CPU allocation (~12-16 GB RAM). The codebase is framework-agnostic and executes efficiently on standard processors without requiring hardware acceleration (GPU/TPU).

## Citations
If you use this code or methodology in your research, please cite our paper:
> [L. Gibert-García, R. Guerra, J. Dunstan, J. Palma, A.J. Soto, A.G. Maguitman, and C. Chesñevar. Leveraging Transfer Learning for Predicting Acute Graft-versus-Host Disease in Stem Cell Transplantation. Under review 2026.]


**Dataset References:**

* **UCHW Dataset:**
  > Hierlmeier, S., Eyrich, M., Wölfl, M., & Wiegering, V. (2018). Early and late complications following hematopoietic stem cell transplantation in pediatric patients – A retrospective analysis over 11 years. *PLoS One*, 13(10), e0204914. https://doi.org/10.1371/journal.pone.0204914

* **BMTCH Dataset:**
  > Gudyś, A., Sikora, M., & Wróbel, Ł.  Bone marrow transplant - children. https://www.kaggle.com/datasets/adamgudys/bone-marrow-transplant-children
  > 
  > Sikora, M., Wróbel, Ł., & Gudyś, A. (2019). GuideR: A guided separate-and-conquer rule learning in classification, regression, and survival settings. *Knowledge-Based Systems*, 173, 1-14. https://doi.org/10.1016/j.knosys.2019.02.019
  
  * **CIBMTR Dataset:**
    >Dandoy, C. E., Davies, S. M., Ahn, K. W., He, Y., Kolb, A. E., Levine, J., Bo-Subait, S., Abdel-Azim, H.,Bhatt, N., Chewing, J., Gadalla, S., Gloude, N., Hayashi, R., Lalefar, N. R., Law, J., MacMillan, M.,OBrien, T., Prestidge, T., Sharma, A., Shaw, P., Winestone, L., and Eapen, M. (2020). Comparison of total body irradiation versus non-total body irradiation containing regimens for de novo acute myeloid leukemia in children. *Haematologica*, 106(7):18391845. https://doi.org/10.3324/haematol.2020.249458
  
  > ## License
## License
This project operates under a split license to accurately protect both the software and the written content:

* **Software / Source Code:** The Python scripts and Jupyter Notebooks (`.py`, `.ipynb`) are licensed under the **MIT License**.
* **Documentation & Media:** The README, documentation, and generated figures/images (`.md`, `.pdf`, `.png`) are licensed under a **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

  >