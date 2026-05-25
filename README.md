# Breast Cancer XAI Experiment

A hands-on experiment demonstrating core Explainable AI (XAI) techniques on a medical dataset.

## Objective
To move beyond model accuracy and interrogate *why* a model makes its predictions, and whether those reasons align with established medical knowledge.

## Dataset
**Breast Cancer Wisconsin Diagnostic Dataset** from scikit-learn.
30 features describing cell nucleus characteristics from breast mass biopsies.

## Models Trained
- **Logistic Regression** (Inherently Interpretable Glass Box)
- **Random Forest** (High-Performance Black Box)

## XAI Techniques Applied
- **Permutation Importance:** Global feature importance by scrambling columns.
- **SHAP (SHapley Additive exPlanations):** Game-theoretic local and global explanations.
  - Summary Plot
  - Waterfall Plot
  - Force Plot
- **Model Coefficients:** Direct interpretability of Logistic Regression.

## Key Findings
- "Worst concave points" was the most dominant feature across all methods.
- Model predictions are driven by nuclear size (radius, perimeter) and shape irregularity (concavity), which are pathologically sound indicators of malignancy.
- The models independently learned medically validated diagnostic criteria, building significant trust for a hypothetical deployment.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/[Your-Username]/breast-cancer-xai-experiment.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the experiment:
   ```bash
   python xai_experiment.py


### Experiment Report

**Experiment Title:** An Explainability Audit of Machine Learning Models for Breast Cancer Diagnosis
**Author:**  Nur Nafis Fuad
**Date:** 25th May 2026

---

**1. Model Performance Summary**

Two models were trained: a glass-box Logistic Regression (LR) and a black-box Random Forest (RF). Both achieved high classification accuracy on the held-out test set, with the RF performing marginally better, confirming the standard accuracy-interpretability trade-off.

---

**2. Key Finding: Which Features Dominate?**

A multi-method triangulation was performed using Permutation Importance, SHAP values, and Logistic Regression coefficients. A feature was considered a "robust signal" if it appeared as a top contributor across at least two independent methods.

**The dominant feature across all methods was "worst concave points."** This feature measures the severity of indentations on the nucleus of the worst-presenting cells in the biopsy.

Other consistently top-ranked features included "mean concave points," "worst perimeter," "mean radius," and "worst texture."

---

**3. Critical Question: Do Explanations Make Medical Sense?**

This is the cardinal question of an XAI audit. An accurate model that relies on nonsensical features is useless and dangerous.

- **Finding A: "Worst concave points strongly increased malignancy probability."**
    - **Medical Validity:** Concavity quantifies nuclear contour irregularity. In pathology, highly irregular, indented nuclear membranes are a classic hallmark of malignant cells. The model has independently identified a well-established cytological criterion for cancer.
- **Finding B: "Worst perimeter and mean radius are major positive contributors."**
    - **Medical Validity:** Uncontrolled proliferation is the fundamental characteristic of cancer. This manifests as larger-than-normal cell nuclei. The model's reliance on direct measurements of nuclear size (perimeter, radius) is precisely what a trained pathologist looks for under a microscope.
- **Finding C: "Worst texture is a significant predictor."**
    - **Medical Validity:** Texture, in this context, refers to the variation in the gray-scale staining pattern of the nuclear chromatin. Malignant nuclei typically display a coarse, clumped, and irregular "texture" compared to the fine, even texture of benign nuclei. Again, the model's feature attribution is medically sound.

**Conclusion:** The model is not a "Clever Hans" relying on spurious correlations in the data (like image artifacts or patient ID). It is making its high-stakes diagnostic predictions based on the same medically meaningful features of nuclear size, shape, and texture that a human expert would use. This XAI audit builds a legally and ethically defensible case for trust in the model's reasoning.

## Author
Nur Nafis Fuad  ([Email](mailto:nnfuad01@gmail.com))

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.