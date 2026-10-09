# 🏥 Multi-Class Chronic Disease & Multi-Morbidity Classification Pipeline

An end-to-end, leak-free machine learning system designed to predict single and comorbid chronic conditions across 8 imbalanced diagnostic classes using clinical biomarkers and demographic risk factors.

---

## 📌 Project Overview

Most public benchmarks on multi-morbidity datasets inflate diagnostic scores by training on direct proxy columns (e.g., predicting hypertension while including blood pressure status as a feature). 

This project implements a rigorous, production-ready pipeline that eliminates data leakage and shortcut learning, prioritizing **Macro F1** and clinical validity over superficial accuracy.

---

## 🎯 Target Space (8 Diagnostic Cohorts)

The model performs multi-class classification across 8 distinct physiological profiles:

| Class Code | Clinical Condition | Description |
| :--- | :--- | :--- |
| `N` | Normal / Healthy | Baseline population with no diagnosed conditions |
| `HY` | Hypertension Only | High blood pressure without metabolic involvement |
| `DI_HY` | Diabetes + Hypertension | Dual metabolic/cardiovascular comorbidity |
| `DI` | Diabetes Only | Elevated glycemic markers without hypertension |
| `HT` | Heart Disease Only | Isolated cardiac pathology |
| `DI_HT` | Diabetes + Heart Disease | Metabolic dysregulation with cardiac involvement |
| `HT_HY` | Heart Disease + Hypertension | Dual cardiovascular disease presentation |
| `DI_HT_HY` | Triple Comorbidity | Rare, complex presentation across all three axes |

---

## 🛡️ Leakage Control & Architectural Integrity

1. **Explicit Target Leakage Removal:** Dropped `high_blood_pressure`, `diabetes`, and `heart_disease`. Including these features allows tree models to trivially partition target classes without learning physiological relationships from blood panels.
2. **Metadata Elimination:** Dropped `source_dataset` and redundant features like `age_normalized` to prevent shortcut learning based on cohort collection provenance.
3. **No Lookahead Bias:** Preprocessing transformers (`SimpleImputer`, `RobustScaler`, `OneHotEncoder`) were fit strictly on the training partition and serialized together with the classifier.

---

## 📊 Benchmark & Evaluation Results

All models were evaluated on an unseen held-out stratified test set (56,197 samples) with class-frequency balancing enabled:

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Fit Time |
| :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting (Tuned)** | **68.71%** | **0.3819** | **0.6869** | ~5.1s |
| **Random Forest (Max Depth 16)** | 67.68% | 0.3798 | 0.6777 | ~5.7s |
| **LightGBM (Baseline)** | 76.27% | 0.3566 | 0.7363 | ~4.3s |
| **XGBoost (Baseline)** | 76.33% | 0.3544 | 0.7365 | ~5.8s |

> **Evaluation Insight:** Standard baseline models achieved high raw accuracy (~76.3%) by ignoring the rare ~0.3% tail classes. Incorporating balanced class penalization sacrificed majority accuracy to achieve clinically meaningful sensitivity on minority cardiac comorbidity cases.

---

## ⚙️ Final Model Architecture

The deployed estimator is an optimized **`HistGradientBoostingClassifier`** tuned via 3-Fold `StratifiedKFold` `RandomizedSearchCV`:

* **Objective:** `scoring='f1_macro'`
* `class_weight`: `'balanced'`
* `learning_rate`: `0.1`
* `max_iter`: `150`
* `max_leaf_nodes`: `63`
* `min_samples_leaf`: `20`
* `l2_regularization`: `0.1`

---

## 🚀 Running the Streamlit App Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/rohansapra7781/chronic-disease-multiclass-classification.git](https://github.com/rohansapra7781/chronic-disease-multiclass-classification.git)
cd chronic-disease-multiclass-classification
