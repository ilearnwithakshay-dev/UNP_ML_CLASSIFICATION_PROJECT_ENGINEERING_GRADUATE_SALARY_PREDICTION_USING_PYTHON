# Engineering Graduate Salary Prediction: A Binary Classification Study Using ML & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellowgreen.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Links & Access

### Interactive Notebook (Google Colab)

* **Engineering Graduate Salary Prediction Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10Zt1ZimdlIdvSUUqpngjwaqFu9QZy66J?usp=sharing)
  *Colab URL:* https://colab.research.google.com/drive/10Zt1ZimdlIdvSUUqpngjwaqFu9QZy66J?usp=sharing

### Project Resources & Drive Links

* **Project Presentation Link:** https://docs.google.com/presentation/d/18XUGuDe5n2AG8k4HTQdcGuW6KK_xFALV/edit?usp=drive_link&ouid=108531411860157531016&rtpof=true&sd=true
* **Dataset Access Link:** https://drive.google.com/file/d/1EFXYQi3xxT4_zzKoOGi2mXSZlNt1Z8sF/view?usp=drive_link
* **Dataset Details Link:** https://docs.google.com/spreadsheets/d/1t4MYPEgIlkVdaB0ojwSB2ODDRaUimr5x/edit?usp=drive_link&ouid=108531411860157531016&rtpof=true&sd=true

---

## Project Overview

I have done this Engineering Graduate Salary Prediction project using Python in collaboration with **United Network of Professionals (UNP)** as part of my data science internship. The steps are as follows.

The primary objective of this project is to build a predictive **binary classification** system that sorts engineering graduates into **High-salary** or **Low-salary** categories (split at the median salary of ₹3,00,000), and then benchmark multiple industry-standard machine learning algorithms against a deep Artificial Neural Network (ANN) to identify the most accurate, reliable, and deployable model for real-world talent acquisition.

The dataset encompasses **2,998 engineering graduate profiles** across demographics, academic history, aptitude test scores, domain-specific engineering module scores, and psychometric traits — all sourced from an Indian campus placement dataset.

The steps, methodology, and experimental results are presented below.

---

## Team & Institutional Affiliation

* **Academic Institution:** Bhavans Vivekananda College of Science, Humanities and Commerce
* **Program:** BSc (Honours) in Data Science — Data Science Internship Program
* **Collaborating Organization:** United Network of Professionals (UNP)
* **Project Group:** Group 6
* **Team Members:**
  * **Akhiranand Boddani**
  * **Uppaluru S V N Santhosha Roopa**
  * **Deepthi Hansika**
  * **Akshay Kumar**

---

## Dataset Description

The dataset contains **2,998 engineering graduate candidate records** (reduced to 2,997 after dropping one invalid row) across **34 original columns**, with **zero missing or null values** in the raw data.

* **Total Records (Raw):** 2,998 candidate profiles
* **Total Records (After Cleaning):** 2,897 rows × 19 features
* **Target Variable:** `SalaryLabel` — a binary label derived from the continuous `Salary` column via median split:
  * `Low` → Salary ≤ ₹3,00,000 (median)
  * `High` → Salary > ₹3,00,000

### Feature Groups

| Group                              | Features                                                                                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Demographics**             | `Gender`, `Age`, `CollegeTier`, `CollegeCityTier`                                                                                       |
| **Academic History**         | `10percentage`, `12percentage`, `collegeGPA`, `Degree`, `Specialization`                                                              |
| **Aptitude Scores**          | `English`, `Logical`, `Quant`, `Domain`                                                                                                 |
| **Engineering Test Modules** | `ComputerProgramming`, `ElectronicsAndSemicon`, `ComputerScience`, `MechanicalEngg`, `ElectricalEngg`, `TelecomEngg`, `CivilEngg` |
| **Psychometrics**            | `conscientiousness`, `agreeableness`, `extraversion`, `nueroticism`, `openess_to_experience`                                          |
| **Target**                   | `Salary` → converted to `SalaryLabel` (Low / High)                                                                                         |

---

## Project Methodology & Step-by-Step Execution

```
┌─────────────────────────────────────┐
│     Raw Data (salary.csv)           │
│       2,998 Candidate Records       │
│         34 Original Columns         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Data Cleaning & Hygiene       │
│  - Drop invalid GraduationYear = 0  │
│  - Fix GPA scale (×9.5 for 0–10)   │
│  - Impute Domain −1 → median        │
│  - Impute module −1 scores → 0      │
│  - Median-split Salary → SalaryLabel│
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│    Exploratory Data Analysis (EDA)  │
│  - Distributions, KDE, Boxplots     │
│  - Count plots for categoricals     │
│  - Skewness analysis                │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Feature Engineering            │
│  - Group & standardize Specialization│
│  - Bin 7 engineering module scores  │
│    (pd.cut: bins 0–9, 100-pt bands) │
│  - Drop raw module columns          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   Outlier Detection via IQR         │
│  - Age, 10%, 12%, GPA, Aptitude     │
│  - 144 flagged; rows dropped        │
│  - 2,997 → 2,897 clean records      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   Correlation & Multicollinearity   │
│  - Pearson Heatmap + PhiK matrix    │
│  - VIF analysis                     │
│  - Drop psychometric redundancies   │
│  - Drop GroupedSpecialization        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│    Encoding & Final Feature Matrix  │
│  - Dummy encode Gender              │
│  - Ordinal encode Degree            │
│  - Ordinal cast Binned columns      │
│  - Encode SalaryLabel: Low=0 High=1 │
│  - 2,897 rows × 19 features         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   StandardScaler + 5-Fold Split Suite│
│  Splits: 80/20, 75/25, 70/30,       │
│          65/35, 60/40                │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│     9 Classical ML Classifiers      │
│  Logistic Regression, Naive Bayes,  │
│  KNN, Linear SVM, Decision Tree,    │
│  Random Forest, AdaBoost,           │
│  Gradient Boosting, XGBoost         │
│                                     │
│    GridSearchCV 5-Fold Optimization │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   Artificial Neural Network (ANN)   │
│  Architecture, Optimizer & Epoch    │
│  Tuning (Adam vs. SGD)              │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   Evaluation & Comparative Analysis │
│  Metrics: Accuracy, Precision,      │
│  Recall, F1-Score, ROC-AUC          │
│  Best Model: Logistic Regression    │
└─────────────────────────────────────┘
```

---

### Step 1: Data Loading & Initial Inspection

* Loaded `salary.csv` containing **2,998 candidate records** and **34 columns**.
* Verified data types with `sal.info()` and inspected sample rows with `sal.sample(5)`.
* Confirmed the dataset had **0 missing or null values** across all 34 columns.

### Step 2: Data Cleaning & Sentinel Value Treatment

Initial validation confirmed the dataset was completely populated; cleaning instead focused on fixing invalid sentinel values and structural inconsistencies:

| Issue Found                                       | Action Taken                                                                            | Result                   |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------ |
| Invalid `GraduationYear` = 0                    | Row dropped                                                                             | 2,998 → 2,997 rows      |
| `collegeGPA` on 0–10 scale (9 rows)            | Rescaled × 9.5 to match 0–100 scale                                                   | min 49.07 – max 99.93   |
| `Domain` score = −1 (not attempted)            | Replaced with median of valid scores                                                    | median ≈ 0.636          |
| Engineering test modules with −1 (not attempted) | Replaced with 0 (e.g. 649 missing `ComputerProgramming`, 2,971 missing `CivilEngg`) | 7 module columns cleaned |
| `Gender` values (`'m'`/`'f'`)               | Standardized to `'MALE'`/`'FEMALE'`                                                 | Consistent labels        |
| `Degree` values (MCA, M.Tech, M.Sc.)            | Consolidated to `'MASTERS'` vs `'B.Tech/B.E.'`                                      | 2 clean categories       |
| Continuous `Salary` column                      | Median-split into `SalaryLabel` (Low / High)                                          | Split at ₹3,00,000      |

* The original `ID`, `CollegeID`, `CollegeCityID`, `DOB`, `10board`, `12board`, `12graduation`, `CollegeState`, `GraduationYear`, and `CollegeCityTier` columns were dropped as they carry no generalizable predictive signal.
* An `Age` feature was derived as `GraduationYear − DOB.year` before the date columns were dropped.

### Step 3: Exploratory Data Analysis (EDA)

* **Categorical distributions** (count plots): `Gender`, `CollegeTier`, `Degree`, `SalaryLabel`, `GroupedSpecialization`.
* **Numerical distributions** (KDE plots + histograms with skewness annotation): `Age`, `10percentage`, `12percentage`, `collegeGPA`, `English`, `Logical`, `Quant`, `Domain`.
* **Engineering module score distributions** (KDE + Box plots): `ComputerProgramming`, `ElectronicsAndSemicon`, `ComputerScience`, `MechanicalEngg`, `ElectricalEngg`, `TelecomEngg`, `CivilEngg`.
* **Key EDA Findings:**
  * Core cognitive aptitude scores (`English`, `Logical`, `Quant`) follow a near-normal Gaussian distribution.
  * Applicant `Age` is right-skewed, with a concentration of recent graduates.
  * Engineering test-module scores are **heavily zero-inflated** — most candidates never attempted domain-specific modules other than their own specialization.

### Step 4: Feature Engineering

#### 4a. Specialization Grouping

Raw `Specialization` had dozens of granular categories. These were consolidated into broader groups (`CS_IT`, `COMP_APPL`, `ELECTRONICS`, `MECH`, `ELECTRICAL`, `TELECOM`, `CIVIL`, `OTHER`) to reduce dimensionality while preserving signal. The column was renamed `GroupedSpecialization`.

#### 4b. Engineering Module Score Binning

Seven engineering test-module scores are highly zero-inflated. Raw continuous scores were converted into **ordinal bins** using `pd.cut` to make zero-sparsity model-friendly:

```python
pd.cut(
    bins = [-inf, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
```

* **Bin 0**: Score = 0 (module not attempted)
* **Bins 1–9**: 100-point score bands (100–900 range)
* Applied to all 7 module columns → `*Binned` suffix columns created; raw columns dropped.

### Step 5: Outlier Detection & Removal via IQR

* Interquartile Range (IQR) analysis ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) was applied across core numerical features:

| Feature          | Outliers Flagged     |
| ---------------- | -------------------- |
| `Age`          | 56                   |
| `10percentage` | 17                   |
| `12percentage` | 1                    |
| `collegeGPA`   | 23                   |
| `English`      | 12                   |
| `Logical`      | 13                   |
| `Quant`        | 22                   |
| `Domain`       | 0                    |
| **Total**  | **144 values** |

* Outlier rows were removed in a single pass using a combined boolean mask (a row was dropped only if any of its values breached any fence).
* Distribution checks via Box Plots confirmed improved distributional symmetry post-cleaning.
* **Final cleaned dataset:** 2,897 rows.

### Step 6: Correlation & Multicollinearity Analysis

* **Pearson Correlation Heatmap** on numerical features: revealed a notable `10percentage`–`12percentage` correlation of **0.64** (expected academic continuity signal). No pair exceeded 0.85, so no further numeric feature removal was required.
* **PhiK (ΦK) Non-linear Correlation Matrix** (`phik` library): evaluated mixed-type relationships including categorical features against `SalaryLabel`.
* **Variance Inflation Factor (VIF)** analysis was performed after dummy encoding to detect multicollinearity:

| Feature                                           | VIF   |
| ------------------------------------------------- | ----- |
| `GroupedSpecialization_ELECTRICAL`              | 26.16 |
| `GroupedSpecialization_CS_IT`                   | 24.45 |
| `GroupedSpecialization_MECH`                    | 11.43 |
| `Degree_MASTERS`                                | 6.22  |
| `MechanicalEnggBinned`                          | 5.96  |
| All numeric predictors (Age, percentages, scores) | < 2.0 |

> **Finding:** High VIF values appear only in one-hot encoded `GroupedSpecialization` dummy columns, which is structurally expected and harmless for mutually exclusive category dummies. All genuine numeric predictors sit comfortably below 2.0 VIF.

* **Psychometric Pruning:** `conscientiousness`, `agreeableness`, `extraversion`, and `openess_to_experience` were dropped after correlation review (low target correlation, feature redundancy). `nueroticism` was retained.
* `GroupedSpecialization` column (pre-encoded) was dropped after encoding.

### Step 7: Final Encoding & Feature Matrix Preparation

* **`Gender`**: Dummy-encoded with `drop_first=True` → `Gender_MALE` (binary).
* **`Degree`**: Ordinal encoded as `B.Tech/B.E.` → 0, `MASTERS` → 1.
* **Binned module columns**: Cast from categorical to integer dtype.
* **`SalaryLabel`**: Encoded as `Low` → 0, `High` → 1.
* Cleaned dataset exported to `cleaned_salary.csv`.
* **Final Feature Matrix:** **2,897 rows × 19 features** + 1 target column.

### Step 8: Multi-Split Validation & Feature Scaling

* To eliminate arbitrary partition bias, all baseline models were evaluated across **5 distinct train/test splits**:
  * 80/20, 75/25, 70/30, 65/35, and 60/40.
* Feature scaling was conducted using `StandardScaler`. The scaler was strictly **fitted on the training split only** and applied to the test split to prevent data leakage.
* Metrics tracked per split: **Accuracy, Precision, Recall, F1-Score, ROC-AUC** (Training vs. Testing).

### Step 9: Machine Learning Model Benchmarking

Nine binary classification algorithms were systematically benchmarked:

1. **Logistic Regression**
2. **Gaussian Naïve Bayes**
3. **K-Nearest Neighbors Classifier (KNN)**
4. **Linear Support Vector Machine (Linear SVC)**
5. **Decision Tree Classifier**
6. **Random Forest Classifier**
7. **AdaBoost Classifier**
8. **Gradient Boosting Classifier**
9. **Extreme Gradient Boosting Classifier (XGBoost)**

### Step 10: Hyperparameter Optimization (GridSearchCV)

* Applied **5-Fold Cross Validation** (`cv=5`) on the 80/20 split, optimizing for `accuracy`.
* Hyperparameter search spaces explored:

| Model                         | Parameters Tuned                                                                                                             |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | `C` [0.01, 0.1, 1.0, 10.0]; `penalty` ['l1', 'l2']                                                                       |
| **Naïve Bayes**        | `var_smoothing` [1e-9, 1e-8, 1e-7, 1e-5]                                                                                   |
| **KNN**                 | `n_neighbors` [5, 7, 9]; `weights` ['uniform', 'distance']; `metric` ['manhattan', 'euclidean', 'cosine', 'haversine'] |
| **Linear SVM**          | `penalty` ['l1', 'l2']; `C` [0.1, 1.0, 10.0]                                                                             |
| **Decision Tree**       | `max_depth` [10, 15, 20]; `max_leaf_nodes` [10, 20, 50, 100]                                                             |
| **Random Forest**       | `n_estimators` [100, 150, 200]; `max_depth` [10, 15, 20]                                                                 |
| **AdaBoost**            | `n_estimators` [100, 150, 200]; `learning_rate` [0.01, 0.1, 1.0]                                                         |
| **Gradient Boosting**   | `n_estimators` [100, 150, 200]; `learning_rate` [0.01, 0.05, 0.1]                                                        |
| **XGBoost**             | `n_estimators` [100, 150, 200]; `max_depth` [3, 5, 7]                                                                    |

* Confusion matrices were plotted for each tuned model on the 80/20 split.

### Step 11: Deep Learning Architecture Sweeps (ANN)

* Built using `TensorFlow/Keras` with **Binary Cross-Entropy loss** and **Sigmoid** output activation.
* **Architectures Evaluated:**
  * Shallow: $Input \to 10 \to 5 \to 1$
  * Medium: $Input \to 64 \to 32 \to 16 \to 1$
  * Deep Pyramid: $Input \to 32 \to 32 \to 16 \to 16 \to 8 \to 1$
  * Very Deep: $Input \to 128 \to 64 \to 32 \to 16 \to 8 \to 1$
* **Optimizer Benchmarks:** Adam vs. Stochastic Gradient Descent (SGD) evaluated at **20, 50, and 100 epochs**.
* Training and validation **loss & accuracy curves** were plotted at each epoch configuration to monitor convergence, detect overfitting, and identify optimal checkpoints.
* **Final Best ANN:** Architecture $128 \to 64 \to 32 \to 16 \to 8 \to 1$, **Adam** optimizer, **100 epochs**, batch size 64.

### Step 12: Final Model Selection & Prediction

* All GridSearchCV tuned results across all 9 classifiers and the ANN were aggregated and compared side-by-side.
* **Best Overall Model: Logistic Regression** (consistent test accuracy ~69.8%, highest generalization across all splits, lowest train-test gap — no overfitting).
* The final Logistic Regression model (`C=1.0`, `penalty='l2'`) was retrained on the 80/20 split and used to generate a final `Actual vs. Predicted` comparison DataFrame.

---

## Key Insights & Discussion

1. **Logistic Regression as the Champion:**

   * Despite the complexity of ensemble methods, Logistic Regression consistently achieved the highest test accuracy (~69.8%) and the smallest train-test gap across all 5 splits — signaling robust generalization. Its linear decision boundary is well-suited to the feature space of this dataset.
2. **Tree-Based Overfitting:**

   * Unconstrained Decision Trees, Random Forests, and XGBoost achieved perfect training accuracy (1.0000) but dropped to test accuracies in the range of 0.60–0.67. GridSearchCV with pruning parameters recovered some generalization, but the models remained comparatively weaker on unseen data.
3. **ANN vs. Classical ML:**

   * The deep ANN (`128 × 64 × 32 × 16 × 8 × 1`, Adam, 100 epochs) approached competitive accuracy levels but could not decisively outperform a well-regularized Logistic Regression — reinforcing that for tabular classification on moderate-sized datasets, classical linear models often remain strong baselines.
4. **Feature Insights:**

   * Aptitude scores (`English`, `Logical`, `Quant`) and academic history (`10percentage`, `12percentage`, `collegeGPA`) were the strongest differentiators of salary bracket.
   * Zero-inflated engineering module scores, once binned, contributed positively to model performance without introducing sparsity issues.

---

## Future Scope

* **Feature Enrichment:** Incorporate job location, company tier, years of experience at time of placement, and industry sector to improve predictive power.
* **Advanced Class Balancing:** Apply SMOTE or class-weight tuning to further improve recall for the minority salary class if the split is ever unequal.
* **Explainability (XAI):** Use SHAP (SHapley Additive exPlanations) values to quantify the per-feature contribution to each salary-bracket prediction, making the model interpretable for HR and recruitment applications.
* **Multi-Class Extension:** Instead of a binary High/Low split, predict finer salary brackets (e.g., quartile-based: Very Low, Low, High, Very High) for more granular talent analytics.
* **Deployment:** Package the best Logistic Regression model as a REST API (Flask/FastAPI) for integration into campus placement portals.

---

## Acknowledgments

* **United Network of Professionals (UNP):** For providing internship guidance, project mentoring, and structured technical support throughout the program.
* **Bhavans Vivekananda College of Science, Humanities and Commerce:** For academic encouragement and project enablement.
