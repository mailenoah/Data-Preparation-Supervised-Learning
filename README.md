



**Scenario:** As a junior data analyst at a South African bank, the task is to explore and prepare historical marketing campaign data, then build a machine learning model to predict whether a client will subscribe to a term deposit.

**Learning Objectives:**
- Apply real-world data preparation techniques
- Perform exploratory data analysis and feature engineering
- Encode and preprocess data for modelling
- Train and evaluate classification models
- Generate and interpret model evaluation metrics

---

## 🔍 Project Overview

This project tackles a **binary classification problem**: predicting whether a bank client will subscribe to a term deposit (`yes` / `no`) based on demographic and campaign data. The full supervised learning pipeline is covered — from raw data cleaning and skewness correction through to model evaluation and result discussion.

**Business Problem:** A South African bank ran a series of telemarketing campaigns. The goal is to build a predictive model for subscription likelihood, enabling more targeted and cost-effective future campaigns.

---

## 📦 Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

**File used:** `bank-full(1).csv` (semicolon-delimited)

| Property | Value |
|---|---|
| Rows | 45,211 |
| Columns | 17 |
| Target variable | `y` (yes/no — term deposit subscription) |
| Class distribution | ~88% No / ~12% Yes (imbalanced) |

### Feature Overview

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Client age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Categorical | Has credit in default? |
| `balance` | Numerical | Average yearly account balance (€) |
| `housing` | Categorical | Has housing loan? |
| `loan` | Categorical | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numerical | Last contact day of month |
| `month` | Categorical | Last contact month |
| `duration` | Numerical | Last contact duration (seconds) — *dropped to prevent data leakage* |
| `campaign` | Numerical | Number of contacts this campaign |
| `pdays` | Numerical | Days since last contact from previous campaign |
| `previous` | Numerical | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign |
| `y` | Categorical | **Target:** subscribed to term deposit? |

---

## 📁 Project Structure

```
bank-marketing-supervised-learning/
│
├── Assignment_01_Data_Preparation_and_Supervised_Learning.ipynb  # Main notebook
├── bank-full(1).csv                                               # Raw dataset
├── disegomaile_preprocessed_bank_data.csv                           # Cleaned & encoded dataset (submission)
└── README.md                                                      # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

---

## 🔬 Methodology

The project follows the structured pipeline defined in the INF791 Assignment 01 brief, across two parts:

---

### Part 1 — Data Preparation

#### 1.1 Data Cleaning
- Loaded the dataset using `pandas` with semicolon delimiter into JupyterLab
- Scanned for standard `NaN` values (none found) and `'unknown'` placeholder strings
- Four columns contained `'unknown'` entries, handled as follows:

| Column | Count | Strategy & Justification |
|---|---|---|
| `poutcome` | 36,959 | Replaced with `'none'` — no prior campaign contact; semantically meaningful |
| `contact` | 13,020 | Replaced with `'not-contactable'` — preserves the informational absence |
| `education` | 1,857 | Mode imputation (`secondary`) — avoids data loss on moderately-sized gap |
| `job` | 288 | Mode imputation (`blue-collar`) — small count, mode is representative |

- **7 Numerical features:** age, balance, day, duration, campaign, pdays, previous
- **10 Categorical features:** job, marital, education, default, housing, loan, contact, month, poutcome, y

#### 1.2 Feature Engineering
Three new features were engineered to enhance model performance (see [Feature Engineering](#feature-engineering) section).

#### 1.3 Basic Statistics
- Generated summary statistics (mean, median, std, min, max) for all numerical features using `.describe()`
- Identified notable outliers in `balance`, `campaign`, and `pdays`
- Computed a correlation matrix and plotted a heatmap to identify relationships between numerical variables

#### 1.4 Data Visualisation
At least four plot types were used to explore distributions and relationships with the target variable:
- **Histograms** — distribution of numerical features
- **Bar charts** — subscription counts across categorical features
- **Boxplots** — spread and outliers per feature, grouped by target
- **Heatmap** — correlation matrix across numerical variables

**Skewness Analysis & Normalisation (as per assignment requirement):**
- All numerical variables were first examined for skewness
- Variables with significant skewness were transformed using logarithmic, square root, or Box-Cox transformations
- After transformation, all numerical variables were scaled using **Min-Max scaling** to bring them into a comparable range
- A combined density plot was generated to visualise all normalized and scaled variables in a single graph, facilitating cross-feature comparison (see Figure 1 in the report)

#### 1.5 Data Preprocessing
- Dropped `duration` to eliminate data leakage (value only known after a call ends)
- Converted engineered features to appropriate numeric types
- Applied **StandardScaler** to numerical features for model input readiness
- Verified consistent data types and formatting across all columns

#### 1.6 Data Encoding
- Applied **One-Hot Encoding** to all nominal categorical variables
- Binary columns (`housing`, `loan`, `default`, `y`) converted to 0/1
- Verified that all final features are numeric and model-ready
- Exported the preprocessed dataset as a `.csv` file

---

### Part 2 — Supervised Learning

#### 2.1–2.2 Model Training & Selection
Two classifiers were trained on an 80/20 train-test split (`random_state=42`):

| Model | Rationale |
|---|---|
| **Logistic Regression** | Baseline linear model — fast, interpretable, well-suited for binary classification; captures linear relationships between features and subscription likelihood |
| **Random Forest** | Ensemble method — captures non-linear relationships and feature interactions; robust to mixed feature types and class imbalance |

#### 2.3 Model Evaluation
All models were evaluated on the following metrics:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~88% | ~90%+ |
| Precision | Moderate | Higher |
| Recall | Lower | Higher |
| F1 Score | Lower | Higher |
| ROC-AUC | ~0.75 | ~0.85 |
| Training Time | Faster | Slower |

A ROC curve was plotted comparing both models against the random classifier baseline (AUC = 0.5).

#### 2.4 Discussion of Results
- Careful missing value handling preserved dataset size while improving data quality
- Dropping `duration` ensured models learned from genuinely predictive features only
- Feature engineering (particularly `age_group` and `campaign_engagement`) benefited Random Forest's ability to capture complex patterns
- Class imbalance (~88% No) means accuracy alone is insufficient — F1 and ROC-AUC are the more reliable indicators

---

## 🛠️ Feature Engineering

Three new features were created to improve predictive power:

### `age_group`
Segments clients into meaningful life-stage buckets:
- Young (< 30) · Adult (30–45) · Middle-aged (45–60) · Senior (60+)

*Purpose:* Different age groups exhibit different financial behaviour and risk appetite — this provides richer signal than raw age alone, and helps predict term deposit subscription likelihood per life stage.

### `is_employed`
Binary flag derived from `job`:
- `yes` — active employment category
- `no` — retired, student, unemployed, or unknown

*Purpose:* Employment status is a strong proxy for financial stability and the capacity to make long-term investments like term deposits.

### `campaign_engagement`
Composite feature combining current and historical campaign contact data — ratio of previous contacts to total contacts.

*Purpose:* Clients with prior engagement history show different subscription patterns than first-time contacts; this captures that recurring-interest signal in a single numeric value.

---

## 💡 Key Findings

**1. Missing Value Strategy Matters**
Replacing `'unknown'` with semantically meaningful labels (`'none'`, `'not-contactable'`) preserves information rather than introducing artificial bias through deletion or blanket imputation.

**2. Skewness Correction Improves Model Stability**
Several numerical features exhibited right skewness (notably `balance`, `campaign`, `pdays`). Applying log/Box-Cox transformations followed by Min-Max scaling brought features into a comparable range, improving model convergence and fairness between features.

**3. Data Leakage Prevention is Critical**
The `duration` feature was removed — it is only known after a call occurs, so including it would produce unrealistically optimistic performance that would not generalise to real deployment.

**4. Class Imbalance Awareness**
With ~88% of labels being `'no'`, accuracy alone is misleading. Precision, Recall, F1, and ROC-AUC together give a complete picture of model performance.

**5. Recommendation: Random Forest**
Random Forest is recommended for deployment due to superior ROC-AUC (~0.85), better F1 score, and stronger precision-recall balance. The additional training time is justified by significantly improved ability to correctly identify potential term deposit subscribers.

---

## 🧰 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations and transformations |
| `matplotlib` | Base visualisations |
| `seaborn` | Statistical plots and heatmaps |
| `scipy` | Statistical analysis, skewness testing, Box-Cox transformation |
| `scikit-learn` | Preprocessing, encoding, scaling, model training, evaluation |

---





