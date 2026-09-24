# 🎬 Netflix Customer Churn & Engagement Analytics

> **⚠️ Data Disclaimer:** This project uses the public/synthetic Kaggle dataset
> *"Netflix Customer Churn and Engagement Analytics"* by Zeyad Mohamed.
> It does **NOT** contain confidential, proprietary, or real Netflix customer data.
> All findings are illustrative and intended for academic purposes only.

---

## Project Overview

This project analyses a Netflix-style customer dataset to identify churn patterns,
understand engagement behaviour, and build an interpretable churn-prediction model.
It covers the full data-science pipeline: data cleaning, exploratory analysis, KPI
reporting, customer segmentation (K-Means), and two classification models
(Logistic Regression and Random Forest with GridSearchCV tuning).

---

## Objectives

1. Understand the overall churn rate and key customer engagement KPIs.
2. Identify which subscription plan, demographic, and behavioural features are most
   associated with churn.
3. Segment customers into interpretable groups with different risk profiles.
4. Build and evaluate two churn-prediction models with full metric reporting.
5. Derive five actionable, data-supported retention recommendations.

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | Kaggle — Zeyad Mohamed |
| **URL** | https://www.kaggle.com/datasets/zeyadmohamed26/netflix-customer-churn-and-engagement-analytics |
| **Format** | CSV |
| **Expected filename** | `netflix_customer_churn.csv` |
| **Location** | Place inside the `data/` folder (see setup below) |

> The dataset is **not included** in this repository. You must download it from Kaggle.

---

## Project Structure

```
netflix-churn-analytics/
├── data/
│   └── netflix_customer_churn.csv        ← download from Kaggle & place here
├── figures/                              ← auto-created by notebook at runtime
│   └── *.png                             ← exported charts (for Word report)
├── MINAL_NetflixCustomerChurnEngagementAnalytics.ipynb
├── MINAL_NetflixCustomerChurnEngagementAnalytics_ProjectReport.docx
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone or download this repository
cd netflix-churn-analytics

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place it at:
#    data/netflix_customer_churn.csv

# 5. Launch Jupyter Notebook
jupyter notebook
```

---

## How to Run

1. Open `MINAL_NetflixCustomerChurnEngagementAnalytics.ipynb` in Jupyter.
2. Ensure `data/netflix_customer_churn.csv` is in place.
3. Select **Kernel → Restart & Run All**.
4. All outputs, charts, and model results are generated automatically from the data.
5. Exported chart PNGs appear in the `figures/` directory — use these for the Word report.

> ℹ️ The notebook will print a clear error message and exit gracefully if the CSV file
> is missing, with instructions on where to place it.

---

## Technologies

| Tool / Library | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Core language |
| pandas | ≥ 2.0 | Data loading, cleaning, aggregation |
| NumPy | ≥ 1.24 | Numerical operations |
| Matplotlib | ≥ 3.7 | Base plotting |
| Seaborn | ≥ 0.12 | Statistical visualisations |
| scikit-learn | ≥ 1.3 | ML pipeline, models, metrics, clustering |
| Jupyter Notebook | ≥ 7.0 | Interactive notebook environment |
| openpyxl | ≥ 3.1 | Excel export support |

---

## Key Findings

> ℹ️ This section is populated after running the notebook on the actual dataset.

- **Overall churn rate:** 50.30%
- **Best model:** Random Forest (GridSearchCV Tuned) — Test ROC-AUC: 0.9981
- **Top churn drivers:** Days since last active, watch time, subscription tenure, payment failures, plan type
- **Highest-risk segment:** Dormant High-Churn Risk cluster
- **5 retention actions** targeting inactivity, payment friction, plan upgrades, loyalty rewards, and support escalation

---

## Models & Results

| Model | Accuracy | Recall | ROC-AUC (Test) | ROC-AUC (CV Mean) |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.8870	| 0.9046 | 0.9658 | 0.9662 |
| Random Forest (GridSearchCV Tuned) | 0.9850 | 0.9821 |	0.9981 |	0.9971 |


## Visualisations

The notebook produces the following charts (saved to `figures/`):

| File | Description |
|---|---|
| `01_churn_distribution.png` | Overall churn split — bar + pie |
| `02_churn_by_plan.png` | Churn rate by subscription plan |
| `03_monthly_spend_churn.png` | Monthly spend KDE + box plot |
| `04_tenure_churn.png` | Subscription tenure KDE + binned churn rate |
| `05_watch_time_churn.png` | Watch time KDE + box plot |
| `06_profiles_churn.png` | Churn by number of profiles |
| `07_payment_method_churn.png` | Churn rate by payment method |
| `08_devices_churn.png` | Churn by device count |
| `09_demographics_churn.png` | Gender bar chart + age KDE |
| `10_correlation_heatmap.png` | Feature correlation matrix (annotated) |
| `11_tickets_failures_churn.png` | Support tickets & payment failures vs churn |
| `12_kmeans_elbow_silhouette.png` | K-Means elbow + silhouette plots |
| `13_pca_segments.png` | PCA 2D segment scatter |
| `14_confusion_matrices.png` | Side-by-side confusion matrices |
| `15_roc_curves.png` | ROC curves — LR (solid) vs RF (dashed) |
| `16_rf_feature_importance.png` | Random Forest top-15 feature importances |
| `17_lr_coefficients.png` | Logistic Regression top-15 coefficients |

---

## Limitations

- Dataset is public/synthetic — findings may not reflect real Netflix customer behaviour.
- Churn is modelled as a static binary label; no temporal/survival analysis is performed.
- All associations are correlational, not causal.
- GridSearchCV covers a controlled parameter grid; wider searches may improve performance.

---


## Conclusion

This project analyzed a synthetic Netflix-style customer dataset to understand churn patterns, customer engagement, and retention opportunities. The analysis found that low average watch time, longer time since last login, subscription plan, and payment method were key indicators of churn risk.

A tuned Random Forest model outperformed the Logistic Regression baseline, achieving strong predictive performance. Customer segmentation also identified high-risk groups that can be targeted with personalised re-engagement campaigns, payment-support interventions, plan-upgrade offers, and content recommendations.

Because the dataset is synthetic, these findings are intended for academic use. In a real-world setting, the model should be validated on current customer data and monitored for fairness, privacy, and changing customer behaviour.

---

