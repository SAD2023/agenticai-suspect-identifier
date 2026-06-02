# Suspect Prioritization System

A Python tool that helps analysts triage large lists of individuals by training a
machine-learning model to score each person and rank them by priority.

Analysts submit a list (Excel, CSV, or PDF). The system returns a sorted, color-coded
report with each person's risk score and a plain-English explanation of the key
factors driving that score.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Configuration (TODO)](#configuration-todo)
5. [How to Train the Model](#how-to-train-the-model)
6. [How to Score a List (Predict)](#how-to-score-a-list-predict)
7. [Input File Format](#input-file-format)
8. [Output Report Format](#output-report-format)
9. [Ethics and Bias Guidance](#ethics-and-bias-guidance)
10. [Extending the System](#extending-the-system)

---

## Architecture Overview

```
Analyst input file                    Labeled training dataset
     (.xlsx / .csv / .pdf)                (.csv or .xlsx)
           │                                    │
           ▼                                    ▼
    data_loader.py                       data_loader.py
           │                                    │
           ▼                                    ▼
    predictor.py ◄── trained model ──── model_trainer.py
           │              │
           │         (saved to disk
           │          via joblib)
           ▼
  report_generator.py
           │
           ▼
  Output report (.xlsx or .pdf)
  — sorted by risk score
  — color-coded by tier
  — explanation per person
```

The model is a **Random Forest classifier** (scikit-learn). It is wrapped in a
scikit-learn Pipeline alongside preprocessing steps, so the same transform + predict
logic is applied consistently at both training and inference time.

---

## File Structure

```
suspect_prioritization/
├── main.py                  CLI entry point. Run this file.
├── config.py                ⚠ Central config. Edit this before training.
├── data_loader.py           Reads training data and analyst input files.
├── model_trainer.py         Trains, evaluates, and saves the model.
├── predictor.py             Scores individuals and generates explanations.
├── report_generator.py      Writes Excel and PDF output reports.
├── create_sample_template.py Generates a blank Excel input template.
├── requirements.txt         Python package dependencies.
├── README.md                This file.
├── models/                  Saved model files (created automatically).
│   ├── trained_model.joblib
│   └── model_metadata.joblib
├── outputs/                 Output reports (created automatically).
└── logs/                    Run logs (created automatically).
```

---

## Setup and Installation

### Requirements
- Python 3.9 or newer
- pip

### Steps

```bash
# 1. Clone or download this project directory.

# 2. (Recommended) Create a virtual environment.
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies.
pip install -r requirements.txt
```

---

## Configuration (TODO)

**Before training the model**, open `config.py` and fill in the three TODO items:

```python
# In config.py — FEATURE_CONFIG section:

"FEATURE_COLUMNS": [
    # TODO: Replace with behavioral indicator column names from your dataset.
    # Example: "travel_anomaly_score", "watchlist_match_flag", ...
],

"TARGET_COLUMN": None,
    # TODO: Replace None with the name of your binary label column.
    # Example: "is_confirmed_suspect"

"CATEGORICAL_COLUMNS": [],
    # TODO: If any feature columns are categorical (non-numeric text values),
    # list them here so they are one-hot encoded during preprocessing.
```

Also update `ID_COLUMN` and `NAME_COLUMN` to match your dataset's actual column names.

> ⚠ **See the Ethics section below for guidance on which features are appropriate.**

---

## How to Train the Model

Once `config.py` is configured with your approved dataset's columns:

```bash
python main.py train --data path/to/your_dataset.csv
```

The training pipeline will:
1. Load and validate the dataset.
2. Split it 80/20 into training and test sets (stratified).
3. Fit the preprocessing + Random Forest pipeline on the training split.
4. Print evaluation metrics (precision, recall, F1, ROC-AUC) on the test split.
5. Save the fitted pipeline to `models/trained_model.joblib`.

Training only needs to be re-run when:
- A new or updated dataset becomes available.
- Feature columns are changed.
- Model hyperparameters in `config.py` are adjusted.

---

## How to Score a List (Predict)

```bash
# Score an Excel file and produce an Excel report (default):
python main.py predict --input todays_list.xlsx

# Score a CSV and produce a PDF report:
python main.py predict --input todays_list.csv --format pdf

# Specify a custom output path:
python main.py predict --input todays_list.xlsx --output reports/monday_report.xlsx
```

Output is saved to `outputs/` by default. The console will show a summary of
tier counts and the top HIGH-priority individuals immediately.

---

## Input File Format

Analyst input files must contain:
- One row per individual to score.
- All feature columns defined in `config.py FEATURE_CONFIG["FEATURE_COLUMNS"]`.
- The ID column (`person_id` by default, configurable in config.py).
- Optionally a name column for display purposes.

To generate a blank template:
```bash
python main.py template --output my_template.xlsx
```

### Supported formats
| Format | Notes |
|--------|-------|
| `.xlsx` / `.xls` | Preferred. First sheet is read. |
| `.csv` | UTF-8 or Latin-1 encoding auto-detected. |
| `.pdf` | First structured table found in the document is extracted. Must be machine-readable (not scanned). For complex PDFs, convert to Excel first. |

---

## Output Report Format

### Excel output (default)
Three sheets:

| Sheet | Contents |
|-------|----------|
| **Prioritized List** | All individuals sorted by risk score, color-coded by tier. |
| **HIGH Priority** | HIGH-tier individuals only, for immediate analyst action. |
| **Summary Stats** | Tier counts, score distribution, and run timestamp. |

Color coding:
- 🔴 **RED** = HIGH priority (risk score ≥ 0.70 by default)
- 🟡 **AMBER** = MEDIUM priority (risk score ≥ 0.40)
- 🟢 **GREEN** = LOW priority

### PDF output
A printable summary listing all HIGH and MEDIUM individuals with their ID, name,
risk score, and the key factors that drove their score.

### Explanation text
Every individual in the output receives an explanation, for example:

> *"Risk score 0.83 — HIGH priority. Key indicators: watchlist_score = 8.7
> [strongly elevated (pop. avg ≈ 1.2)]; travel_anomaly_flag = 1 [above average
> (pop. avg ≈ 0.06)]; doc_validity_score = 2.1."*

This text tells the analyst **which specific features** drove the score so they
can validate or challenge the model's output.

---

## Ethics and Bias Guidance

> This section must be reviewed by your ethics committee before any dataset
> is connected to this system.

### Features must be behavioral, not demographic

The model learns patterns from historical data. If the training data contains
demographic attributes (race, ethnicity, religion, national origin, gender, age),
the model will learn statistical correlations between those attributes and the
target label — which reflects historical arrest or prosecution patterns, not
actual criminal behavior. This produces a system that:

- Flags people based on who they are, not what they have done.
- Encodes and amplifies existing systemic biases.
- Creates legally actionable discrimination under equal-protection laws.

**Appropriate feature types:**
- Travel pattern anomaly scores
- Watchlist match flags (based on specific prior behavior, not demographics)
- Document authenticity indicators
- Network association scores (connections to known persons of interest)
- Transaction anomaly flags

**Features that must NOT be used:**
- Race or ethnicity
- Religion
- National origin or citizenship status
- Gender or sex
- Age (unless legally justified for a specific, narrow use case and ethically cleared)

### The model is a prioritization aid, not a decision

The output is a ranked list for analyst review. **Every flagged individual must
still be reviewed by a trained human analyst.** The model does not and must not
make final determinations.

### Audit and oversight

- All predictions are logged to `logs/`.
- Explanation text is included in every output row to support analyst challenge.
- Periodically re-evaluate the model for demographic parity and equal error rates
  across groups. Document these reviews.

---

## Extending the System

### Switching to a different model
Replace `RandomForestClassifier` in `model_trainer.py → build_full_pipeline()`.
The rest of the code is model-agnostic. LightGBM or XGBoost are good alternatives
for larger datasets.

### Enabling SHAP explanations
See the `_shap_explanations()` function in `predictor.py` and follow the
instructions in its docstring. SHAP provides more rigorous per-prediction
attribution and is recommended for any formal or legal use.

### Adding a web interface
`predictor.run_prediction_pipeline()` and `report_generator` can be wrapped in a
Flask or FastAPI web service. Analysts could then upload files via a browser
rather than the command line.
