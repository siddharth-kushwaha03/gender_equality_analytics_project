# AI-Based Workplace Gender Equality Analytics

Minor Project — SDG 5 (Gender Equality)

## What this does

Analyzes a workforce (HR) dataset to detect gender gaps in **pay**,
**promotion**, **representation at leadership levels**, and
**attrition** — then trains a Machine Learning model to predict
promotions and *audits that model for gender bias* using standard
fairness metrics (Disparate Impact Ratio / four-fifths rule,
Statistical Parity Difference).

## Project structure

```
project/
├── generate_data.py   # creates the synthetic HR dataset
├── analytics.py        # descriptive analytics + ML fairness audit (CLI, generates charts)
├── app.py               # interactive Streamlit dashboard (the demo)
├── data/employees.csv   # generated dataset
├── outputs/              # charts + metrics_summary.json (from analytics.py)
└── requirements.txt
```

## How to run

### Option A — Jupyter Notebook (recommended, run in VS Code)

Open `gender_equality_analytics.ipynb` in VS Code (with the Jupyter extension) and click **Run All**.
It does everything end-to-end: generates the dataset, runs all descriptive analytics + statistical
tests, trains the ML model, runs the fairness/bias audit, plots all 4 charts inline, and saves
`outputs/metrics_summary.json`.

```bash
pip install -r requirements.txt ipykernel
# then open gender_equality_analytics.ipynb in VS Code and Run All
```

### Option B — Plain Python scripts

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python3 generate_data.py

# 2. Run the analytics + ML pipeline (creates charts + metrics_summary.json)
python3 analytics.py

# 3. Launch the interactive dashboard
streamlit run app.py
```

### Dashboard Authentication (Demo Credentials)

The dashboard is secured behind an enterprise login page simulating HR role-based access control:

| Role | Username | Password | Access Scope |
| :--- | :--- | :--- | :--- |
| **HR Director / Admin** | `admin` | `admin123` | Full dashboard access with executive privileges |
| **People Analytics Lead** | `analyst` | `analyst123` | Analytics & bias audit access |

*(You can also sign out at any time using the **Sign Out** button in the sidebar).*

## Notes for the viva / presentation

- The dataset is **synthetic** (randomly generated, not real employee
  data) but built with realistic, literature-based patterns (a residual
  pay gap even after controlling for experience/performance, and a
  "leaky pipeline" where women's representation shrinks at senior
  levels) — this is standard practice for an academic project when
  real HR data isn't available for privacy reasons.
- The core innovation to highlight: the ML model is trained **without
  gender as an input feature** (a common naive assumption that this
  makes it "fair"), and the project shows that its predictions *still*
  produce a gender gap — because other features (e.g., which
  department, which level) correlate with gender. This is the real
  lesson employers need: removing a protected attribute from a model
  does not guarantee fairness.
- Swap `data/employees.csv` for a real (anonymized) dataset from
  Kaggle or an actual company export to make this production-relevant.
