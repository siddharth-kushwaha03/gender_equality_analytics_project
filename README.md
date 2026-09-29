# AI-Based Workplace Gender Equality Analytics

Minor Project — SDG 5 (Gender Equality)

---

## What this does

Analyzes a workforce (HR) dataset to detect gender gaps in **pay**,
**promotion**, **representation at leadership levels**, and **attrition** —
then trains Machine Learning models to predict promotions and *audits those
models for gender bias* using standard fairness metrics.

---

## What's new in v2

| # | Fix | Details |
|---|-----|---------|
| 1 | ✅ Data validation | `validate_data()` checks columns, nulls, salary > 0, binary targets — raises clear error messages |
| 2 | ✅ Department pay gap | New analysis + chart: pay gap broken down per department |
| 3 | ✅ Two ML models | Logistic Regression **and** Random Forest — both trained, compared, best used for fairness audit |
| 4 | ✅ Equalized Odds | TPR gap + FPR gap metrics added alongside DIR and SPD |
| 5 | ✅ Model persistence | `save_model` / `load_model` via `joblib` → `models/promotion_model.pkl` |
| 6 | ✅ CSV upload | Dashboard accepts your own employee CSV |
| 7 | ✅ Download buttons | Every chart and table is downloadable from the dashboard |
| 8 | ✅ Env-var credentials | `ADMIN_USERNAME`, `ADMIN_PASSWORD` etc. — no hardcoded secrets in production |
| 9 | ✅ Unit tests | 40+ tests in `tests/test_analytics.py` (pytest) |
| 10 | ✅ Pinned deps | `requirements.txt` has exact versions for reproducibility |
| 11 | ✅ `.gitignore` | DS_Store, outputs, models, venv all excluded |
| 12 | ✅ DRY code | `app.py` now imports from `analytics.py` — no duplicated logic |

---

## Project structure

```
project/
├── generate_data.py          # creates the synthetic HR dataset
├── analytics.py              # core engine (validation, analytics, ML, fairness, charts)
├── app.py                    # interactive Streamlit dashboard
├── data/
│   └── employees.csv         # generated dataset
├── outputs/                  # charts (6 PNGs) + metrics_summary.json
├── models/
│   └── promotion_model.pkl   # saved best model (joblib)
├── tests/
│   └── test_analytics.py     # 40+ pytest unit tests
├── requirements.txt          # pinned dependencies
└── .gitignore
```

---

## How to run

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### Option A — Jupyter Notebook

Open `gender_equality_analytics.ipynb` in VS Code and click **Run All**.

### Option B — Python scripts (recommended)

```bash
# Step 1 — Generate the synthetic dataset
python3 generate_data.py

# Step 2 — Run analytics pipeline (saves charts + model + metrics JSON)
python3 analytics.py

# Step 3 — Launch interactive dashboard
streamlit run app.py
```

### Option C — Run unit tests

```bash
pytest tests/ -v
```

---

## Dashboard features

- **Login** with role-based access (HR Director / Analytics Lead)
- **Upload your own CSV** or use the default synthetic dataset
- **6 tabs**: Representation | Pay (Level) | Pay (Dept) | Promotion & Attrition | AI Bias Audit | Raw Data
- **Download** every chart (PNG) and table (CSV) directly from the dashboard
- **Equalized Odds** displayed alongside Disparate Impact Ratio

### Demo Credentials

| Role | Username | Password | Scope |
|------|----------|----------|-------|
| **HR Director / Admin** | `admin` | `admin123` | Full dashboard |
| **Analytics Lead** | `analyst` | `analyst123` | Analytics & bias audit |

> **Production tip:** Set credentials via environment variables:
> ```bash
> export ADMIN_USERNAME=myuser
> export ADMIN_PASSWORD=supersecret
> streamlit run app.py
> ```

---

## Notes for the viva / presentation

- The dataset is **synthetic** (randomly generated) but built with realistic,
  literature-based patterns (residual pay gap even after controlling for
  experience/performance, and a "leaky pipeline" at senior levels).
- The core ML insight: the model is trained **without gender as an input**,
  yet its predictions still produce a gender gap — because correlated features
  (department, level) act as *proxies* for gender. This is **proxy
  discrimination**, and it's the real lesson employers need.
- **Equalized Odds** goes beyond simple selection rate parity: it checks
  whether the model makes equally accurate decisions for both groups.
- Swap `data/employees.csv` for a real anonymized dataset to make this
  production-relevant.
