"""
analytics.py
------------
AI-Based Workplace Gender Equality Analytics — core engine.

What's new vs v1:
    ✅ Data validation (validate_data) — checks columns, nulls, types, salary > 0
    ✅ Department-wise pay gap analysis (pay_gap_by_department)
    ✅ Two ML models trained & compared: Logistic Regression + Random Forest
    ✅ Extended fairness audit: Equalized Odds (TPR gap, FPR gap) added to DIR + SPD
    ✅ Model persistence: save_model / load_model via joblib
    ✅ Two new output charts: dept pay gap + model comparison bar

Pipeline:
    1. Validate data
    2. Descriptive analytics: representation, pay gap (overall / by level / by dept),
       promotion gap, attrition gap.
    3. Statistical significance: Welch t-test, chi-square.
    4. ML: Logistic Regression + Random Forest (best by AUC is used for fairness audit).
    5. Fairness audit: DIR, SPD, Equalized Odds.
    6. Save best model as models/promotion_model.pkl (joblib).
    7. Save 6 charts to outputs/ and metrics to outputs/metrics_summary.json.

Run:
    python3 analytics.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# Use Agg backend only when NOT running inside Streamlit
if "streamlit" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT = "outputs"
MODELS_DIR = "models"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

REQUIRED_COLUMNS = [
    "gender",
    "annual_salary",
    "job_level",
    "education",
    "department",
    "age",
    "experience_years",
    "performance_rating",
    "promoted_last_cycle",
    "exited_last_12mo",
]
VALID_GENDERS = {"Male", "Female"}
LEVEL_ORDER = ["Associate", "Senior", "Lead", "Manager", "Director"]


# ---------------------------------------------------------------------------
# 0.  Data Validation  (NEW)
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates an employee DataFrame.

    Checks:
        - All required columns are present
        - Gender values are only 'Male' / 'Female'
        - No null values in required columns
        - Annual salary is strictly positive
        - promoted_last_cycle and exited_last_12mo are binary (0 / 1)

    Returns True on success; raises ValueError with a human-readable
    message listing all problems found.
    """
    errors: list[str] = []

    # --- Required columns ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Gender values ---
    invalid_genders = set(df["gender"].dropna().unique()) - VALID_GENDERS
    if invalid_genders:
        errors.append(
            f"Unexpected gender values: {invalid_genders}. "
            f"Expected: {VALID_GENDERS}"
        )

    # --- Nulls ---
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        errors.append(f"Null values found in: {null_cols.to_dict()}")

    # --- Positive salary ---
    if (df["annual_salary"] <= 0).any():
        n = int((df["annual_salary"] <= 0).sum())
        errors.append(f"{n} row(s) have non-positive annual_salary.")

    # --- Binary targets ---
    for col in ["promoted_last_cycle", "exited_last_12mo"]:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, True, False}):
            errors.append(
                f"Column '{col}' must be binary (0/1), found: {unique_vals}"
            )

    if errors:
        raise ValueError(
            "Data validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return True


def load_data(path: str = "data/employees.csv") -> pd.DataFrame:
    """Load CSV, validate it, and return a clean DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run:  python3 generate_data.py"
        )
    df = pd.read_csv(path)
    validate_data(df)
    return df


# ---------------------------------------------------------------------------
# 1.  Descriptive Analytics
# ---------------------------------------------------------------------------

def representation_by_level(df: pd.DataFrame):
    """Returns (count_table, percentage_table) indexed by LEVEL_ORDER."""
    tbl = df.groupby(["job_level", "gender"]).size().unstack(fill_value=0)
    tbl = tbl.reindex(LEVEL_ORDER)
    tbl_pct = tbl.div(tbl.sum(axis=1), axis=0) * 100
    return tbl, tbl_pct


def pay_gap_overall(df: pd.DataFrame) -> dict:
    """Overall gender pay gap with Welch's t-test."""
    m = df.loc[df.gender == "Male", "annual_salary"].mean()
    f = df.loc[df.gender == "Female", "annual_salary"].mean()
    gap_pct = (m - f) / m * 100
    tstat, pval = stats.ttest_ind(
        df.loc[df.gender == "Male", "annual_salary"],
        df.loc[df.gender == "Female", "annual_salary"],
        equal_var=False,
    )
    return {
        "male_avg_salary": round(m, 2),
        "female_avg_salary": round(f, 2),
        "gap_pct": round(gap_pct, 2),
        "t_stat": round(tstat, 3),
        "p_value": round(pval, 5),
        "significant_at_5pct": bool(pval < 0.05),
    }


def pay_gap_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Average salary and pay gap % per job level."""
    g = df.groupby(["job_level", "gender"])["annual_salary"].mean().unstack()
    g = g.reindex(LEVEL_ORDER)
    g["gap_pct"] = (g["Male"] - g["Female"]) / g["Male"] * 100
    return g.round(2)


def pay_gap_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """(NEW) Average salary, pay gap %, and absolute gap per department."""
    g = df.groupby(["department", "gender"])["annual_salary"].mean().unstack()
    if "Male" in g.columns and "Female" in g.columns:
        g["gap_pct"] = (g["Male"] - g["Female"]) / g["Male"] * 100
        g["gap_abs"] = g["Male"] - g["Female"]
    return g.round(2)


def promotion_rates(df: pd.DataFrame):
    """Promotion rate per gender + chi-square significance test."""
    rates = df.groupby("gender")["promoted_last_cycle"].mean() * 100
    ct = pd.crosstab(df.gender, df.promoted_last_cycle)
    chi2, pval, _, _ = stats.chi2_contingency(ct)
    return rates.round(2), {"chi2": round(chi2, 3), "p_value": round(pval, 5)}


def attrition_rates(df: pd.DataFrame) -> pd.Series:
    """Attrition rate per gender."""
    return (df.groupby("gender")["exited_last_12mo"].mean() * 100).round(2)


# ---------------------------------------------------------------------------
# 2.  ML Models + Fairness / Bias Audit
# ---------------------------------------------------------------------------

def _build_preprocessor() -> ColumnTransformer:
    cat_cols = ["job_level", "education", "department"]
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )


def train_promotion_model(df: pd.DataFrame):
    """
    Trains Logistic Regression AND Random Forest on strictly job-relevant
    features. 'gender' is deliberately withheld to simulate a gender-blind
    algorithm.

    Both models are evaluated; the one with the higher AUC is selected as
    the 'best' model for the fairness audit.

    Returns:
        best_pipeline  — fitted sklearn Pipeline
        result_df      — test-set predictions + gender labels
        model_results  — dict of {model_name: {accuracy, auc}}
    """
    features = [
        "experience_years",
        "performance_rating",
        "job_level",
        "education",
        "department",
        "age",
    ]
    X = df[features].copy()
    y = df["promoted_last_cycle"].astype(int)

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, df["gender"], test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    model_results: dict = {}
    best_pipe = None
    best_preds: np.ndarray | None = None
    best_probs: np.ndarray | None = None
    best_auc = -1.0

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        model_results[name] = {"accuracy": round(acc, 3), "auc": round(auc, 3)}
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
            best_preds = preds
            best_probs = probs

    result_df = pd.DataFrame({
        "gender": g_test.values,
        "actual": y_test.values,
        "predicted": best_preds,
        "pred_prob": best_probs,
    })

    return best_pipe, result_df, model_results


def fairness_metrics(result_df: pd.DataFrame) -> dict:
    """
    Computes three families of fairness metrics:

    1. Disparate Impact Ratio (DIR) — four-fifths rule
       P(pred=1 | Female) / P(pred=1 | Male).  < 0.8 → adverse impact risk.

    2. Statistical Parity Difference (SPD)
       P(pred=1 | Female) − P(pred=1 | Male).  0 = perfect parity.

    3. Equalized Odds  (NEW)
       Requires equal True Positive Rates (TPR) AND False Positive Rates (FPR)
       across groups.  Gap > 0.10 is a common threshold for concern.
    """
    male = result_df[result_df.gender == "Male"]
    female = result_df[result_df.gender == "Female"]

    rate_male = male["predicted"].mean()
    rate_female = female["predicted"].mean()
    dir_ratio = (
        rate_female / rate_male if rate_male > 0 else float("nan")
    )
    spd = rate_female - rate_male

    def tpr(grp: pd.DataFrame) -> float:
        pos = grp[grp.actual == 1]
        return float(pos["predicted"].mean()) if len(pos) > 0 else float("nan")

    def fpr(grp: pd.DataFrame) -> float:
        neg = grp[grp.actual == 0]
        return float(neg["predicted"].mean()) if len(neg) > 0 else float("nan")

    tpr_m, tpr_f = tpr(male), tpr(female)
    fpr_m, fpr_f = fpr(male), fpr(female)
    tpr_gap = abs(tpr_f - tpr_m)
    fpr_gap = abs(fpr_f - fpr_m)

    return {
        "predicted_promotion_rate_male": round(rate_male * 100, 2),
        "predicted_promotion_rate_female": round(rate_female * 100, 2),
        "disparate_impact_ratio": round(dir_ratio, 3),
        "passes_four_fifths_rule": bool(dir_ratio >= 0.8),
        "statistical_parity_difference": round(spd * 100, 2),
        "equalized_odds": {
            "tpr_male": round(tpr_m, 3),
            "tpr_female": round(tpr_f, 3),
            "tpr_gap": round(tpr_gap, 3),
            "fpr_male": round(fpr_m, 3),
            "fpr_female": round(fpr_f, 3),
            "fpr_gap": round(fpr_gap, 3),
            "passes_equalized_odds_01": bool(tpr_gap <= 0.1 and fpr_gap <= 0.1),
        },
    }


# ---------------------------------------------------------------------------
# 3.  Model Persistence  (NEW)
# ---------------------------------------------------------------------------

def save_model(model, path: str | None = None) -> str:
    """Serialize the trained sklearn Pipeline to disk using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(MODELS_DIR, "promotion_model.pkl")
    joblib.dump(model, path)
    return path


def load_model(path: str | None = None):
    """Deserialize a previously saved Pipeline from disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "promotion_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model at '{path}'. Run:  python3 analytics.py"
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 4.  Charts  (6 charts now — 2 new)
# ---------------------------------------------------------------------------

def make_charts(
    df,
    rep_pct,
    pay_level,
    pay_dept,
    promo_rates,
    attr_rates,
    fairness,
    model_results,
):
    os.makedirs(OUT, exist_ok=True)

    # --- Chart 1: Representation by Level ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rep_pct.plot(kind="bar", stacked=True, ax=ax, color=["#e07a5f", "#3d5a80"])
    ax.set_ylabel("% of employees")
    ax.set_title("Gender Representation by Job Level")
    ax.legend(title="Gender")
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart1_representation_by_level.png")
    plt.close(fig)

    # --- Chart 2: Pay Gap by Level ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pay_level[["Male", "Female"]].plot(kind="bar", ax=ax, color=["#3d5a80", "#e07a5f"])
    ax.set_ylabel("Average Annual Salary")
    ax.set_title("Average Salary by Job Level and Gender")
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart2_pay_gap_by_level.png")
    plt.close(fig)

    # --- Chart 3: Promotion & Attrition ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    promo_rates.plot(kind="bar", ax=axes[0], color=["#e07a5f", "#3d5a80"])
    axes[0].set_title("Promotion Rate (%)")
    axes[0].set_ylabel("%")
    attr_rates.plot(kind="bar", ax=axes[1], color=["#e07a5f", "#3d5a80"])
    axes[1].set_title("Attrition Rate (%)")
    axes[1].set_ylabel("%")
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart3_promotion_attrition.png")
    plt.close(fig)

    # --- Chart 4: Fairness / Bias Audit ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(
        ["Male", "Female"],
        [
            fairness["predicted_promotion_rate_male"],
            fairness["predicted_promotion_rate_female"],
        ],
        color=["#3d5a80", "#e07a5f"],
    )
    ax.axhline(
        fairness["predicted_promotion_rate_male"] * 0.8,
        ls="--",
        color="gray",
        label="4/5ths rule threshold",
    )
    ax.set_ylabel("% predicted promoted")
    ax.set_title("ML Model Predicted Promotion Rate by Gender\n(Bias / Fairness Audit)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart4_model_fairness_audit.png")
    plt.close(fig)

    # --- Chart 5 (NEW): Department Pay Gap ---
    if "gap_pct" in pay_dept.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        dept_gap = pay_dept["gap_pct"].sort_values(ascending=False)
        colors = ["#e07a5f" if v > 0 else "#3d5a80" for v in dept_gap]
        dept_gap.plot(kind="bar", ax=ax, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Pay Gap (%)")
        ax.set_title(
            "Gender Pay Gap by Department\n(positive = men earn more)"
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(f"{OUT}/chart5_dept_pay_gap.png")
        plt.close(fig)

    # --- Chart 6 (NEW): Model Comparison ---
    if model_results and len(model_results) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(model_results.keys())
        accs = [model_results[n]["accuracy"] for n in names]
        aucs = [model_results[n]["auc"] for n in names]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, accs, w, label="Accuracy", color="#3d5a80")
        ax.bar(x + w / 2, aucs, w, label="AUC-ROC", color="#e07a5f")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("ML Model Comparison")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{OUT}/chart6_model_comparison.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    rep_counts, rep_pct = representation_by_level(df)
    pay_overall = pay_gap_overall(df)
    pay_level = pay_gap_by_level(df)
    pay_dept = pay_gap_by_department(df)
    promo_rates, promo_test = promotion_rates(df)
    attr_rates = attrition_rates(df)

    model, result_df, model_results = train_promotion_model(df)
    fairness = fairness_metrics(result_df)

    model_path = save_model(model)
    print(f"✅ Model saved → {model_path}")

    make_charts(
        df, rep_pct, pay_level, pay_dept,
        promo_rates, attr_rates, fairness, model_results,
    )

    summary = {
        "dataset_size": len(df),
        "representation_pct_by_level": rep_pct.round(2).to_dict(),
        "pay_gap_overall": pay_overall,
        "pay_gap_by_level": pay_level.to_dict(),
        "pay_gap_by_department": pay_dept.to_dict(),
        "promotion_rates_pct": promo_rates.to_dict(),
        "promotion_chi_square_test": promo_test,
        "attrition_rates_pct": attr_rates.to_dict(),
        "ml_model_comparison": model_results,
        "fairness_audit": fairness,
    }

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(
        "\n✅ 6 charts saved to outputs/"
        "\n✅ Summary saved to outputs/metrics_summary.json"
    )


if __name__ == "__main__":
    main()
