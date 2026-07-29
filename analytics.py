"""
analytics.py
------------
AI-Based Workplace Gender Equality Analytics — core engine.

Pipeline:
    1. Descriptive analytics: representation, pay gap, promotion gap,
       attrition gap (overall + by department/level).
    2. Statistical significance testing (t-test, chi-square).
    3. ML model: predicts promotion probability from job-relevant
       features (gender deliberately EXCLUDED from training features)
       and then measures whether the model's decisions still produce
       a gender disparity in outcomes (fairness / bias audit) using
       standard fairness metrics: Disparate Impact Ratio and
       Statistical Parity Difference.
    4. Saves all charts to outputs/ and a metrics summary to
       outputs/metrics_summary.json for use in the project report.

Run:
    python3 analytics.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

OUT = "outputs"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

def load_data():
    return pd.read_csv("data/employees.csv")


# ----------------------------------------------------------------------
# 1. Descriptive analytics
# ----------------------------------------------------------------------
def representation_by_level(df):
    tbl = (
        df.groupby(["job_level", "gender"]).size().unstack(fill_value=0)
    )
    tbl = tbl.reindex(["Associate", "Senior", "Lead", "Manager", "Director"])
    tbl_pct = tbl.div(tbl.sum(axis=1), axis=0) * 100
    return tbl, tbl_pct


def pay_gap_overall(df):
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


def pay_gap_by_level(df):
    g = df.groupby(["job_level", "gender"])["annual_salary"].mean().unstack()
    g = g.reindex(["Associate", "Senior", "Lead", "Manager", "Director"])
    g["gap_pct"] = (g["Male"] - g["Female"]) / g["Male"] * 100
    return g.round(2)


def promotion_rates(df):
    rates = df.groupby("gender")["promoted_last_cycle"].mean() * 100
    ct = pd.crosstab(df.gender, df.promoted_last_cycle)
    chi2, pval, _, _ = stats.chi2_contingency(ct)
    return rates.round(2), {"chi2": round(chi2, 3), "p_value": round(pval, 5)}


def attrition_rates(df):
    return (df.groupby("gender")["exited_last_12mo"].mean() * 100).round(2)


# ----------------------------------------------------------------------
# 2. ML model + fairness / bias audit
# ----------------------------------------------------------------------
def train_promotion_model(df):
    """
    Trains a model to predict 'promoted_last_cycle' from strictly
    job-relevant features (experience, performance, level, education,
    department). 'gender' is withheld from the model's inputs on
    purpose — this simulates a "gender-blind" hiring/promotion
    algorithm. We then check whether the model's OWN predictions still
    carry a gender gap, which would indicate the job-relevant features
    are acting as proxies for gender (a common real-world bias
    mechanism).
    """
    features = ["experience_years", "performance_rating", "job_level",
                "education", "department", "age"]
    X = df[features].copy()
    y = df["promoted_last_cycle"].astype(int)

    cat_cols = ["job_level", "education", "department"]
    num_cols = ["experience_years", "performance_rating", "age"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, df["gender"], test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    result_df = pd.DataFrame({
        "gender": gender_test.values,
        "actual": y_test.values,
        "predicted": preds,
        "pred_prob": probs,
    })

    return pipe, result_df, {"accuracy": round(acc, 3), "auc": round(auc, 3)}


def fairness_metrics(result_df):
    """
    Disparate Impact Ratio (DIR): P(pred=1 | Female) / P(pred=1 | Male)
      - A common rule of thumb (US EEOC "four-fifths rule"): DIR < 0.8
        signals adverse impact against the disadvantaged group.
    Statistical Parity Difference (SPD): P(pred=1|Female) - P(pred=1|Male)
      - 0 = perfect parity.
    """
    rate_male = result_df.loc[result_df.gender == "Male", "predicted"].mean()
    rate_female = result_df.loc[result_df.gender == "Female", "predicted"].mean()
    dir_ratio = rate_female / rate_male if rate_male > 0 else np.nan
    spd = rate_female - rate_male
    return {
        "predicted_promotion_rate_male": round(rate_male * 100, 2),
        "predicted_promotion_rate_female": round(rate_female * 100, 2),
        "disparate_impact_ratio": round(dir_ratio, 3),
        "passes_four_fifths_rule": bool(dir_ratio >= 0.8),
        "statistical_parity_difference": round(spd * 100, 2),
    }


# ----------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------
def make_charts(df, rep_pct, pay_level, promo_rates, attr_rates, fairness):
    # Chart 1: representation by level (stacked %)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rep_pct.plot(kind="bar", stacked=True, ax=ax, color=["#e07a5f", "#3d5a80"])
    ax.set_ylabel("% of employees")
    ax.set_title("Gender Representation by Job Level")
    ax.legend(title="Gender")
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart1_representation_by_level.png")
    plt.close(fig)

    # Chart 2: pay gap by level
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pay_level[["Male", "Female"]].plot(kind="bar", ax=ax, color=["#3d5a80", "#e07a5f"])
    ax.set_ylabel("Average Annual Salary")
    ax.set_title("Average Salary by Job Level and Gender")
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart2_pay_gap_by_level.png")
    plt.close(fig)

    # Chart 3: promotion & attrition rates
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

    # Chart 4: model fairness - predicted promotion rate by gender
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(
        ["Male", "Female"],
        [fairness["predicted_promotion_rate_male"], fairness["predicted_promotion_rate_female"]],
        color=["#3d5a80", "#e07a5f"],
    )
    ax.axhline(fairness["predicted_promotion_rate_male"] * 0.8, ls="--", color="gray",
               label="4/5ths rule threshold")
    ax.set_ylabel("% predicted promoted")
    ax.set_title("ML Model's Predicted Promotion Rate by Gender\n(Bias / Fairness Audit)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{OUT}/chart4_model_fairness_audit.png")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_data()

    rep_counts, rep_pct = representation_by_level(df)
    pay_overall = pay_gap_overall(df)
    pay_level = pay_gap_by_level(df)
    promo_rates, promo_test = promotion_rates(df)
    attr_rates = attrition_rates(df)

    model, result_df, model_perf = train_promotion_model(df)
    fairness = fairness_metrics(result_df)

    make_charts(df, rep_pct, pay_level, promo_rates, attr_rates, fairness)

    summary = {
        "dataset_size": len(df),
        "representation_pct_by_level": rep_pct.round(2).to_dict(),
        "pay_gap_overall": pay_overall,
        "pay_gap_by_level": pay_level.to_dict(),
        "promotion_rates_pct": promo_rates.to_dict(),
        "promotion_chi_square_test": promo_test,
        "attrition_rates_pct": attr_rates.to_dict(),
        "ml_model_performance": model_perf,
        "fairness_audit": fairness,
    }

    with open(f"{OUT}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nCharts saved to outputs/. Summary saved to outputs/metrics_summary.json")


if __name__ == "__main__":
    main()
