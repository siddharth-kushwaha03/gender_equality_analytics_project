"""
app.py
------
AI-Based Workplace Gender Equality Analytics — Interactive Dashboard.

Run with:
    streamlit run app.py

This is the demo-able "product" for the minor project: an HR team
uploads / uses workforce data and instantly sees representation,
pay-gap, promotion-gap and ML-driven bias-audit results, plus
plain-language recommendations.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Workplace Gender Equality Analytics", layout="wide")

# ----------------------------------------------------------------------
@st.cache_data
def load_data(path="data/employees.csv"):
    return pd.read_csv(path)


def pay_gap_overall(df):
    m = df.loc[df.gender == "Male", "annual_salary"].mean()
    f = df.loc[df.gender == "Female", "annual_salary"].mean()
    gap_pct = (m - f) / m * 100
    tstat, pval = stats.ttest_ind(
        df.loc[df.gender == "Male", "annual_salary"],
        df.loc[df.gender == "Female", "annual_salary"],
        equal_var=False,
    )
    return m, f, gap_pct, pval


@st.cache_resource
def train_model(df):
    features = ["experience_years", "performance_rating", "job_level",
                "education", "department", "age"]
    X = df[features].copy()
    y = df["promoted_last_cycle"].astype(int)
    cat_cols = ["job_level", "education", "department"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                             remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, df["gender"], test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    res = pd.DataFrame({"gender": g_test.values, "actual": y_test.values,
                         "predicted": preds, "pred_prob": probs})
    perf = {"accuracy": accuracy_score(y_test, preds), "auc": roc_auc_score(y_test, probs)}
    return pipe, res, perf


def fairness_metrics(res):
    rm = res.loc[res.gender == "Male", "predicted"].mean()
    rf = res.loc[res.gender == "Female", "predicted"].mean()
    dir_ratio = rf / rm if rm > 0 else np.nan
    return rm, rf, dir_ratio


# ----------------------------------------------------------------------
st.title("🧭 AI-Based Workplace Gender Equality Analytics")
st.caption("SDG 5 — Gender Equality  |  Minor Project Dashboard")

df = load_data()

with st.sidebar:
    st.header("Filters")
    dept_filter = st.multiselect("Department", sorted(df.department.unique()),
                                  default=list(df.department.unique()))
    df = df[df.department.isin(dept_filter)]
    st.metric("Employees in view", len(df))

col1, col2, col3, col4 = st.columns(4)
m, f, gap_pct, pval = pay_gap_overall(df)
col1.metric("Avg Male Salary", f"₹{m:,.0f}")
col2.metric("Avg Female Salary", f"₹{f:,.0f}")
col3.metric("Pay Gap", f"{gap_pct:.1f}%", delta=f"{'significant' if pval < 0.05 else 'not significant'}")
col4.metric("Women in Workforce", f"{(df.gender=='Female').mean()*100:.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Representation", "💰 Pay Equity", "📈 Promotion & Attrition", "🤖 AI Bias Audit"]
)

with tab1:
    st.subheader("Gender Representation by Job Level")
    rep = df.groupby(["job_level", "gender"]).size().unstack(fill_value=0)
    rep = rep.reindex(["Associate", "Senior", "Lead", "Manager", "Director"])
    rep_pct = rep.div(rep.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    rep_pct.plot(kind="bar", stacked=True, ax=ax, color=["#e07a5f", "#3d5a80"])
    ax.set_ylabel("%")
    st.pyplot(fig)
    st.caption("A shrinking female share at higher levels indicates a 'leaky pipeline' — "
               "women are represented at entry level but drop off in seniority.")

with tab2:
    st.subheader("Pay Gap by Job Level")
    pay_lvl = df.groupby(["job_level", "gender"])["annual_salary"].mean().unstack()
    pay_lvl = pay_lvl.reindex(["Associate", "Senior", "Lead", "Manager", "Director"])
    fig, ax = plt.subplots(figsize=(7, 4))
    pay_lvl.plot(kind="bar", ax=ax, color=["#e07a5f", "#3d5a80"])
    ax.set_ylabel("Average Salary")
    st.pyplot(fig)
    st.dataframe(pay_lvl.style.format("{:,.0f}"))

with tab3:
    st.subheader("Promotion & Attrition Rates by Gender")
    promo = df.groupby("gender")["promoted_last_cycle"].mean() * 100
    attr = df.groupby("gender")["exited_last_12mo"].mean() * 100
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        promo.plot(kind="bar", ax=ax, color=["#e07a5f", "#3d5a80"])
        ax.set_title("Promotion rate (%)")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        attr.plot(kind="bar", ax=ax, color=["#e07a5f", "#3d5a80"])
        ax.set_title("Attrition rate (%)")
        st.pyplot(fig)

with tab4:
    st.subheader("ML Promotion Model — Fairness / Bias Audit")
    st.write(
        "A logistic regression model is trained to predict promotion "
        "using only job-relevant features (experience, performance, "
        "level, education, department). **Gender is withheld** from "
        "the model's inputs. We then check whether its predictions "
        "still differ by gender — evidence that other features are "
        "acting as proxies for gender."
    )
    model, res, perf = train_model(df)
    rm, rf, dir_ratio = fairness_metrics(res)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{perf['accuracy']*100:.1f}%")
    c2.metric("Predicted Promotion Rate (Male)", f"{rm*100:.1f}%")
    c3.metric("Predicted Promotion Rate (Female)", f"{rf*100:.1f}%")

    st.metric("Disparate Impact Ratio (4/5ths rule)", f"{dir_ratio:.2f}",
              delta="PASS (≥0.8)" if dir_ratio >= 0.8 else "FAIL (<0.8) — adverse impact risk")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Male", "Female"], [rm*100, rf*100], color=["#3d5a80", "#e07a5f"])
    ax.axhline(rm*100*0.8, ls="--", color="gray", label="4/5ths threshold")
    ax.legend()
    ax.set_ylabel("% predicted promoted")
    st.pyplot(fig)

st.divider()
st.caption("Synthetic demo data — built for an academic minor project mapped to UN SDG 5 (Gender Equality).")
