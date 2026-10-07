"""
app.py
------
AI-Based Workplace Gender Equality Analytics — Interactive Dashboard.

What's new vs v1:
    ✅ Credentials loaded from environment variables (secure; demo fallback kept)
    ✅ CSV file upload — analyse your OWN workforce data
    ✅ Download buttons for every chart and table
    ✅ Department-wise pay gap tab (new Tab 3)
    ✅ Extended fairness audit: Equalized Odds (TPR/FPR gap) displayed
    ✅ ML model comparison table (Logistic Regression vs Random Forest)
    ✅ Raw data viewer with filtered CSV download
    ✅ All analytics imported from analytics.py (DRY — no duplicate logic)
    ✅ User-friendly error handling with st.error / st.stop

Run with:
    streamlit run app.py

Credentials via environment variables (optional — demo values used as fallback):
    ADMIN_USERNAME   ADMIN_PASSWORD
    ANALYST_USERNAME ANALYST_PASSWORD
"""

import io
import json
import os
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analytics import (
    validate_data,
    pay_gap_overall,
    representation_by_level,
    pay_gap_by_level,
    pay_gap_by_department,
    promotion_rates,
    attrition_rates,
    train_promotion_model,
    fairness_metrics,
    save_model,
    LEVEL_ORDER,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workplace Gender Equality Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Multi-User Database Init
# ---------------------------------------------------------------------------
import sqlite3

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            name TEXT,
            role TEXT,
            scope TEXT
        )
    ''')
    
    # Check if admin exists, if not create demo accounts
    c.execute('SELECT username FROM users WHERE username = "admin"')
    if not c.fetchone():
        # Demo admin
        admin_hash = hashlib.sha256(os.getenv("ADMIN_PASSWORD", "admin123").encode()).hexdigest()
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        c.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?)', 
                  (admin_user, admin_hash, "Dr. Aris Thorne", "HR Director / Admin", "Full executive privileges & bias audit"))
        
        # Demo analyst
        analyst_hash = hashlib.sha256(os.getenv("ANALYST_PASSWORD", "analyst123").encode()).hexdigest()
        analyst_user = os.getenv("ANALYST_USERNAME", "analyst")
        c.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?)', 
                  (analyst_user, analyst_hash, "Neha Sharma", "People Analytics Lead", "Analytics & bias audit access"))
    conn.commit()
    conn.close()

init_db()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT username, name, role, scope FROM users WHERE username = ? AND password_hash = ?', 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    if user:
        return {"username": user[0], "name": user[1], "role": user[2], "scope": user[3]}
    return None

def register_new_user(username, password, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password_hash, name, role, scope) VALUES (?, ?, ?, ?, ?)', 
                  (username, hash_password(password), name, "User", "Standard Analytics"))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _load_default_csv(path: str = "data/employees.csv") -> pd.DataFrame:
    """Load and cache the default synthetic dataset."""
    return pd.read_csv(path)


def _load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded file, validate it, and return a DataFrame."""
    df = pd.read_csv(uploaded_file)
    validate_data(df)       # raises ValueError on bad data
    return df


def _df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame — used as cache key for the model."""
    h = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    return h


# ---------------------------------------------------------------------------
# Cached model training
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="🤖 Training ML models…")
def _get_model_results(df_hash_key: str, _df: pd.DataFrame):
    """
    Train + cache models. df_hash_key (str) drives cache invalidation;
    _df (underscore prefix) is excluded from Streamlit hashing.
    """
    model, result_df, model_results = train_promotion_model(_df)
    save_model(model)
    return model, result_df, model_results


# ---------------------------------------------------------------------------
# Chart-to-bytes helper (for download buttons)
# ---------------------------------------------------------------------------

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def render_login():
    # Inject CSS for gradient background and glassmorphism (Only on login page)
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #8e2de2 100%) !important;
    }
    .stAppHeader {
        background-color: transparent !important;
    }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        padding: 2rem;
    }
    .stMarkdownContainer, .stMarkdownContainer p, h1, h2, h3, label {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; margin-top: 40px;'>",
            unsafe_allow_html=True,
        )
        st.title("🏢 Workplace Gender Equality Portal")
        st.caption("AI-Powered HR Analytics & Bias Detection (SDG 5)")
        st.markdown("</div><br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. admin or analyst")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    user = authenticate_user(username.strip(), password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username.strip()
                        st.session_state["user"] = user
                        st.success(f"Welcome back, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")

            st.markdown("##### 🔑 Demo Credentials")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.info("**Admin**\n- User: `admin`\n- Pass: `admin123`")
            with dc2:
                st.info("**Analyst**\n- User: `analyst`\n- Pass: `analyst123`")

        with tab_register:
            with st.form("register_form"):
                reg_name = st.text_input("Full Name")
                reg_user = st.text_input("Username")
                reg_pass = st.text_input("Password", type="password")
                reg_confirm = st.text_input("Confirm Password", type="password")
                reg_submitted = st.form_submit_button("Register", use_container_width=True)

                if reg_submitted:
                    if not reg_name or not reg_user or not reg_pass:
                        st.error("All fields are required!")
                    elif reg_pass != reg_confirm:
                        st.error("Passwords do not match!")
                    else:
                        success = register_new_user(reg_user.strip(), reg_pass, reg_name.strip())
                        if success:
                            st.success("Registration successful! Please go to the Login tab to sign in.")
                        else:
                            st.error("Username already exists. Please choose a different one.")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    curr_user = st.session_state.get("user", {})
    st.markdown("### 👤 Logged In")
    st.markdown(f"**{curr_user.get('name', 'User')}**")
    st.caption(f"Role: {curr_user.get('role', 'Member')}")
    st.caption(f"Scope: {curr_user.get('scope', 'Standard')}")

    if st.button("🚪 Sign Out", use_container_width=True):
        for key in ["authenticated", "username", "user"]:
            st.session_state[key] = None
        st.session_state["authenticated"] = False
        st.rerun()

    st.divider()

    # --- Data source ---
    st.header("📂 Data Source")
    data_source = st.radio(
        "Choose data source:", ["Default dataset", "Upload your Data"]
    )

    df: pd.DataFrame | None = None
    
    # User specific data directory
    USER_DATA_DIR = "user_data"
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    current_username = st.session_state.get("username", "default")
    user_csv_path = os.path.join(USER_DATA_DIR, f"{current_username}_data.csv")

    if data_source == "Upload your Data":
        uploaded_file = st.file_uploader(
            "Upload employee CSV",
            type=["csv"],
            help=(
                "Required columns: gender, annual_salary, job_level, education, "
                "department, age, experience_years, performance_rating, "
                "promoted_last_cycle, exited_last_12mo"
            ),
        )
        if uploaded_file is not None:
            try:
                df = _load_uploaded_csv(uploaded_file)
                # Save specifically for this user
                df.to_csv(user_csv_path, index=False)
                st.success(f"✅ Data uploaded securely for user '{current_username}'. Loaded {len(df):,} records.")
            except (ValueError, Exception) as exc:
                st.error(f"❌ Invalid data:\n\n{exc}")
                st.stop()
        else:
            if os.path.exists(user_csv_path):
                try:
                    df = pd.read_csv(user_csv_path)
                    st.info(f"Using your previously uploaded data ({len(df):,} records). Upload a new file to overwrite.")
                except Exception as e:
                    st.error(f"Could not load previous data: {e}")
                    st.stop()
            else:
                st.info("⬆️ Please upload a CSV file to continue.")
                st.stop()
    else:
        try:
            df = _load_default_csv()
        except FileNotFoundError:
            st.error(
                "❌ Default dataset not found at `data/employees.csv`.\n\n"
                "Run: `python3 generate_data.py`"
            )
            st.stop()

    st.divider()

    # --- Filters ---
    st.header("🔍 Filters")
    all_depts = sorted(df["department"].unique())
    dept_filter = st.multiselect(
        "Department", all_depts, default=all_depts
    )
    df = df[df["department"].isin(dept_filter)].copy()
    st.metric("Employees in view", f"{len(df):,}")

    if len(df) < 30:
        st.warning("⚠️ Very few records — stats may be unreliable.")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🧭 AI-Based Workplace Gender Equality Analytics")
st.caption("SDG 5 — Gender Equality  |  Minor Project Dashboard")

# ---------------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------------

try:
    pay = pay_gap_overall(df)
except Exception as exc:
    st.error(f"Error computing pay gap: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Male Salary", f"₹{pay['male_avg_salary']:,.0f}")
c2.metric("Avg Female Salary", f"₹{pay['female_avg_salary']:,.0f}")
c3.metric(
    "Pay Gap",
    f"{pay['gap_pct']:.1f}%",
    delta=(
        "significant (p<0.05)"
        if pay["significant_at_5pct"]
        else "not significant"
    ),
)
c4.metric(
    "Women in Workforce",
    f"{(df.gender == 'Female').mean() * 100:.1f}%",
)

st.divider()

# Metrics JSON download (if exists from last analytics.py run)
metrics_path = "outputs/metrics_summary.json"
if os.path.exists(metrics_path):
    with open(metrics_path) as fh:
        st.download_button(
            "⬇️ Download Full Metrics (JSON)",
            data=fh.read(),
            file_name="metrics_summary.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Representation",
    "💰 Pay Equity — Level",
    "🏢 Pay Equity — Dept",
    "📈 Promotion & Attrition",
    "🤖 AI Bias Audit",
    "📋 Data",
])


# ── Tab 1: Representation ──────────────────────────────────────────────────

with tab1:
    st.subheader("Gender Representation by Job Level")
    try:
        _, rep_pct = representation_by_level(df)
        fig, ax = plt.subplots(figsize=(7, 4))
        rep_pct.plot(kind="bar", stacked=True, ax=ax, color=["#e07a5f", "#3d5a80"])
        ax.set_ylabel("%")
        ax.set_xlabel("Job Level")
        ax.legend(title="Gender")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "⬇️ Download Chart",
            _fig_to_bytes(fig),
            "representation_by_level.png",
            "image/png",
            key="dl_rep",
        )
        plt.close(fig)
        st.caption(
            "A shrinking female share at higher levels indicates a 'leaky pipeline' — "
            "women are represented at entry level but drop off in seniority."
        )
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Tab 2: Pay Equity by Level ─────────────────────────────────────────────

with tab2:
    st.subheader("Pay Gap by Job Level")
    try:
        pay_lvl = pay_gap_by_level(df)
        fig, ax = plt.subplots(figsize=(7, 4))
        pay_lvl[["Male", "Female"]].plot(kind="bar", ax=ax, color=["#3d5a80", "#e07a5f"])
        ax.set_ylabel("Average Annual Salary (₹)")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "⬇️ Download Chart",
            _fig_to_bytes(fig),
            "pay_gap_by_level.png",
            "image/png",
            key="dl_pay_lvl_chart",
        )
        plt.close(fig)

        display_cols = [c for c in ["Male", "Female", "gap_pct"] if c in pay_lvl.columns]
        fmt = {c: "{:,.0f}" for c in ["Male", "Female"] if c in pay_lvl.columns}
        if "gap_pct" in pay_lvl.columns:
            fmt["gap_pct"] = "{:.1f}%"
        st.dataframe(pay_lvl[display_cols].style.format(fmt), use_container_width=True)

        st.download_button(
            "⬇️ Download Table (CSV)",
            pay_lvl.to_csv().encode(),
            "pay_gap_by_level.csv",
            "text/csv",
            key="dl_pay_lvl_csv",
        )
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Tab 3: Pay Equity by Department (NEW) ─────────────────────────────────

with tab3:
    st.subheader("Pay Gap by Department")
    st.caption("Breaks down gender pay disparity at the department level.")
    try:
        pay_dept = pay_gap_by_department(df)
        if "gap_pct" not in pay_dept.columns:
            st.warning("Not enough gender data per department to compute gaps.")
        else:
            dept_gap = pay_dept["gap_pct"].sort_values(ascending=False)
            bar_colors = ["#e07a5f" if v > 0 else "#3d5a80" for v in dept_gap]

            fig, ax = plt.subplots(figsize=(8, 5))
            dept_gap.plot(kind="bar", ax=ax, color=bar_colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Pay Gap (%)")
            ax.set_title("Gender Pay Gap by Department\n(positive = men earn more)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button(
                "⬇️ Download Chart",
                _fig_to_bytes(fig),
                "dept_pay_gap.png",
                "image/png",
                key="dl_dept_chart",
            )
            plt.close(fig)

            display_cols_d = [
                c for c in ["Male", "Female", "gap_pct", "gap_abs"]
                if c in pay_dept.columns
            ]
            fmt_d = {}
            for c in ["Male", "Female", "gap_abs"]:
                if c in pay_dept.columns:
                    fmt_d[c] = "{:,.0f}"
            if "gap_pct" in pay_dept.columns:
                fmt_d["gap_pct"] = "{:.1f}%"
            st.dataframe(
                pay_dept[display_cols_d].style.format(fmt_d),
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download Table (CSV)",
                pay_dept.to_csv().encode(),
                "pay_gap_by_dept.csv",
                "text/csv",
                key="dl_dept_csv",
            )
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Tab 4: Promotion & Attrition ──────────────────────────────────────────

with tab4:
    st.subheader("Promotion & Attrition Rates by Gender")
    try:
        promo, promo_test = promotion_rates(df)
        attr = attrition_rates(df)

        pc1, pc2 = st.columns(2)
        with pc1:
            fig, ax = plt.subplots()
            promo.plot(kind="bar", ax=ax, color=["#e07a5f", "#3d5a80"])
            ax.set_title("Promotion Rate (%)")
            ax.set_ylabel("%")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button(
                "⬇️ Download Chart",
                _fig_to_bytes(fig),
                "promotion_rate.png",
                "image/png",
                key="dl_promo",
            )
            plt.close(fig)
        with pc2:
            fig, ax = plt.subplots()
            attr.plot(kind="bar", ax=ax, color=["#e07a5f", "#3d5a80"])
            ax.set_title("Attrition Rate (%)")
            ax.set_ylabel("%")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button(
                "⬇️ Download Chart",
                _fig_to_bytes(fig),
                "attrition_rate.png",
                "image/png",
                key="dl_attr",
            )
            plt.close(fig)

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Male Promotion Rate", f"{promo.get('Male', 0):.1f}%")
        km2.metric("Female Promotion Rate", f"{promo.get('Female', 0):.1f}%")
        km3.metric("Male Attrition Rate", f"{attr.get('Male', 0):.1f}%")
        km4.metric("Female Attrition Rate", f"{attr.get('Female', 0):.1f}%")

        with st.expander("Chi-square test result"):
            st.json(promo_test)
    except Exception as exc:
        st.error(f"Error: {exc}")


# ── Tab 5: AI Bias Audit ──────────────────────────────────────────────────

with tab5:
    st.subheader("ML Promotion Model — Fairness / Bias Audit")
    st.write(
        "**Logistic Regression** and **Random Forest** are both trained on "
        "job-relevant features (experience, performance, level, education, department). "
        "**Gender is deliberately withheld.** "
        "We then check if the model's predictions still differ by gender — "
        "evidence that other features act as proxies for gender."
    )

    try:
        key = _df_hash(df)
        model, res, model_results = _get_model_results(key, df)
        fm = fairness_metrics(res)

        # --- Model Comparison Table ---
        st.markdown("#### 🔬 Model Comparison")
        comp_df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": f"{v['accuracy'] * 100:.1f}%",
                "AUC-ROC": f"{v['auc']:.3f}",
            }
            for name, v in model_results.items()
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Model comparison bar chart
        names = list(model_results.keys())
        accs = [model_results[n]["accuracy"] for n in names]
        aucs = [model_results[n]["auc"] for n in names]
        x = np.arange(len(names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w / 2, accs, w, label="Accuracy", color="#3d5a80")
        ax.bar(x + w / 2, aucs, w, label="AUC-ROC", color="#e07a5f")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("ML Model Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "⬇️ Download Chart",
            _fig_to_bytes(fig),
            "model_comparison.png",
            "image/png",
            key="dl_model_cmp",
        )
        plt.close(fig)

        st.divider()

        # --- Fairness Metrics ---
        st.markdown("#### ⚖️ Fairness Metrics (Disparate Impact + SPD)")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric(
            "Predicted Promotion Rate — Male",
            f"{fm['predicted_promotion_rate_male']:.1f}%",
        )
        fc2.metric(
            "Predicted Promotion Rate — Female",
            f"{fm['predicted_promotion_rate_female']:.1f}%",
        )
        fc3.metric(
            "Disparate Impact Ratio",
            f"{fm['disparate_impact_ratio']:.3f}",
            delta=(
                "✅ PASS (≥ 0.80)"
                if fm["passes_four_fifths_rule"]
                else "❌ FAIL (< 0.80) — adverse impact risk"
            ),
        )
        st.metric(
            "Statistical Parity Difference",
            f"{fm['statistical_parity_difference']:.2f}%",
            help="Difference in predicted promotion rates. 0 = perfect parity.",
        )

        # Bar chart — fairness
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(
            ["Male", "Female"],
            [fm["predicted_promotion_rate_male"], fm["predicted_promotion_rate_female"]],
            color=["#3d5a80", "#e07a5f"],
        )
        ax.axhline(
            fm["predicted_promotion_rate_male"] * 0.8,
            ls="--",
            color="gray",
            label="4/5ths threshold",
        )
        ax.legend()
        ax.set_ylabel("% predicted promoted")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button(
            "⬇️ Download Chart",
            _fig_to_bytes(fig),
            "fairness_audit.png",
            "image/png",
            key="dl_fair",
        )
        plt.close(fig)

        st.divider()

        # --- Equalized Odds (NEW) ---
        st.markdown("#### 🎯 Equalized Odds")
        eq = fm["equalized_odds"]
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric(
            "TPR Gap",
            f"{eq['tpr_gap']:.3f}",
            delta="✅ OK (≤ 0.10)" if eq["tpr_gap"] <= 0.1 else "❌ High bias",
            help="True Positive Rate gap between male and female groups.",
        )
        ec2.metric(
            "FPR Gap",
            f"{eq['fpr_gap']:.3f}",
            delta="✅ OK (≤ 0.10)" if eq["fpr_gap"] <= 0.1 else "❌ High bias",
            help="False Positive Rate gap between male and female groups.",
        )
        ec3.metric(
            "Equalized Odds",
            "✅ PASS" if eq["passes_equalized_odds_01"] else "❌ FAIL",
        )

        with st.expander("📖 What is Equalized Odds?"):
            st.markdown("""
**Equalized Odds** is a fairness criterion that requires the model to have
*equal True Positive Rates* **and** *equal False Positive Rates* across groups.

| Metric | Meaning |
|--------|---------|
| **TPR (Recall)** | Of all employees who *actually* got promoted, what % did the model correctly predict? |
| **FPR** | Of all employees who were *not* promoted, what % did the model wrongly predict as promoted? |

If these rates differ significantly between men and women, the model treats the
groups **unequally** — even though it never directly sees the gender column.
This is a classic example of **proxy discrimination** through correlated features.
""")

        with st.expander("🔢 Full fairness metrics (JSON)"):
            st.json(fm)

    except Exception as exc:
        st.error(f"Error in bias audit: {exc}")


# ── Tab 6: Raw Data ───────────────────────────────────────────────────────

with tab6:
    st.subheader("Employee Dataset (filtered view)")
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        df.to_csv(index=False).encode(),
        "employees_filtered.csv",
        "text/csv",
        key="dl_raw",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Synthetic demo data — built for an academic minor project "
    "mapped to UN SDG 5 (Gender Equality)."
)
