"""
generate_data.py
-----------------
Generates a synthetic HR / workforce dataset used by the
"AI-Based Workplace Gender Equality Analytics" minor project.

The data is FAKE (no real company/employee), but it is generated with
realistic, literature-backed patterns of workplace gender gaps
(pay gap, promotion gap, leadership under-representation) so the
analytics and ML modules have something meaningful to detect.

Run:
    python3 generate_data.py
Produces:
    data/employees.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1200

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
LEVELS = ["Associate", "Senior", "Lead", "Manager", "Director"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
EDUCATION = ["Bachelors", "Masters", "PhD"]

def generate_employees(n=N):
    gender = RNG.choice(["Male", "Female"], size=n, p=[0.58, 0.42])
    department = RNG.choice(DEPARTMENTS, size=n)
    education = RNG.choice(EDUCATION, size=n, p=[0.6, 0.32, 0.08])
    experience_years = np.clip(RNG.normal(7, 4, n), 0, 30).round(1)
    age = np.clip((experience_years + RNG.normal(24, 3, n)), 21, 62).round().astype(int)

    performance_rating = np.clip(RNG.normal(3.4, 0.6, n), 1, 5).round(1)

    # --- Baked-in structural pattern: women are progressively under-represented
    # at higher levels (a real, widely-documented "leaky pipeline" effect). ---
    level_probs_male = np.array([0.30, 0.27, 0.20, 0.15, 0.08])
    level_probs_female = np.array([0.38, 0.28, 0.18, 0.11, 0.05])
    level = np.empty(n, dtype=object)
    for i in range(n):
        probs = level_probs_male if gender[i] == "Male" else level_probs_female
        level[i] = RNG.choice(LEVELS, p=probs)

    level_rank = np.array([LEVEL_RANK[l] for l in level])

    # --- Base salary model: experience, performance, level all matter ---
    base = 40000 + level_rank * 18000 + experience_years * 1500 + performance_rating * 3000
    noise = RNG.normal(0, 4000, n)

    # --- Baked-in pattern: a residual gender pay gap (~6-9%) even after
    # controlling for level/experience/performance — this is what the
    # analytics + ML pipeline is designed to surface. ---
    gender_penalty = np.where(gender == "Female", RNG.normal(-0.07, 0.02, n), 0.0)
    salary = (base * (1 + gender_penalty) + noise).round(0)

    # --- Promotion in the last review cycle ---
    promotion_logit = (
        -1.2
        + 0.35 * (performance_rating - 3)
        + 0.05 * experience_years
        - 0.25 * (gender == "Female")   # harder to get promoted, all else equal
    )
    promotion_prob = 1 / (1 + np.exp(-promotion_logit))
    promoted = RNG.random(n) < promotion_prob

    # --- Attrition (left the company in last 12 months) ---
    attrition_logit = (
        -2.0
        + 0.15 * (gender == "Female")  # slightly higher attrition
        - 0.10 * (performance_rating - 3)
    )
    attrition_prob = 1 / (1 + np.exp(-attrition_logit))
    exited = RNG.random(n) < attrition_prob

    hire_year = (2026 - experience_years - RNG.integers(0, 2, n)).round().astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i+1:05d}" for i in range(n)],
        "gender": gender,
        "department": department,
        "job_level": level,
        "education": education,
        "age": age,
        "experience_years": experience_years,
        "performance_rating": performance_rating,
        "annual_salary": salary,
        "promoted_last_cycle": promoted,
        "exited_last_12mo": exited,
        "hire_year": hire_year,
    })
    return df


if __name__ == "__main__":
    df = generate_employees()
    df.to_csv("data/employees.csv", index=False)
    print(f"Generated {len(df)} employee records -> data/employees.csv")
    print(df.head())
