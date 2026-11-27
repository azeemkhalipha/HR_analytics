"""Generate SYNTHETIC employee data. No real people. Seeded for reproducibility.

The attrition label comes from a hidden logistic function plus noise, so the
model has real signal to learn. Gender/age are included on purpose so the
bias check has something to audit; they do NOT drive the hidden function.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
random.seed(42)

tax = json.loads((ROOT / "data" / "taxonomy.json").read_text())
roles = list(tax["roles"].keys())
role_skills = {r: sorted({s for t in v["tasks"] for s in t["skills"]}) for r, v in tax["roles"].items()}

FIRST = ["Aarav", "Diya", "Rohan", "Meera", "Kabir", "Isha", "Vikram", "Ananya", "Arjun", "Neha",
         "Sameer", "Priya", "Rahul", "Sneha", "Imran", "Pooja", "Karan", "Tara", "Nikhil", "Riya"]
LAST = ["Shah", "Iyer", "Khan", "Patil", "Nair", "Gupta", "Joshi", "Rao", "Mehta", "Singh"]

N = 600
rows = []
for i in range(N):
    role = random.choice(roles)
    tenure = round(float(rng.gamma(2.2, 1.6)), 1)
    age = int(np.clip(rng.normal(34, 7), 22, 60))
    gender = random.choice(["F", "M"])
    perf = int(np.clip(rng.normal(3.3, 0.9), 1, 5))
    salary_ratio = float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.4))  # pay vs market
    months_since_promo = int(np.clip(rng.gamma(2.0, 10), 0, 96))
    engagement = float(np.clip(rng.normal(3.6, 0.8), 1, 5))
    overtime = int(rng.random() < 0.3)
    manager_changes = int(rng.poisson(0.6))
    known = random.sample(role_skills[role], k=max(2, int(len(role_skills[role]) * rng.uniform(0.5, 1.0))))
    # extra adjacent skills
    other = [s for r in roles for s in role_skills[r] if s not in role_skills[role]]
    known += random.sample(other, k=int(rng.integers(0, 4)))
    logit = (-1.9 + 1.1 * (1.0 - salary_ratio) * 4 - 0.7 * (engagement - 3.5)
             + 0.025 * months_since_promo + 0.6 * overtime + 0.25 * manager_changes
             - 0.08 * tenure + 0.2 * (perf >= 4))
    p = 1 / (1 + np.exp(-logit))
    left = int(rng.random() < p)
    rows.append(dict(
        employee_id=f"E{i:04d}", name=f"{random.choice(FIRST)} {random.choice(LAST)}",
        email=f"user{i}@example-corp.com", phone=f"+91-9{rng.integers(10**8, 10**9 - 1)}",
        role=role, tenure_years=tenure, age=age, gender=gender, performance=perf,
        salary_ratio=round(salary_ratio, 3), months_since_promo=months_since_promo,
        engagement=round(engagement, 2), overtime=overtime, manager_changes=manager_changes,
        skills="|".join(sorted(set(known))), attrited=left))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "data" / "employees.csv", index=False)
print(df.shape, "attrition rate:", round(df.attrited.mean(), 3))
