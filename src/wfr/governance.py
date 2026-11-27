"""Governance checks: bias audit on the attrition model + grounding validation for LLM output."""
from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency

from .tools import HIGH_RISK_THRESHOLD, load_employees, load_taxonomy, risk_scores_frame

FOUR_FIFTHS = 0.8


def _band(age: int) -> str:
    return "<30" if age < 30 else "30-45" if age <= 45 else ">45"


def bias_report() -> dict:
    """Compare the rate at which each group is flagged high-risk (gender, age band).

    Gender and age are NOT model inputs (fairness through unawareness), but proxies can leak,
    so we audit outcomes. For each group we report n, flag rate and mean risk. Verdict logic:
      fail   : four-fifths ratio < 0.8 AND the gap is statistically significant (chi-square across all groups, p < 0.05)
      review : ratio < 0.8 but not significant (small groups: evidence is inconclusive; human must look)
      pass   : ratio >= 0.8
    """
    scores = risk_scores_frame()[["employee_id", "risk"]]
    emp = load_employees()[["employee_id", "gender", "age"]]
    df = scores.merge(emp, on="employee_id")
    df["flag"] = df.risk >= HIGH_RISK_THRESHOLD
    df["age_band"] = df.age.map(_band)
    out, verdicts = {}, []
    for col in ["gender", "age_band"]:
        g = df.groupby(col).agg(n=("flag", "size"), flagged=("flag", "sum"), rate=("flag", "mean"), mean_risk=("risk", "mean"))
        rates = {k: round(float(v), 4) for k, v in g.rate.items()}
        ratio = round(float(g.rate.min() / g.rate.max()), 4) if g.rate.max() > 0 else 1.0
        # One omnibus test across ALL groups (picking the two most extreme groups post hoc inflates false positives).
        table = [[int(r.flagged), int(r.n - r.flagged)] for r in g.itertuples()]
        p = float(chi2_contingency(table)[1])
        verdict = "pass" if ratio >= FOUR_FIFTHS else ("fail" if p < 0.05 else "review")
        verdicts.append(verdict)
        out[col] = {"group_sizes": {k: int(v) for k, v in g.n.items()}, "flag_rate": rates,
                    "mean_risk": {k: round(float(v), 4) for k, v in g.mean_risk.items()},
                    "ratio_min_over_max": ratio, "p_value_all_groups": round(p, 4), "verdict": verdict}
    out["overall"] = "fail" if "fail" in verdicts else "review" if "review" in verdicts else "pass"
    out["overall_pass"] = out["overall"] == "pass"
    return out


def grounding_problems(report, allowed_ids: set[str]) -> list[str]:
    """Every employee ID, role and move named in the report must exist in the input facts."""
    roles = set(load_taxonomy())
    problems = []
    for r in report.recommendations:
        if r.role not in roles:
            problems.append(f"unknown role '{r.role}'")
        for m in r.suggested_moves:
            if m not in roles:
                problems.append(f"unknown move target '{m}'")
        bad = [i for i in r.employee_ids_to_prioritise if i not in allowed_ids]
        if bad:
            problems.append(f"ungrounded employee ids {bad} for role '{r.role}'")
    return problems
