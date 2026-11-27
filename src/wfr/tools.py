"""Deterministic analytics tools. Used directly by the graph and exposed over MCP.

Idempotency: every tool call goes through `cached`, keyed on (tool, args). Re-running the
same call (e.g. after a crash/resume) returns the same result and does no duplicate work.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .pii import strip_direct_identifiers
from .schemas import AdjacencyResult, AdjacentMove, ImpactResult, RiskResult, RoleImpact, RoleRisk, TaskImpact

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["tenure_years", "performance", "salary_ratio", "months_since_promo",
            "engagement", "overtime", "manager_changes"]  # gender/age deliberately excluded
HIGH_RISK_THRESHOLD = 0.5
MIN_ADJACENCY_OVERLAP = 0.15  # below this a move is not a realistic reskilling path

_CALL_CACHE: dict[str, str] = {}


def cached(tool: str, args: dict, fn):
    key = hashlib.sha256(json.dumps([tool, args], sort_keys=True).encode()).hexdigest()
    if key not in _CALL_CACHE:
        _CALL_CACHE[key] = fn().model_dump_json()
    return _CALL_CACHE[key]


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads((ROOT / "data" / "taxonomy.json").read_text())["roles"]


@lru_cache(maxsize=1)
def load_employees() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "employees.csv")


def _risk_frame() -> tuple[pd.DataFrame, float]:
    """Out-of-fold risk scores (no leakage) + CV AUC."""
    df = strip_direct_identifiers(load_employees()).copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    proba = cross_val_predict(model, df[FEATURES], df["attrited"], cv=5, method="predict_proba")[:, 1]
    df["risk"] = proba
    return df, float(roc_auc_score(df["attrited"], proba))


def _drivers() -> list[str]:
    df = strip_direct_identifiers(load_employees())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[FEATURES], df["attrited"])
    coef = model[-1].coef_[0]
    order = np.argsort(-np.abs(coef))[:3]
    return [f"{FEATURES[i]} ({'+' if coef[i] > 0 else '-'})" for i in order]


def attrition_risk(role: str | None = None) -> RiskResult:
    def run():
        df, auc = _risk_frame()
        drivers = _drivers()
        roles = []
        for r, g in df.groupby("role"):
            if role and r != role:
                continue
            roles.append(RoleRisk(role=r, headcount=len(g), mean_attrition_risk=round(float(g.risk.mean()), 4),
                                  high_risk_count=int((g.risk >= HIGH_RISK_THRESHOLD).sum()), top_drivers=drivers))
        sub = df if not role else df[df.role == role]
        ids = sub.loc[sub.risk >= HIGH_RISK_THRESHOLD, "employee_id"].tolist()
        return RiskResult(model_auc_cv=round(auc, 4), threshold=HIGH_RISK_THRESHOLD, roles=roles, high_risk_employee_ids=ids)

    return RiskResult.model_validate_json(cached("attrition_risk", {"role": role}, run))


def risk_scores_frame() -> pd.DataFrame:
    """Per-employee scores for governance checks (bias audit). Internal use, not sent to the LLM."""
    df, _ = _risk_frame()
    return df


def task_impact(role: str) -> RoleImpact:
    tax = load_taxonomy()
    if role not in tax:
        raise KeyError(f"unknown role: {role}")

    def run():
        tasks = []
        for t in tax[role]["tasks"]:
            a = t["automation"]
            treat = "automate" if a >= 0.65 else "augment" if a >= 0.35 else "retain"
            tasks.append(TaskImpact(task=t["name"], share=t["share"], automation_score=a, treatment=treat,
                                    rationale=f"Automation score {a:.2f} from taxonomy; treatment = {treat}."))
        exposure = round(sum(t.share * t.automation_score for t in tasks), 4)
        return RoleImpact(role=role, exposure=exposure, tasks=tasks)

    return RoleImpact.model_validate_json(cached("task_impact", {"role": role}, run))


def _role_skills(role: str) -> set[str]:
    return {s for t in load_taxonomy()[role]["tasks"] for s in t["skills"]}


def role_adjacency(role: str, top_k: int = 2) -> AdjacencyResult:
    tax = load_taxonomy()
    if role not in tax:
        raise KeyError(f"unknown role: {role}")

    def run():
        src = _role_skills(role)
        moves = []
        for other in tax:
            if other == role:
                continue
            tgt = _role_skills(other)
            jac = len(src & tgt) / len(src | tgt)
            # Prefer moves to roles with LOWER automation exposure than the source.
            moves.append((jac, other, sorted(tgt - src)))
        moves.sort(key=lambda m: -m[0])
        src_exp = task_impact(role).exposure
        picked = [m for m in moves if m[0] >= MIN_ADJACENCY_OVERLAP and task_impact(m[1]).exposure < src_exp][:top_k]
        return AdjacencyResult(moves=[AdjacentMove(from_role=role, to_role=o, skill_overlap=round(j, 4), gap_skills=g)
                                      for j, o, g in picked])

    return AdjacencyResult.model_validate_json(cached("role_adjacency", {"role": role, "top_k": top_k}, run))


def skills_gap(employee_id: str, target_role: str) -> dict:
    df = load_employees()
    row = df[df.employee_id == employee_id]
    if row.empty:
        raise KeyError(f"unknown employee: {employee_id}")
    have = set(row.iloc[0].skills.split("|"))
    need = _role_skills(target_role)
    return {"employee_id": employee_id, "target_role": target_role,
            "missing": sorted(need - have), "coverage": round(len(need & have) / len(need), 3)}


# ---- lightweight retrieval (RAG over the role taxonomy) -------------------------------
@lru_cache(maxsize=1)
def _index():
    tax = load_taxonomy()
    names = list(tax)
    docs = [f"{n}. {tax[n]['description']} " + " ".join(t["name"] + " " + " ".join(t["skills"]) for t in tax[n]["tasks"])
            for n in names]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(docs)
    return names, docs, vec, vec.transform(docs)


def retrieve_roles(query: str, k: int = 3) -> list[dict]:
    names, docs, vec, mat = _index()
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    order = np.argsort(-sims)[:k]
    return [{"role": names[i], "score": round(float(sims[i]), 4), "snippet": docs[i][:200]} for i in order]
