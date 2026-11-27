"""Golden-set evals with a regression gate.

    python evals/run_evals.py                 # compute metrics, compare to evals/baseline.json, exit 1 on regression
    python evals/run_evals.py --update-baseline

Every metric is deterministic (mock LLM, seeded data) so CI can fail the build on any drop.
Golden expectations below come from domain judgement, NOT from tuning to the current output.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfr import governance, graph, tools  # noqa: E402
from wfr.llm import LLMError, MockLLM, structured_call  # noqa: E402
from wfr.schemas import Report  # noqa: E402
from wfr.tracing import Tracer  # noqa: E402

BASELINE = ROOT / "evals" / "baseline.json"
TOLERANCE = {"risk_auc": 0.02}  # continuous metrics may wobble slightly; rates must not drop at all
HARD_FLOORS = {"risk_auc": 0.65}

# --- goldens -----------------------------------------------------------------------------
EXPOSURE_ORDER = [  # (more exposed, less exposed): routine admin work vs judgement/relationship work
    ("HR Operations Analyst", "HR Business Partner"),
    ("HR Operations Analyst", "AI Solutions Consultant"),
    ("Compensation & Benefits Analyst", "HR Business Partner"),
    ("Data Analyst", "AI Solutions Consultant"),
]
ADJACENCY = {  # role -> acceptable lower-exposure destinations (any one in top-2 counts)
    "Data Analyst": {"Data Scientist", "People Analytics Analyst", "Workforce Planning Analyst"},
    "HR Operations Analyst": {"Compensation & Benefits Analyst", "HR Business Partner", "People Analytics Analyst", "Learning & Development Specialist"},
    "Compensation & Benefits Analyst": {"People Analytics Analyst", "Workforce Planning Analyst", "HR Business Partner"},
}
RETRIEVAL = [
    ("processing payroll inputs and onboarding paperwork for new hires", "HR Operations Analyst"),
    ("build LLM agents and RAG pipelines for clients", "AI Solutions Consultant"),
    ("source and screen candidates, run interviews", "Talent Acquisition Recruiter"),
    ("forecast headcount and map skills supply and demand", "Workforce Planning Analyst"),
]


def _isolated_graph(llm):
    d = tempfile.mkdtemp()
    os.environ["WFR_TRACE_DIR"] = f"{d}/traces"
    return graph.build_graph(llm, checkpoint_path=f"{d}/ck.sqlite", release_dir=f"{d}/rel"), d


def compute() -> dict:
    m: dict = {}
    m["risk_auc"] = tools.attrition_risk().model_auc_cv

    m["impact_ordering"] = sum(tools.task_impact(a).exposure > tools.task_impact(b).exposure for a, b in EXPOSURE_ORDER) / len(EXPOSURE_ORDER)

    hits = 0
    for role, ok in ADJACENCY.items():
        hits += any(mv.to_role in ok for mv in tools.role_adjacency(role).moves)
    m["adjacency_hit_rate"] = hits / len(ADJACENCY)

    m["retrieval_top1"] = sum(tools.retrieve_roles(q, 1)[0]["role"] == exp for q, exp in RETRIEVAL) / len(RETRIEVAL)

    # structured output: model fails the first call every time; retry must recover 100% of runs
    ok = 0
    for i in range(20):
        g, _ = _isolated_graph(MockLLM(fail_first_n=1))
        out = graph.start_run(g, run_id=f"flaky{i}")["state"]
        ok += out.get("status") == "awaiting_approval"
    m["schema_valid_after_retry"] = ok / 20

    # grounding: a hallucinating model must be BLOCKED, never silently released
    g, _ = _isolated_graph(MockLLM(hallucinate_ids=True))
    out = graph.start_run(g, run_id="halluc")["state"]
    m["hallucination_blocked"] = float(out.get("status") == "synthesis_failed" and "__interrupt__" not in out)

    # PII: no direct identifier may appear in any prompt; HITL: nothing released before approval
    g, d = _isolated_graph(MockLLM())
    out = graph.start_run(g, run_id="pii")["state"]
    m["pii_leaks"] = out["flags"]["pii_leaks_in_prompts"]
    before = list(Path(d, "rel").glob("*.json"))
    graph.resume_run(g, "pii", {"decision": "reject", "reviewer": "QA Bot"})
    m["hitl_no_release_before_or_on_reject"] = float(not before and not list(Path(d, "rel").glob("*.json")))
    g, d = _isolated_graph(MockLLM())
    graph.start_run(g, run_id="ok")
    graph.resume_run(g, "ok", {"decision": "approve", "reviewer": "QA Bot"})
    graph.resume_run(g, "ok", {"decision": "approve", "reviewer": "QA Bot"})
    m["release_idempotent"] = float(len(list(Path(d, "rel").glob("*.json"))) == 1)

    bias = governance.bias_report()["overall"]
    m["bias_not_failing"] = float(bias in ("pass", "review"))
    m["_bias_overall"] = bias  # informational, not gated
    return m


def check(m: dict, base: dict) -> list[str]:
    fails = []
    for k, v in m.items():
        if k.startswith("_"):
            continue
        if k == "pii_leaks":
            if v != 0:
                fails.append(f"{k}={v} (must be 0)")
            continue
        if k in HARD_FLOORS and v < HARD_FLOORS[k]:
            fails.append(f"{k}={v:.4f} below hard floor {HARD_FLOORS[k]}")
        if k in base and v < base[k] - TOLERANCE.get(k, 0.0):
            fails.append(f"{k}={v:.4f} regressed from baseline {base[k]:.4f}")
    return fails


def main() -> int:
    m = compute()
    print(json.dumps(m, indent=2))
    if "--update-baseline" in sys.argv:
        BASELINE.write_text(json.dumps({k: v for k, v in m.items() if not k.startswith("_")}, indent=2))
        print("baseline updated")
        return 0
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    fails = check(m, base)
    for f in fails:
        print("FAIL:", f)
    print("EVALS", "FAILED" if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
