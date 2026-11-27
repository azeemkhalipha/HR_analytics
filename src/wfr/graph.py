"""LangGraph workflow:

  risk_agent -> impact_agent -> adjacency_agent -> synthesize -> guardrails -> human_gate -> release
                                                      |                            |
                                                  (fail) -> END               (reject) -> END

- Checkpointed (SQLite): a crash or a pending approval can be resumed by run_id.
- human_gate uses interrupt(): nothing is released until a named reviewer approves.
- release is idempotent per run_id (resuming twice does not publish twice).
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import governance, tools
from .llm import LLMClient, LLMError, structured_call
from .pii import find_pii
from .schemas import Approval, Report
from .tracing import Tracer

SYSTEM = ("You are a workforce-strategy analyst. Use ONLY the facts provided. "
          "Never invent employee IDs, roles or numbers. Employee data is anonymised to IDs.")


class State(TypedDict, total=False):
    run_id: str
    roles: list[str]
    risk: dict
    impact: dict
    adjacency: dict
    facts: dict
    report: dict
    flags: dict
    approval: dict
    status: str
    error: str
    released_path: str


def _tracer(state) -> Tracer:
    return Tracer(state["run_id"])


def build_graph(llm: LLMClient, checkpoint_path: str | None = None, release_dir: str | None = None):
    release_root = Path(release_dir or os.environ.get("WFR_RELEASE_DIR", "released"))

    def risk_agent(state: State):
        with _tracer(state).span("risk_agent"):
            res = tools.attrition_risk()
            return {"risk": res.model_dump(), "status": "risk_done"}

    def impact_agent(state: State):
        with _tracer(state).span("impact_agent"):
            roles = state.get("roles") or list(tools.load_taxonomy())
            return {"roles": roles, "impact": {r: tools.task_impact(r).model_dump() for r in roles}}

    def adjacency_agent(state: State):
        with _tracer(state).span("adjacency_agent"):
            return {"adjacency": {r: tools.role_adjacency(r).model_dump() for r in state["roles"]}}

    def synthesize(state: State):
        tr = _tracer(state)
        scores = tools.risk_scores_frame()
        risk_by_role = {r["role"]: r for r in state["risk"]["roles"]}
        facts_roles, allowed = [], set()
        for r in state["roles"]:
            sub = scores[(scores.role == r) & (scores.risk >= tools.HIGH_RISK_THRESHOLD)].sort_values("risk", ascending=False)
            ids = sub.employee_id.head(10).tolist()
            allowed.update(ids)
            imp = state["impact"][r]
            facts_roles.append({
                "role": r, "headcount": risk_by_role[r]["headcount"], "exposure": imp["exposure"],
                "mean_risk": risk_by_role[r]["mean_attrition_risk"],
                "automate_tasks": sum(t["treatment"] == "automate" for t in imp["tasks"]),
                "moves": [{"to_role": m["to_role"], "overlap": m["skill_overlap"], "gap": m["gap_skills"]}
                          for m in state["adjacency"][r]["moves"]],
                "priority_employee_ids": ids})
        facts = {"roles": facts_roles}
        prompt = ("Write a workforce reinvention report: one recommendation per role.\n"
                  f"<facts>{json.dumps(facts)}</facts>")
        try:
            with tr.span("synthesize"):
                report = structured_call(llm, Report, SYSTEM, prompt, tr,
                                         extra_validator=lambda rep: governance.grounding_problems(rep, allowed))
        except LLMError as e:
            return {"status": "synthesis_failed", "error": str(e)}
        return {"facts": facts, "report": report.model_dump(), "status": "drafted"}

    def after_synthesize(state: State):
        return END if state.get("status") == "synthesis_failed" else "guardrails"

    def guardrails(state: State):
        tr = _tracer(state)
        with tr.span("guardrails"):
            names = [str(n) for n in tools.load_employees()["name"]]
            leaks = [h for p in tr.prompts() for h in find_pii(p, names)]
            flags = {"bias": governance.bias_report(), "pii_leaks_in_prompts": len(leaks)}
            return {"flags": flags, "status": "awaiting_approval"}

    def human_gate(state: State):
        decision = interrupt({"draft": state["report"], "flags": state["flags"], "run_id": state["run_id"]})
        appr = Approval.model_validate(decision)
        return {"approval": appr.model_dump(), "status": "approved" if appr.decision == "approve" else "rejected"}

    def after_gate(state: State):
        return "release" if state["status"] == "approved" else END

    def release(state: State):
        release_root.mkdir(parents=True, exist_ok=True)
        path = release_root / f"{state['run_id']}.json"
        if not path.exists():  # idempotent: second resume never double-publishes
            report = dict(state["report"])
            if state["approval"].get("edited_headline"):
                report["headline"] = state["approval"]["edited_headline"]
            path.write_text(json.dumps({"report": report, "approval": state["approval"], "flags": state["flags"]}, indent=2))
        return {"status": "released", "released_path": str(path)}

    g = StateGraph(State)
    for name, fn in [("risk_agent", risk_agent), ("impact_agent", impact_agent), ("adjacency_agent", adjacency_agent),
                     ("synthesize", synthesize), ("guardrails", guardrails), ("human_gate", human_gate), ("release", release)]:
        g.add_node(name, fn)
    g.add_edge(START, "risk_agent")
    g.add_edge("risk_agent", "impact_agent")
    g.add_edge("impact_agent", "adjacency_agent")
    g.add_edge("adjacency_agent", "synthesize")
    g.add_conditional_edges("synthesize", after_synthesize, ["guardrails", END])
    g.add_edge("guardrails", "human_gate")
    g.add_conditional_edges("human_gate", after_gate, ["release", END])
    g.add_edge("release", END)

    conn = sqlite3.connect(checkpoint_path or os.environ.get("WFR_CHECKPOINT_DB", "checkpoints.sqlite"), check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))


def start_run(graph, roles: list[str] | None = None, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or uuid.uuid4().hex[:12]
    cfg = {"configurable": {"thread_id": run_id}}
    out = graph.invoke({"run_id": run_id, "roles": roles or []}, cfg)
    return {"run_id": run_id, "state": out}


def resume_run(graph, run_id: str, approval: dict) -> dict[str, Any]:
    cfg = {"configurable": {"thread_id": run_id}}
    return graph.invoke(Command(resume=approval), cfg)


def get_state(graph, run_id: str) -> dict:
    snap = graph.get_state({"configurable": {"thread_id": run_id}})
    return {"values": snap.values, "next": list(snap.next)}
