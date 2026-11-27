import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from wfr import api, governance, graph, tools  # noqa: E402
from wfr.llm import LLMError, MockLLM, structured_call  # noqa: E402
from wfr.pii import find_pii, mask_text  # noqa: E402
from wfr.schemas import Report  # noqa: E402
from wfr.tracing import Tracer  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WFR_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("WFR_API_KEY", "test-key")
    return tmp_path


def make_graph(tmp, llm=None):
    return graph.build_graph(llm or MockLLM(), checkpoint_path=str(tmp / "ck.sqlite"), release_dir=str(tmp / "rel"))


# ---- tools --------------------------------------------------------------------------
def test_exposure_in_range_and_weights_sum():
    for role, spec in tools.load_taxonomy().items():
        assert abs(sum(t["share"] for t in spec["tasks"]) - 1.0) < 1e-9, role
        assert 0 <= tools.task_impact(role).exposure <= 1


def test_adjacency_moves_go_to_lower_exposure():
    for role in tools.load_taxonomy():
        src = tools.task_impact(role).exposure
        for mv in tools.role_adjacency(role).moves:
            assert tools.task_impact(mv.to_role).exposure < src


def test_unknown_role_raises():
    with pytest.raises(KeyError):
        tools.task_impact("Wizard")


def test_tool_calls_are_idempotent():
    assert tools.attrition_risk().model_dump() == tools.attrition_risk().model_dump()


def test_model_excludes_protected_attributes():
    assert "gender" not in tools.FEATURES and "age" not in tools.FEATURES


# ---- llm / structured output ---------------------------------------------------------
def test_retry_recovers_from_invalid_json(env):
    tr = Tracer("t1")
    role = "HR Operations Analyst"
    facts = {"roles": [{"role": role, "headcount": 5, "exposure": 0.7, "mean_risk": 0.3, "automate_tasks": 2,
                        "moves": [], "priority_employee_ids": ["E0001"]}]}
    out = structured_call(MockLLM(fail_first_n=2), Report, "s", f"<facts>{json.dumps(facts)}</facts>", tr)
    assert out.recommendations[0].role == role
    assert tr.summary()["llm_retries"] == 2


def test_gives_up_after_max_attempts(env):
    with pytest.raises(LLMError):
        structured_call(MockLLM(fail_first_n=99), Report, "s", '<facts>{"roles": []}</facts>', Tracer("t2"), max_attempts=2)


# ---- pii -----------------------------------------------------------------------------
def test_pii_masking():
    t = mask_text("mail bob@x.com or +91-9876543210, ask Priya", known_names=["Priya"])
    assert "bob@x.com" not in t and "9876543210" not in t and "Priya" not in t
    assert find_pii("clean text about roles") == []


# ---- governance ----------------------------------------------------------------------
def test_bias_report_shape():
    r = governance.bias_report()
    assert r["overall"] in {"pass", "review", "fail"}
    assert set(r["gender"]["group_sizes"]) == {"F", "M"}


# ---- graph: human in the loop ----------------------------------------------------------
def test_gate_blocks_until_approved_then_releases_once(env):
    g = make_graph(env)
    st = graph.start_run(g, run_id="r1")["state"]
    assert st["status"] == "awaiting_approval" and "__interrupt__" in st
    assert not list((env / "rel").glob("*.json"))
    out = graph.resume_run(g, "r1", {"decision": "approve", "reviewer": "Test Reviewer", "edited_headline": "Edited by human reviewer"})
    assert out["status"] == "released"
    assert json.loads(Path(out["released_path"]).read_text())["report"]["headline"] == "Edited by human reviewer"


def test_reject_does_not_release(env):
    g = make_graph(env)
    graph.start_run(g, run_id="r2")
    out = graph.resume_run(g, "r2", {"decision": "reject", "reviewer": "Test Reviewer"})
    assert out["status"] == "rejected" and not list((env / "rel").glob("*.json"))


def test_hallucinated_ids_are_blocked(env):
    g = make_graph(env, MockLLM(hallucinate_ids=True))
    st = graph.start_run(g, run_id="r3")["state"]
    assert st["status"] == "synthesis_failed" and "__interrupt__" not in st


def test_run_survives_restart(env):
    """Checkpointing: a brand-new graph object (simulated process restart) can resume a pending approval."""
    g1 = make_graph(env)
    graph.start_run(g1, run_id="r4")
    g2 = make_graph(env)
    out = graph.resume_run(g2, "r4", {"decision": "approve", "reviewer": "Test Reviewer"})
    assert out["status"] == "released"


def test_prompts_contain_no_direct_identifiers(env):
    g = make_graph(env)
    graph.start_run(g, run_id="r5")
    names = [str(n) for n in tools.load_employees()["name"]]
    assert [p for p in Tracer("r5").prompts() if find_pii(p, names)] == []


# ---- api -----------------------------------------------------------------------------
def client(env, llm=None):
    return TestClient(api.create_app(make_graph(env, llm)))


def test_api_requires_key(env):
    c = client(env)
    assert c.post("/runs", json={}).status_code == 401
    assert c.post("/runs", json={}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_api_fails_closed_without_configured_key(env, monkeypatch):
    monkeypatch.delenv("WFR_API_KEY")
    assert client(env).post("/runs", json={}, headers={"X-API-Key": "x"}).status_code == 503


def test_api_full_flow(env):
    c, h = client(env), {"X-API-Key": "test-key"}
    r = c.post("/runs", json={"roles": ["HR Operations Analyst", "Data Analyst"]}, headers=h).json()
    assert r["awaiting_approval"] and len(r["draft"]["recommendations"]) == 2
    rid = r["run_id"]
    assert c.get(f"/runs/{rid}", headers=h).json()["status"] == "awaiting_approval"
    d = c.post(f"/runs/{rid}/decision", json={"decision": "approve", "reviewer": "Azeem H"}, headers=h).json()
    assert d["status"] == "released"
    # second decision on a finished run is rejected
    assert c.post(f"/runs/{rid}/decision", json={"decision": "approve", "reviewer": "Azeem H"}, headers=h).status_code == 409
    assert c.get(f"/runs/{rid}/trace", headers=h).json()["llm_calls"] >= 1


def test_api_validation(env):
    c, h = client(env), {"X-API-Key": "test-key"}
    assert c.post("/runs", json={"roles": ["Nope"]}, headers=h).status_code == 422
    assert c.get("/runs/doesnotexist", headers=h).status_code == 404
    assert c.post("/runs/x/decision", json={"decision": "maybe", "reviewer": "A"}, headers=h).status_code == 422
