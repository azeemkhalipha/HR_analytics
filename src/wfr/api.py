"""FastAPI service. Auth via X-API-Key; the key comes from the environment (a secrets manager in prod), never code.

Fails CLOSED: if WFR_API_KEY is not set, every protected route returns 503.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import graph as g
from .llm import get_llm
from .schemas import Approval
from .tracing import Tracer


class RunRequest(BaseModel):
    roles: list[str] | None = None


def create_app(graph=None) -> FastAPI:
    app = FastAPI(title="Workforce Reinvention Copilot", version="0.1.0")
    state = {"graph": graph}

    def _graph():
        if state["graph"] is None:
            state["graph"] = g.build_graph(get_llm())
        return state["graph"]

    def auth(x_api_key: str | None = Header(default=None)):
        expected = os.environ.get("WFR_API_KEY")
        if not expected:
            raise HTTPException(503, "server not configured: WFR_API_KEY missing")
        if not x_api_key or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(401, "invalid api key")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/runs", dependencies=[Depends(auth)])
    def create_run(req: RunRequest):
        if req.roles:
            from .tools import load_taxonomy
            unknown = [r for r in req.roles if r not in load_taxonomy()]
            if unknown:
                raise HTTPException(422, f"unknown roles: {unknown}")
        out = g.start_run(_graph(), req.roles)
        return _view(out["run_id"], out["state"])

    @app.get("/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str):
        snap = g.get_state(_graph(), run_id)
        if not snap["values"]:
            raise HTTPException(404, "run not found")
        return _view(run_id, snap["values"], snap["next"])

    @app.post("/runs/{run_id}/decision", dependencies=[Depends(auth)])
    def decide(run_id: str, approval: Approval):
        snap = g.get_state(_graph(), run_id)
        if not snap["values"]:
            raise HTTPException(404, "run not found")
        if "human_gate" not in snap["next"]:
            raise HTTPException(409, f"run is not awaiting approval (status={snap['values'].get('status')})")
        out = g.resume_run(_graph(), run_id, approval.model_dump())
        return _view(run_id, out)

    @app.get("/runs/{run_id}/trace", dependencies=[Depends(auth)])
    def trace(run_id: str):
        return Tracer(run_id).summary()

    return app


def _view(run_id: str, st: dict, nxt: list[str] | None = None) -> dict:
    return {"run_id": run_id, "status": st.get("status"), "error": st.get("error"),
            "draft": st.get("report"), "flags": st.get("flags"),
            "approval": st.get("approval"), "released_path": st.get("released_path"),
            "awaiting_approval": st.get("status") == "awaiting_approval"}


app = create_app()
