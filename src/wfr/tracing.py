"""Minimal tracer: one JSONL file per run, spans with latency, token and cost estimates.

Deliberately dependency-free. To ship spans to Langfuse/OpenTelemetry, wrap `Tracer.span`.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

# USD per 1M tokens (input, output). Illustrative; set real prices via env/config.
PRICES = {"mock": (0.0, 0.0), "haiku": (0.80, 4.00), "default": (3.0, 15.0)}


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Tracer:
    def __init__(self, run_id: str, root: str | None = None):
        self.run_id = run_id
        self.dir = Path(root or os.environ.get("WFR_TRACE_DIR", "traces"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"

    def _write(self, rec: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        rec = {"run_id": self.run_id, "span": name, "attrs": attrs, "status": "ok"}
        try:
            yield rec
        except Exception as e:  # record then re-raise
            rec["status"] = "error"
            rec["error"] = repr(e)
            raise
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self._write(rec)

    def llm_call(self, model: str, prompt: str, output: str, latency_ms: float, ok: bool, attempt: int):
        pin, pout = PRICES.get(model if model in PRICES else "default")
        tin, tout = approx_tokens(prompt), approx_tokens(output)
        self._write({
            "run_id": self.run_id, "span": "llm_call", "model": model, "attempt": attempt,
            "valid": ok, "latency_ms": round(latency_ms, 2), "tokens_in": tin, "tokens_out": tout,
            "cost_usd": round((tin * pin + tout * pout) / 1e6, 6), "prompt": prompt,
        })

    def summary(self) -> dict:
        spans = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        llm = [s for s in spans if s["span"] == "llm_call"]
        return {
            "run_id": self.run_id,
            "spans": len(spans),
            "llm_calls": len(llm),
            "llm_retries": sum(1 for s in llm if not s["valid"]),
            "tokens_in": sum(s["tokens_in"] for s in llm),
            "tokens_out": sum(s["tokens_out"] for s in llm),
            "cost_usd": round(sum(s["cost_usd"] for s in llm), 6),
            "total_latency_ms": round(sum(s["latency_ms"] for s in spans if s["span"] != "llm_call"), 1),
            "errors": [s for s in spans if s.get("status") == "error"],
        }

    def prompts(self) -> list[str]:
        if not self.path.exists():
            return []
        return [json.loads(l)["prompt"] for l in self.path.read_text().splitlines() if '"llm_call"' in l]
