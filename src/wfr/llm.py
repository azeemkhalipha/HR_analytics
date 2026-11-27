"""LLM layer: provider-agnostic client + structured output with validate/retry.

Providers
- MockLLM     : deterministic, offline. Used by tests/evals/CI. Can inject failures to test retry.
- BedrockLLM  : Claude on AWS Bedrock via Converse API with a FORCED tool call, so the model must
                return JSON matching the Pydantic schema (structured output through tool calling).
Select with WFR_LLM=mock|bedrock. Credentials come from the standard AWS chain (IAM role / env),
never from code.
"""
from __future__ import annotations

import json
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .tracing import Tracer

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: str, prompt: str, schema: type[BaseModel]) -> str: ...


class MockLLM:
    """Deterministic stand-in. Builds a schema-valid answer from the <facts> block in the prompt."""

    model = "mock"

    def __init__(self, fail_first_n: int = 0, hallucinate_ids: bool = False):
        self.fail_first_n = fail_first_n
        self.hallucinate_ids = hallucinate_ids
        self.calls = 0

    def complete_json(self, system, prompt, schema):
        self.calls += 1
        if self.calls <= self.fail_first_n:
            return '{"headline": "broken json'  # simulates truncated/invalid model output
        facts = json.loads(prompt.split("<facts>")[1].split("</facts>")[0])
        if schema.__name__ == "Report":
            return json.dumps(self._report(facts))
        raise LLMError(f"MockLLM has no handler for {schema.__name__}")

    def _report(self, facts: dict) -> dict:
        recs = []
        for r in facts["roles"]:
            pr = "high" if r["exposure"] >= 0.55 else "medium" if (r["exposure"] >= 0.45 or r["mean_risk"] >= 0.28) else "low"
            moves = [m["to_role"] for m in r["moves"]]
            ids = r["priority_employee_ids"][:5]
            if self.hallucinate_ids:
                ids = ids + ["E9999"]
            recs.append({
                "role": r["role"], "priority": pr,
                "action": (f"Start reskilling {r['role']} staff toward {', '.join(moves) or 'adjacent roles'}; "
                           f"{r['automate_tasks']} tasks are strong automation candidates."),
                "suggested_moves": moves, "employee_ids_to_prioritise": ids,
                "evidence": f"Exposure {r['exposure']:.2f}, mean attrition risk {r['mean_risk']:.2f}, headcount {r['headcount']}."})
        return {"headline": f"{sum(x['priority']=='high' for x in recs)} roles need immediate reinvention planning",
                "recommendations": recs,
                "caveats": ["Synthetic demo data; scores are illustrative.",
                            "Recommendations require human approval before release."]}


class BedrockLLM:
    def __init__(self, model_id: str | None = None, region: str | None = None):
        import boto3  # imported lazily so mock mode needs no AWS config

        self.model = model_id or os.environ.get("WFR_BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def complete_json(self, system, prompt, schema):
        tool = {"toolSpec": {"name": "emit", "description": f"Return a {schema.__name__}",
                             "inputSchema": {"json": schema.model_json_schema()}}}
        resp = self.client.converse(
            modelId=self.model, system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig={"tools": [tool], "toolChoice": {"tool": {"name": "emit"}}},
            inferenceConfig={"maxTokens": 2000, "temperature": 0.0})
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                return json.dumps(block["toolUse"]["input"])
        raise LLMError("no tool call in model response")


def get_llm() -> LLMClient:
    provider = os.environ.get("WFR_LLM", "mock")
    if provider == "mock":
        return MockLLM()
    if provider == "bedrock":
        return BedrockLLM()
    raise LLMError(f"unknown WFR_LLM={provider}")


def structured_call(llm: LLMClient, schema: type[T], system: str, prompt: str, tracer: Tracer,
                    max_attempts: int = 3, extra_validator=None) -> T:
    """Call the model, validate against the schema, retry with the error fed back.

    extra_validator(obj) -> list[str] lets callers add semantic checks (e.g. grounding);
    any returned messages count as a validation failure and trigger a retry.
    """
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        p = prompt if not last_err else f"{prompt}\n\nYour previous answer was rejected: {last_err}\nReturn corrected JSON only."
        t0 = time.perf_counter()
        raw = llm.complete_json(system, p, schema)
        ms = (time.perf_counter() - t0) * 1000
        try:
            obj = schema.model_validate_json(raw)
            problems = extra_validator(obj) if extra_validator else []
            if problems:
                raise ValueError("; ".join(problems))
            tracer.llm_call(llm.model, p, raw, ms, True, attempt)
            return obj
        except (ValidationError, ValueError) as e:
            last_err = str(e)[:500]
            tracer.llm_call(llm.model, p, raw, ms, False, attempt)
    raise LLMError(f"no valid {schema.__name__} after {max_attempts} attempts: {last_err}")
