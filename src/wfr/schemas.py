"""Pydantic contracts for every agent output. Anything the LLM returns must pass these."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RoleRisk(BaseModel):
    role: str
    headcount: int
    mean_attrition_risk: float = Field(ge=0, le=1)
    high_risk_count: int = Field(ge=0)
    top_drivers: list[str]


class RiskResult(BaseModel):
    model_auc_cv: float
    threshold: float
    roles: list[RoleRisk]
    high_risk_employee_ids: list[str]


class TaskImpact(BaseModel):
    task: str
    share: float
    automation_score: float = Field(ge=0, le=1)
    treatment: Literal["automate", "augment", "retain"]
    rationale: str = Field(min_length=10)


class RoleImpact(BaseModel):
    role: str
    exposure: float = Field(ge=0, le=1)
    tasks: list[TaskImpact]


class ImpactResult(BaseModel):
    roles: list[RoleImpact]


class AdjacentMove(BaseModel):
    from_role: str
    to_role: str
    skill_overlap: float = Field(ge=0, le=1)
    gap_skills: list[str]


class AdjacencyResult(BaseModel):
    moves: list[AdjacentMove]


class Recommendation(BaseModel):
    role: str
    priority: Literal["high", "medium", "low"]
    action: str = Field(min_length=15)
    suggested_moves: list[str]
    employee_ids_to_prioritise: list[str]
    evidence: str = Field(min_length=10)

    @field_validator("suggested_moves")
    @classmethod
    def _moves_nonempty_strings(cls, v):
        return [m for m in v if m.strip()]


class Report(BaseModel):
    headline: str = Field(min_length=10)
    recommendations: list[Recommendation] = Field(min_length=1)
    caveats: list[str] = Field(min_length=1)


class Approval(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=2)
    comment: str = ""
    edited_headline: str | None = None
