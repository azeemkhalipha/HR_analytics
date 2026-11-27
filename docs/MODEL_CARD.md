# Model & System Card: Workforce Reinvention Copilot

**Status:** demo / portfolio. Not for decisions about real employees.

## Components
| Component | Type | Role |
|---|---|---|
| Attrition model | Logistic regression (scikit-learn), 5-fold out-of-fold scores | Per-employee risk, aggregated by role |
| Task exposure | Rule-based over hand-assigned task automation scores | Role exposure = sum(task share x automation score) |
| Adjacency | Jaccard skill overlap, lower-exposure targets only, min overlap 0.15 | Candidate reskilling moves + skill gaps |
| Report writer | LLM (Claude on Bedrock; deterministic mock offline) | Turns facts into a validated `Report` |

## Intended use
Prioritising workforce-planning conversations at the role level. Output is a *draft for a human reviewer*, never an automated decision.

## Out of scope
Hiring, firing, promotion, pay or performance decisions about individuals. Scoring real employees without a lawful basis and works-council/consent review.

## Data
Synthetic: 600 employees, seeded generator (`scripts/make_data.py`); attrition label from a hidden logistic function plus noise. Gender and age are generated but are **not model inputs**.

## Controls
- **PII:** direct identifiers (name, email, phone) are dropped before modelling; the LLM receives employee IDs and aggregates only. Every run scans all prompts for identifiers and reports a leak count (target 0).
- **Grounding:** every employee ID, role and move in the report must exist in the input facts; otherwise the output is rejected and retried, then the run fails (no release).
- **Human oversight:** a named reviewer must approve; rejection releases nothing; release is idempotent.
- **Bias audit:** outcome audit across gender and age band; four-fifths ratio plus an omnibus chi-square test; verdict pass / review / fail shown to the reviewer.
- **Traceability:** per-run JSONL trace (prompts, tokens, cost estimate, latency, retries).

## Known results (synthetic data)
- CV AUC 0.77.
- Bias audit: **review** (gender ratio 0.68, p = 0.08; age-band ratio 0.48, p = 0.10; smallest group n = 30). Not significant, but low ratios plus small groups mean the audit cannot show the model is fair. Needs more data before any real use.

## Risks and mitigations not yet implemented
Access control and audit log for who viewed which scores; data-retention policy for traces (prompts are stored); differential privacy or k-anonymity thresholds for small roles; live-model evals (LLM-as-judge / human-rated) and drift monitoring.
