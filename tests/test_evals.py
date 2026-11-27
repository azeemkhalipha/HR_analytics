import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import run_evals  # noqa: E402


def test_no_eval_regression():
    m = run_evals.compute()
    base = json.loads(run_evals.BASELINE.read_text())
    assert run_evals.check(m, base) == []
