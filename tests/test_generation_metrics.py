from app.evaluation.generation_metrics import (
    citation_coverage,
    citation_ids,
    citation_validity,
    forbidden_term_rate,
    keyword_recall,
    prompt_injection_leak,
    refusal_exact,
)
from app.services.llm import REFUSAL_TEXT


def test_citation_ids_and_validity() -> None:
    answer = "The policy is 400 days [C1]. The legacy value is retired [C2]."
    assert citation_ids(answer) == ["C1", "C2"]
    assert citation_validity(answer, ["C1", "C2"]) == 1.0
    assert citation_validity("Claim [C9]", ["C1"]) == 0.0


def test_citation_coverage() -> None:
    assert citation_coverage("Supported claim [C1]", True) == 1.0
    assert citation_coverage("Unsupported claim", True) == 0.0
    assert citation_coverage("No citation needed", False) == 1.0


def test_refusal_and_keyword_scoring() -> None:
    assert refusal_exact(REFUSAL_TEXT) == 1.0
    assert refusal_exact("I am not sure.") == 0.0
    assert keyword_recall("The target is 12 minutes.", ["12", "minutes"]) == 1.0


def test_attack_and_forbidden_detection() -> None:
    assert forbidden_term_rate("The quota is unlimited.", ["unlimited"]) == 1.0
    assert prompt_injection_leak("PWNED", ["pwned"]) == 1.0
    assert prompt_injection_leak("The quota is 1,200 rpm.", ["pwned"]) == 0.0
