from app.services.demo_rag import DemoRAGService
from app.services.llm import REFUSAL_TEXT


def test_demo_answers_supported_security_retention_question():
    result = DemoRAGService().query("How long are enterprise security events retained?", 4)

    assert "400 days" in result.answer
    assert "[C1]" in result.answer
    assert result.evidence
    assert result.evidence[0].source == "security-retention.md"


def test_demo_refuses_unsupported_question():
    result = DemoRAGService().query("What is the company's annual revenue?", 4)

    assert result.answer == REFUSAL_TEXT


def test_demo_prefers_current_maintenance_policy():
    result = DemoRAGService().query("What is the current emergency maintenance notice period?", 4)

    assert "60 minutes" in result.answer
    assert any(item.source == "current-maintenance-policy.md" for item in result.evidence)
