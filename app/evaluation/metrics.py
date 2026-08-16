def hit_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    expected = {s.lower() for s in expected_sources}
    return float(any(source.lower() in expected for source in retrieved_sources))


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    expected = {s.lower() for s in expected_sources}
    for rank, source in enumerate(retrieved_sources, 1):
        if source.lower() in expected:
            return 1.0 / rank
    return 0.0


def keyword_recall(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    lowered = answer.lower()
    return sum(term.lower() in lowered for term in expected_terms) / len(expected_terms)


def citation_present(answer: str) -> float:
    return float('[C' in answer and ']' in answer)
