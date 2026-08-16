import httpx
from app.core.config import Settings
from app.services.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a grounded retrieval assistant. Use only the supplied evidence.
Every factual claim must cite [C1], [C2], etc. Do not invent facts or citations.
If evidence is insufficient, state that clearly."""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, question: str, evidence: list[RetrievedChunk]) -> str:
        if not evidence:
            return "I do not have enough retrieved evidence to answer that question."
        blocks = []
        for i, item in enumerate(evidence, 1):
            loc = item.source + (f", page {item.page}" if item.page is not None else "")
            blocks.append(f"[C{i}] Source: {loc}\n{item.text}")
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question:\n{question}\n\nEvidence:\n\n" + "\n\n".join(blocks)},
            ],
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
