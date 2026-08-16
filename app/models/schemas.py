from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


class Evidence(BaseModel):
    citation_id: str
    chunk_id: str
    source: str
    page: int | None = None
    score: float
    text: str


class RetrieveResponse(BaseModel):
    question: str
    evidence: list[Evidence]
    retrieval_ms: float


class QueryResponse(RetrieveResponse):
    answer: str
    generation_ms: float
    total_ms: float


class IngestResponse(BaseModel):
    document_id: str
    source: str
    chunks_indexed: int
