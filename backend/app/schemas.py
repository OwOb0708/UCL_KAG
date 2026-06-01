from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    top_k: int = 6


class Citation(BaseModel):
    source: str
    excerpt: str
    score: float | None = None


class GraphFact(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    graph_facts: list[GraphFact] = []
    trace: list[str] = []


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    indexing_status: str
    indexed_docs: int = 0
