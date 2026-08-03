"""Reusable contract checks for third-party plugins."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Checkpoint, Chunk, Document, TokenBudget
from .protocols import Chunker, SourceConnector, Tokenizer, VectorStore


def assert_chunker_contract(chunker: Chunker, tokenizer: Tokenizer) -> None:
    document = Document(text="alpha beta gamma delta", source_uri="contract://chunker")
    budget = TokenBudget(tokenizer=tokenizer.name, max_tokens=2, overlap_tokens=0)
    first = tuple(chunker.split(document, budget, tokenizer))
    second = tuple(chunker.split(document, budget, tokenizer))
    assert first == second, "chunkers must be deterministic for identical inputs"
    assert first, "chunker produced no spans for non-empty text"
    for start, end, _ in first:
        assert 0 <= start < end <= len(document.text), "chunk span is outside the document"
        assert tokenizer.count(document.text[start:end]) <= budget.max_tokens, (
            "chunk exceeds the declared budget"
        )


async def assert_connector_contract(connector: SourceConnector) -> None:
    events = [event async for event in connector.sync(None)]
    assert events, "connector emitted no events"
    assert isinstance(events[-1], Checkpoint), (
        "connector must finish a successful page with a checkpoint"
    )


async def assert_vectorstore_contract(
    store: VectorStore, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
) -> None:
    assert len(chunks) == len(vectors), "fixture chunks and vectors differ in length"
    await store.upsert(chunks, vectors)
    await store.delete([chunk.id for chunk in chunks])
