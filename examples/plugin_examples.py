"""Small protocol implementations suitable as plugin starting points."""

from collections.abc import AsyncIterator, Sequence

from chunkkit import Checkpoint, Chunk, Document, TokenBudget
from chunkkit.protocols import Tokenizer


class PipeTokenizer:
    name = "pipe"

    def spans(self, text: str) -> Sequence[tuple[int, int]]:
        spans = []
        cursor = 0
        for part in text.split("|"):
            spans.append((cursor, cursor + len(part)))
            cursor += len(part) + 1
        return tuple(spans)

    def count(self, text: str) -> int:
        return len(self.spans(text)) if text else 0


class WholeDocumentChunker:
    name = "whole"

    def split(self, document: Document, budget: TokenBudget, tokenizer: Tokenizer):  # type: ignore[no-untyped-def]
        if tokenizer.count(document.text) <= budget.max_tokens:
            yield 0, len(document.text), {}


class EmptyConnector:
    name = "empty"

    async def sync(self, cursor: str | None = None) -> AsyncIterator[Checkpoint]:
        yield Checkpoint(cursor=cursor or "complete")


class MemoryVectorStore:
    def __init__(self) -> None:
        self.values: dict[str, tuple[Chunk, tuple[float, ...]]] = {}

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.values[chunk.id] = (chunk, tuple(vector))

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self.values.pop(chunk_id, None)
