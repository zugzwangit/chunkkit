"""Retrieval-time deduplication and model-aware context packing."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from .models import (
    Chunk,
    Citation,
    ContextBundle,
    DroppedChunk,
    ModelTarget,
    ScoredChunk,
)
from .protocols import Tokenizer
from .tokenizers import create_tokenizer


class ContextPacker:
    def __init__(
        self,
        target: ModelTarget,
        *,
        tokenizer: Tokenizer | None = None,
        separator: str = "\n\n",
    ) -> None:
        self.target = target
        self.tokenizer = tokenizer or create_tokenizer(target.tokenizer, model=target.model)
        self.separator = separator

    def assemble(self, chunks: Sequence[ScoredChunk | Chunk]) -> ContextBundle:
        ranked = sorted(
            chunks,
            key=lambda item: item.score if isinstance(item, ScoredChunk) else 0.0,
            reverse=True,
        )
        included: list[Chunk] = []
        dropped: list[DroppedChunk] = []
        seen: set[str] = set()
        parts: list[str] = []
        current = 0
        separator_tokens = self.tokenizer.count(self.separator)

        for item in ranked:
            chunk = item.chunk if isinstance(item, ScoredChunk) else item
            digest = sha256(chunk.text.strip().encode()).hexdigest()
            count = self.tokenizer.count(chunk.text)
            if not chunk.text.strip():
                dropped.append(DroppedChunk(chunk_id=chunk.id, reason="empty", token_count=count))
                continue
            if chunk.id in seen or digest in seen:
                dropped.append(
                    DroppedChunk(chunk_id=chunk.id, reason="duplicate", token_count=count)
                )
                continue
            additional = count + (separator_tokens if parts else 0)
            if current + additional > self.target.available_tokens:
                dropped.append(DroppedChunk(chunk_id=chunk.id, reason="budget", token_count=count))
                continue
            seen.update((chunk.id, digest))
            included.append(chunk)
            parts.append(chunk.text)
            current += additional

        return ContextBundle(
            text=self.separator.join(parts),
            chunks=tuple(included),
            dropped=tuple(dropped),
            citations=tuple(
                Citation(chunk_id=chunk.id, source_uri=chunk.source_uri, spans=chunk.spans)
                for chunk in included
            ),
            token_count=current,
            max_tokens=self.target.available_tokens,
        )


ContextAssembler = ContextPacker
